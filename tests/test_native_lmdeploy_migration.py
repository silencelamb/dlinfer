# Copyright (c) 2026, DeepLink. All rights reserved.
"""CPU tests for retiring framework patches, not NPU kernel validation."""

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _package(monkeypatch, name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    return module


@pytest.fixture
def vendor_namespace(monkeypatch):
    # Isolate the framework entry point from hardware initialization in
    # dlinfer.__init__. Each test still imports the real integration module.
    for name in list(sys.modules):
        if name == "dlinfer" or name.startswith("dlinfer."):
            monkeypatch.delitem(sys.modules, name)
    _package(monkeypatch, "dlinfer", ROOT / "dlinfer")
    vendor = _package(monkeypatch, "dlinfer.vendor", ROOT / "dlinfer/vendor")
    _package(monkeypatch, "dlinfer.framework", ROOT / "dlinfer/framework")
    return vendor


class _BlockFrameworkImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(("lmdeploy", "transformers")) or fullname in (
            "dlinfer.framework.transformers_ext",
            "dlinfer.framework.torch_npu_ext",
        ):
            raise AssertionError("Unexpected framework import: " + fullname)


def test_ascend_legacy_entry_does_not_load_framework_patches(
    monkeypatch, vendor_namespace
):
    vendor_namespace.vendor_name = "ascend"
    monkeypatch.setattr(sys, "meta_path", [_BlockFrameworkImports(), *sys.meta_path])
    compile_before = torch.compile
    importlib.import_module("dlinfer.framework.lmdeploy_ext")
    # Explicit subpackage imports must not reactivate the retired patches.
    device = importlib.import_module("dlinfer.framework.lmdeploy_ext.device")
    graph = importlib.import_module("dlinfer.framework.lmdeploy_ext.cudagraph")
    assert "ascend" not in device.vendor
    assert "ascend" not in graph.graph_vendor
    assert torch.compile is compile_before


@pytest.mark.parametrize("vendor_name", ["maca", "camb", "ppu"])
def test_other_vendors_keep_the_legacy_entry(
    monkeypatch, vendor_namespace, vendor_name
):
    vendor_namespace.vendor_name = vendor_name
    # Stub hardware-owning children to check only the real entry's dispatch.
    children = {}
    for suffix in (
        "transformers_ext",
        "torch_npu_ext",
        "lmdeploy_ext.cudagraph",
        "lmdeploy_ext.device",
    ):
        name = "dlinfer.framework." + suffix
        children[suffix] = _package(monkeypatch, name, ROOT / name.replace(".", "/"))
    module = importlib.import_module("dlinfer.framework.lmdeploy_ext")
    assert module.cudagraph is children["lmdeploy_ext.cudagraph"]
    assert module.device is children["lmdeploy_ext.device"]


@pytest.mark.parametrize("vendor_name", ["ascend", "maca", "camb", "ppu"])
def test_graph_dispatch_keeps_remaining_vendor_modules(
    monkeypatch, vendor_namespace, vendor_name
):
    vendor_namespace.vendor_name = vendor_name
    _package(
        monkeypatch,
        "dlinfer.framework.lmdeploy_ext",
        ROOT / "dlinfer/framework/lmdeploy_ext",
    )
    real_import = importlib.import_module
    calls = []

    def record_import(name, package=None):
        if package == "dlinfer.framework.lmdeploy_ext.cudagraph":
            calls.append(name)
            return SimpleNamespace()
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", record_import)
    real_import("dlinfer.framework.lmdeploy_ext.cudagraph")
    assert calls == ([] if vendor_name == "ascend" else [f".{vendor_name}_cudagraph"])


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("catchall", [False, True])
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_chunked_moe_selects_capture_safe_path_without_lmdeploy(
    monkeypatch, vendor_namespace, capturing, catchall, group_list_type
):
    vendor_namespace.vendor_name = "ascend"
    monkeypatch.setattr(sys, "meta_path", [_BlockFrameworkImports(), *sys.meta_path])
    spec = importlib.util.spec_from_file_location(
        "_standalone_ascend_moe", ROOT / "dlinfer/vendor/ascend/moe.py"
    )
    moe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moe)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=lambda: capturing),
        raising=False,
    )
    monkeypatch.setattr(moe, "MAX_GROUP_LIST_SIZE", 4)
    monkeypatch.setattr(moe, "_MOE_PREFILL_USE_CATCHALL", catchall)
    counts = torch.tensor([1, 0, 2, 0, 1])
    groups = counts.cumsum(0) if group_list_type == 0 else counts
    calls = []

    def record(name):
        def run(hidden, up, down, token_counts, layout):
            calls.append(name)
            torch.testing.assert_close(token_counts, counts)
            return hidden

        return run

    monkeypatch.setattr(moe, "_apply_mlp_chunked_capturable", record("capture"))
    monkeypatch.setattr(moe, "_apply_mlp_chunked_eager", record("eager"))
    hidden = torch.ones(4, 2)
    weights = torch.empty(5, 2, 2)
    result = moe.apply_mlp(hidden, weights, weights, groups, group_list_type)
    assert result is hidden
    assert calls == (["capture"] if capturing or catchall else ["eager"])
