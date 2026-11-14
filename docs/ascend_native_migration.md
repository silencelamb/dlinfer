# LMDeploy Ascend patch 退出 dlinfer

本分支与 [silencelamb/lmdeploy 的 codex/ascend-a3-native](https://github.com/silencelamb/lmdeploy/tree/codex/ascend-a3-native) 配套使用。LMDeploy 原生 backend 已承接 A3 推理所需适配，本仓库不再维护第二套 import-time LMDeploy Ascend patch。**截至 2026-11-14，已补充 A3 独立算子验证；MACA/CAMB 实机验证仍未执行。**

## A3 实测摘要

清理后的实测 commit 为 `05ede38ae879347be33c8f6ebb136d84f3f4ee88`，旧基线为下文 `3b909634`。本次发布只追加验证说明，保留该实测计算代码。

- 独立环境未安装或导入 LMDeploy；framework 边界 CPU suite 为 16 passed。
- 1056 专家、BF16、N32、hidden/intermediate64 的合成 MoE eager 与真实 NPUGraph capture/replay 共 6 项通过；更换 routing 后与 CPU reference 最大差为 0。这不代表完整大模型 checkpoint 性能通过。
- standalone 算子 suite 为 25 passed / 3 failed：通过项含 16 个 CPU ring reference 和 9 个真实 NPU case；三个 paged/decode MLA case 与旧基线同样遇到 CANN keyRope 非连续错误。
- 镜像 Python 3.12 / torch 2.10 超出部分依赖声明，pip check 非零；没有因此修改声明上界。独立 Transformers、DICP/ATB compiler 及其他硬件仍未验证。

详细数据、命令及环境在配套 fork 的 [A3 验证文档分支](https://github.com/silencelamb/lmdeploy/tree/docs/ascend-a3-validation-20261114/docs/zh_cn/advance/ascend_a3_validation)。下文 Mac 记录是先前阶段的检查，不覆盖本段新增硬件证据。

删除前的 dlinfer 基线为 `3b9096341a41e9d067104a39d914e47ca2d7ec50`。下表“原位置”均可用 `git show 3b909634:<文件>` 查看，不能把它当作清理后 HEAD 的现存文件。

| 清理内容 | 原位置 | 当前归属 |
| --- | --- | --- |
| Qwen3.5/GDN、GLM/DeepSeek、ModelSlim、rejection 等模型 patch | `dlinfer/framework/lmdeploy_ext/device/__init__.py:12`、`:140`、`:171`、`:289`、`:952` | LMDeploy 的显式模型入口与 `backends/ascend`；本文件仅保留其他厂商还需要的公共 helper |
| ChatGLM RoPE、bad words、MoE DP/TP、NPU profiler | `dlinfer/framework/lmdeploy_ext/device/ascend.py:38`、`:52`、`:101`、`:382` | LMDeploy native backend；删除原文件 |
| Ascend graph runner 和 CudaGraphMixin 覆盖 | `dlinfer/framework/lmdeploy_ext/cudagraph/ascend_cudagraph.py:1` | LMDeploy `backends/ascend/graph_runner.py`；删除原文件及旧注册 |
| AWQ 构造、转换与 loader 覆盖 | `dlinfer/framework/lmdeploy_ext/quants/ascend_awq.py:17` | LMDeploy native quantization helper 和共享 loader 的明确分支；删除只服务 Ascend 的 quants 包 |
| graph NSA/lifecycle 测试 | `tests/test_ascend_cudagraph_nsa.py:10`、`tests/test_ascend_cudagraph_lifecycle.py:10` | 已在配套 LMDeploy 的 `tests/pytorch/ascend/` 接续，删除旧副本 |
| 旧 allocator patch 测试 | `tests/test_lmdeploy_cache_allocator.py:7` | 原导入函数在基线中已不存在；删除旧测试，在 LMDeploy 原生 cache suite 补充混合 dtype、小池与写入隔离回归 |

保留旧 patch 会再次覆盖 LMDeploy 原生模型和 graph 方法，造成两边重复维护和版本耦合。现在 Ascend 的 `lmdeploy_ext` 导入只保留兼容 namespace，不导入 framework patch；MACA/CAMB/PPU 的原分派继续存在。代价是旧版 LMDeploy 不能再依靠本分支获得 Ascend 适配，须使用配套原生版本；其他推理引擎未核实。位置：`dlinfer/framework/lmdeploy_ext/__init__.py:2`、`device/__init__.py:1`、`cudagraph/__init__.py:6`。

## 保留的代码和依赖

- `dlinfer/vendor/ascend` 的独立算子、Triton kernel、类型及算子注册保留；这些仍属于 dlinfer 对外算子接口，不因 LMDeploy 不再依赖 dlinfer 而删除。对应 `dlinfer/ops/llm.py:1`、`dlinfer/vendor/ascend/torch_npu_ops.py:1`。
- `dlinfer/vendor/ascend/pytorch_patch.py` 中的 torch_npu compatibility shim、编译兼容设置及 ATB 扩展加载保留，供独立 vendor 路径使用。这里的初始化仍有进程级副作用，不能把“framework 入口不再安装 LMDeploy patch”理解为“import dlinfer 完全没有 patch”。位置：`dlinfer/vendor/ascend/pytorch_patch.py:10`、`:30`。
- 独立 Transformers 适配、DICP/ATB graph compiler 与其显式 graph-mode 适配未迁移，不在此次删除范围。Ascend 的普通 `lmdeploy_ext` 入口不再自动安装 Transformers hook。位置：`dlinfer/framework/transformers_ext/__init__.py:11`、`dlinfer/graph/dicp/vendor/AtbGraph/__init__.py:1`。
- MACA/CAMB/PPU 的 device/graph 实现保持原文件内容；共同使用的 compilation/sampling helper 保留。位置：`dlinfer/framework/lmdeploy_ext/device/camb.py:1`、`dlinfer/framework/lmdeploy_ext/cudagraph/maca_cudagraph.py:1`、`camb_cudagraph.py:1`、`ppu_cudagraph.py:1`。

独立 MoE 算子曾为查询 capture 状态而导入旧 AscendGraphRunner，间接触发整个 LMDeploy patch 包。现在使用 `torch.npu.is_current_stream_capturing()`，保留原 eager/catchall 两套计算和 `DLINFER_MOE_PREFILL_CATCHALL` 开关；上述 A3 合成数据已验证真实捕获与 replay。位置：`dlinfer/vendor/ascend/moe.py:335`。基础依赖接口可在 [torch_npu 2.9 源码](https://github.com/Ascend/pytorch/blob/v2.9.0/torch_npu/npu/graphs.py#L62) 核对。

## 检查与配套使用

Mac 上执行 `python -m pytest -q tests/test_native_lmdeploy_migration.py`，**16 passed**。测试隔离硬件初始化，检查 framework 导入边界、其他厂商路由，以及 MoE capture/eager 选择和累计 token counts 的转换；不代表真实 NPU graph 或 GMM 执行通过。

另外按 AST 核对了保留的 6 个公共 helper，函数体与基线一致；其他厂商的 device/graph 文件逐字节未改。配套 LMDeploy 的原生 CPU suite 为 **31 passed**，含接续的 allocator 回归。

静态编译、Black 格式检查和修改文件的 Ruff 错误检查通过。sdist 与从 sdist 构建的 wheel 也完成检查，使用：

```bash
DEVICE=ascend DLINFER_BUILD_DICP=OFF python -m build \
  --no-isolation --skip-dependency-check --outdir .git/ascend-dist
```

这里关闭了可选 DICP 构建，并跳过 Mac 无法安装的 NPU build dependency 检查；没有修改生产构建逻辑。产物是 **Mac 的打包检查产物**，不是 A3 可部署 wheel。Mac 阶段的 Python 3.12 用于 CPU/打包检查，dlinfer 的运行时 Python 约束仍以 `setup.py:125` 的 `>=3.10,<3.12` 为准；后来 A3 镜像也使用 Python 3.12，其超出声明的限制已在文首单列。受支持环境下的重新构建仍待完成。

A3 推理直接安装配套 LMDeploy，不需要安装本 dlinfer 分支。独立 dlinfer 算子用户继续按原安装方法准备 torch_npu/CANN/NNAL；不得把已删除的 LMDeploy Ascend patch 补回入口。两套环境分开验证，完整步骤见 [LMDeploy 迁移与硬件验收说明](https://github.com/silencelamb/lmdeploy/blob/codex/ascend-a3-native/docs/zh_cn/advance/ascend_a3_migration.md)。

后续扩大 A3 验证时，继续运行 attention/RoPE/SFA tests，并增加 >1024 专家模型的实际 shape 和负载；已执行的小维度 MoE 测试范围见文首。旧 LMDeploy + 旧 dlinfer 的参考环境须固定上面的迁移前 commit。
