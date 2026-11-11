# Copyright (c) 2024, DeepLink. All rights reserved.
"""Legacy LMDeploy integration for vendors that still use dlinfer.

Ascend integration lives in LMDeploy's native ``backends/ascend`` package.
Importing this compatibility namespace on Ascend does not install framework
patches; use the paired native LMDeploy version instead of an older engine.
"""

from dlinfer.vendor import vendor_name

if vendor_name != "ascend":
    import dlinfer.framework.transformers_ext  # noqa: F401
    import dlinfer.framework.torch_npu_ext  # noqa: F401
    from . import cudagraph, device  # noqa: F401
