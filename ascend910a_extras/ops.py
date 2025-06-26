import torch
import torch_npu
import ascend910a_extras.ascend910a_extras_C


def swiglu(x: torch.Tensor) -> torch.Tensor:
  return torch.ops.ascend910a.swiglu(x)