import torch
import torch_npu

import ascend910a_extras.ascend910a_extras_C


def swiglu(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.ascend910a.swiglu(x)


def grouped_matmul(
    x: torch.Tensor, w: torch.Tensor, group_list: torch.Tensor
) -> torch.Tensor:
    return torch.ops.ascend910a.grouped_matmul(x, w, group_list)


def matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.ops.ascend910a.matmul(x, w)
