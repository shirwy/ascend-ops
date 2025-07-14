import torch
import torch_npu

import ascend910a_extras.ascend910a_extras_C as _C


def swiglu(x: torch.Tensor) -> torch.Tensor:
    return _C.ops.swiglu(x)


def grouped_matmul(
    x: torch.Tensor, w: torch.Tensor, group_list: torch.Tensor
) -> torch.Tensor:
    return _C.ops.grouped_matmul(x, w, group_list)


def add_rms_norm(
    x: torch.Tensor, 
    residual: torch.Tensor, 
    weight: torch.Tensor, 
    epsilon: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    return _C.ops.add_rms_norm(x, residual, weight, epsilon)


def print_info():
    device_id = torch.npu.current_device()
    _C.print_info(device_id)
