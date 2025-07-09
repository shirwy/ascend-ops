import torch
import torch_npu

import ascend910a_extras.ascend910a_extras_C as _C


def swiglu(x: torch.Tensor) -> torch.Tensor:
    return _C.swiglu(x)


def grouped_matmul(
    x: torch.Tensor, w: torch.Tensor, group_list: torch.Tensor
) -> torch.Tensor:
    return _C.grouped_matmul(x, w, group_list)


def print_info():
    device_id = torch.npu.current_device()
    _C.print_info(device_id)
