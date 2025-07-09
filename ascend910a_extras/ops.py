import torch
import torch_npu

import ascend910a_extras.ascend910a_extras_C as _C


def swiglu(x: torch.Tensor) -> torch.Tensor:
    # return torch.ops.ascend910a.swiglu(x)
    return _C.swiglu(x)


def grouped_matmul(
    x: torch.Tensor, w: torch.Tensor, group_list: torch.Tensor
) -> torch.Tensor:
    return torch.ops.ascend910a.grouped_matmul(x, w, group_list)


def matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.ops.ascend910a.matmul(x, w)


def print_info():
    device_id = torch.npu.current_device()
    _C.print_info(device_id)


def graph_run(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _C.graph_run(x, w)


def mlp(
    x: torch.Tensor, gate_up_proj: torch.Tensor, down_proj: torch.Tensor
) -> torch.Tensor:
    return _C.mlp(x, gate_up_proj, down_proj)


def graph_swiglu(x: torch.Tensor) -> torch.Tensor:
    return _C.graph_swiglu(x)
