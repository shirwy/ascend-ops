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
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    return _C.ops.add_rms_norm(x, residual, weight, epsilon)

def reshape_and_cache(
    key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, slot_indices: torch.Tensor
) -> None:
    if value is None:
        value = torch.empty(0, device=key.device, dtype=key.dtype)
    if value_cache is None:
        value_cache = torch.empty(0, device=key_cache.device, dtype=key_cache.dtype)
    return _C.ops.reshape_and_cache(key, value, key_cache, value_cache, slot_indices)


def print_info():
    device_id = torch.npu.current_device()
    _C.print_info(device_id)


def paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    return _C.ops.paged_attention(q, key_cache, value_cache, block_tables, context_lens)
