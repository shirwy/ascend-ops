import torch

import ascend910a_extras.ops as ops


def reference_reshape_and_cache(
    key, key_cache, slot_indices, value=None, value_cache=None
):
    num_tokens, num_kv_heads, head_size = key.shape
    num_blocks, nh16, block_size, chunk_size = key_cache.shape
    slot_indices_list = slot_indices.tolist()
    for token, slot in enumerate(slot_indices_list):
        block = slot // block_size
        block_offset = slot % block_size
        flat = key[token].flatten()  # shape: [num_kv_heads * head_size]
        idx = 0
        for nh16_idx in range(nh16):
            for j in range(chunk_size):
                if idx < flat.numel():
                    key_cache[block, nh16_idx, block_offset, j] = flat[idx].item()
                    if value is not None and value_cache is not None:
                        value_cache[block, nh16_idx, block_offset, j] = (
                            value[token].flatten()[idx].item()
                        )
                    idx += 1


if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens = 4
    num_blocks = 583
    block_size = 64
    num_kv_heads = 8
    head_size = 128
    nh16 = 128
    slot_indices = torch.tensor([6, 134, 212, 290], dtype=torch.int32)

    # key+value
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float16)
    value = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float16)
    key_cache_cpu = torch.zeros(num_blocks, block_size, nh16, 16, dtype=torch.float16)
    value_cache_cpu = torch.zeros(num_blocks, block_size, nh16, 16, dtype=torch.float16)
    reference_reshape_and_cache(
        key, key_cache_cpu, slot_indices, value, value_cache_cpu
    )
    key_npu = key.to("npu").contiguous()
    value_npu = value.to("npu").contiguous()
    key_cache_npu = torch.zeros_like(key_cache_cpu, device="npu").contiguous()
    value_cache_npu = torch.zeros_like(value_cache_cpu, device="npu").contiguous()
    slot_indices_npu = slot_indices.to("npu").contiguous()
    ops.reshape_and_cache(
        key_npu, value_npu, key_cache_npu, value_cache_npu, slot_indices_npu
    )
    torch.npu.synchronize()
    torch.testing.assert_close(key_cache_cpu, key_cache_npu.cpu(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        value_cache_cpu, value_cache_npu.cpu(), atol=1e-3, rtol=1e-3
    )
    print("PASS: key+value, key_cache and value_cache matched.")

    # only key_cache, no value/value_cache
    key2 = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float16)
    key_cache_cpu2 = torch.zeros(num_blocks, block_size, nh16, 16, dtype=torch.float16)
    value_cache_cpu2 = torch.zeros(
        num_blocks, block_size, nh16, 16, dtype=torch.float16
    )
    slot_indices2 = slot_indices.clone()
    reference_reshape_and_cache(key2, key_cache_cpu2, slot_indices2)
    key2_npu = key2.to("npu").contiguous()
    key_cache2_npu = torch.zeros_like(key_cache_cpu2, device="npu").contiguous()
    value_cache2_npu = torch.zeros_like(value_cache_cpu2, device="npu").contiguous()
    slot_indices2_npu = slot_indices2.to("npu").contiguous()
    ops.reshape_and_cache(key2_npu, None, key_cache2_npu, None, slot_indices2_npu)
    torch.npu.synchronize()
    torch.testing.assert_close(
        key_cache_cpu2, key_cache2_npu.cpu(), atol=1e-3, rtol=1e-3
    )
    torch.testing.assert_close(
        value_cache2_npu.cpu(),
        torch.zeros_like(value_cache2_npu.cpu()),
        atol=1e-6,
        rtol=1e-6,
    )
    print("PASS: only key, key_cache matched, value_cache all zero.")
