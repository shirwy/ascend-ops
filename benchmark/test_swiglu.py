import torch

import ascend910a_extras.ops as ops


def swiglu_cpu(x):
    x1, x2 = x.chunk(2, -1)
    x1_f32, x2_f32 = x1.to(torch.float32), x2.to(torch.float32)
    out = (x1_f32 * x1_f32.sigmoid()) * x2_f32
    # out = x1_f32 * x2_f32
    # out = x1_f32
    return out.to(x.dtype)


def swiglu_ref(x):
    dtype = x.dtype
    x = x.to(torch.float32)
    split_size = x.size(-1) // 2
    x, gate = x.split(split_size, dim=-1)
    out = x * torch.sigmoid(gate)
    return out.to(dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    num_tokens = 2048
    # dim = 12288
    # dim = 1024
    dim = 192
    dtype = torch.float16
    x_npu = torch.randn(num_tokens, dim * 2, device="npu", dtype=dtype)
    # x_npu = torch.arange(num_tokens * dim * 2).view(num_tokens, dim * 2).to(dtype).to('npu')
    print(f"Input tensor: {x_npu.shape}, {x_npu.dtype}, {x_npu.device}")
    print(f"{x_npu=}")
    x_cpu = x_npu.cpu()
    y_npu = ops.swiglu(x_npu)

    stream = torch.npu.current_stream().npu_stream
    print(f"py stream: {stream} {hex(stream)}", flush=True)
    y_npu = ops.swiglu(x_npu)

    # y_cpu = swiglu_cpu(x_cpu)
    y_cpu = swiglu_ref(x_npu).cpu()
    torch.npu.synchronize()
    y_npu_cpu = y_npu.cpu()
    print(f"{y_cpu=}")
    print(f"{y_npu_cpu=}")
    torch.testing.assert_close(y_cpu, y_npu_cpu)
    print("PASS")
