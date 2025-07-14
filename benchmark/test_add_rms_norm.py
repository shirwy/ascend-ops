import torch
import ascend910a_extras.ops as ops

def add_rms_norm_cpu(x, residual, weight, epsilon):
    y = x + residual
    rms = torch.sqrt((y.float() ** 2).mean(-1, keepdim=True) + epsilon)
    out = y / rms * weight
    return out.to(x.dtype), y.to(x.dtype)

if __name__ == "__main__":
    torch.manual_seed(0)
    num_tokens = 1024
    dim = 1024
    dtype = torch.float16
    epsilon = 1e-6

    x_npu = torch.randn(num_tokens, dim, device="npu", dtype=dtype)
    residual_npu = torch.randn(num_tokens, dim, device="npu", dtype=dtype)
    weight_npu = torch.randn(dim, device="npu", dtype=dtype)

    x_cpu = x_npu.cpu()
    residual_cpu = residual_npu.cpu()
    weight_cpu = weight_npu.cpu()

    y_cpu, residual_output_cpu = add_rms_norm_cpu(x_cpu, residual_cpu, weight_cpu, epsilon)
    y_npu, residual_output_npu = ops.add_rms_norm(x_npu, residual_npu, weight_npu, epsilon)
    torch.npu.synchronize()
    y_npu_cpu = y_npu.cpu()
    residual_output_npu_cpu = residual_output_npu.cpu()

    print(f"y_cpu={y_cpu}")
    print(f"y_npu_cpu={y_npu_cpu}")
    torch.testing.assert_close(y_cpu, y_npu_cpu, atol=1e-3, rtol=1e-3)
    print(f"residual_output_cpu={residual_output_cpu}")
    print(f"residual_output_npu_cpu={residual_output_npu_cpu}")
    torch.testing.assert_close(residual_output_cpu, residual_output_npu_cpu, atol=1e-3, rtol=1e-3)
    print("PASS")