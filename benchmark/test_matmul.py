import torch

import ascend910a_extras.ops as ops

REPEAT = 1


def matmul_ref(x, weight):
    out = torch.zeros(x.shape[0], weight.shape[0], device=x.device, dtype=x.dtype)
    for i in range(REPEAT):
        # out += torch.matmul(x, weight.T)
        # out += torch.matmul(x[:, :128], weight.T[:128, :])
        out += torch.matmul(x[:, 128:], weight.T[128:, :])
    return out


torch.manual_seed(0)

# m = 256 + 1
# k = 256
# n = 256
# m = k = n = 256
m = 128
k = 128
n = 128
dtype = torch.float16
device = "npu"

x = torch.randn(m, k, device=device, dtype=dtype)
w = torch.randn(n, k, device=device, dtype=dtype)

y_ref = matmul_ref(x.cpu(), w.cpu())

y = ops.matmul(x, w).cpu()
# torch.npu.synchronize()
print(f"{y_ref}", flush=True)
print(f"{y}", flush=True)


def print_m(d, m, n):
    for i in range(m):
        for j in range(n):
            print(f"{d[i, j]:3.4f}", end=" ")
        print()


# print("--y_ref--")
# print_m(y_ref, m, n)
# print("--y--")
# print_m(y, m, n)
torch.testing.assert_close(y_ref, y, atol=1e-3, rtol=1e-3)


def do_bench(fn, num_iter=10, num_warmup=10):
    import numpy as np

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_iter + num_warmup)

    for i in range(num_iter + num_warmup):
        with torch.no_grad():
            start.record()
            fn()
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)

    times = times[num_warmup:]
    elapsed_time = np.amin(times) / 1000
    return elapsed_time


sec = do_bench(lambda: ops.matmul(x, w))
flops = 2 * m * k * n * REPEAT
tflops_s = flops / sec / 1e12
gb = (m * k * 2 + n * k * 2 + m * n * 2) / 1e9
gb_s = gb / sec
print(f"{m}x{k}x{n} TFLOPS/s: {tflops_s} GB/s: {gb_s} sec: {sec}")
