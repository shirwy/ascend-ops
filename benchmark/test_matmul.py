import torch

import ascend910a_extras.ops as ops


def matmul_ref(x, weight):
    # out = torch.matmul(x, weight)
    out = torch.matmul(x, weight.T)
    return out


torch.manual_seed(0)

m = 256 + 1
k = 256
n = 256
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
