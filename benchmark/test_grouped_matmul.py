import torch

import ascend910a_extras.ops as ops


def grouped_matmul_ref(x, weight, group_list):
    num_exports = weight.shape[0]
    num_tokens = x.shape[0]
    out = torch.empty(num_tokens, weight.shape[2], dtype=x.dtype, device=x.device)
    assert group_list.shape[0] == num_exports

    for ei in range(num_exports):
        if ei == 0:
            token_start = 0
        else:
            token_start = group_list[ei - 1]
        token_end = group_list[ei]

        states = x[token_start:token_end, :]
        w = weight[ei, :, :]

        out[token_start:token_end, :] = torch.matmul(states, w)
    return out


torch.manual_seed(0)

num_tokens = 16384
# num_tokens = 8192
# num_tokens = 256
dim = 2048
# dim = 192
# dim = 128
inner_dim = 768
# inner_dim = 2048
# inner_dim = 128
num_exports = 128
# num_exports = 64
# num_exports = 4
dtype = torch.float16
device = "npu"

x = torch.randn(num_tokens, dim, device=device, dtype=dtype)
w = torch.randn(num_exports, inner_dim, dim, device=device, dtype=dtype).transpose(1, 2)

probs = torch.ones(num_exports, dtype=torch.float)
sample = torch.multinomial(probs, num_samples=num_tokens, replacement=True)
counts = torch.bincount(sample, minlength=num_exports)
assert sum(counts) == num_tokens
group_list = counts.cumsum(dim=0).to(dtype=torch.int64).to(device=device)
print(group_list)

y_ref = grouped_matmul_ref(x.cpu(), w.cpu(), group_list.cpu())
print(f"{y_ref=}", flush=True)

y = ops.grouped_matmul(x, w, group_list).cpu()
torch.npu.synchronize()
print(f"{y}", flush=True)

torch.testing.assert_close(y_ref, y, atol=1e-3, rtol=1e-3)
