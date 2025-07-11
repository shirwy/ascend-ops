import torch
import torch_npu

import ascend910a_extras.graph as graph

device = "npu:0"
torch.npu.set_device(device)


def rmsnorm(x, residual, weight, eps=1e-6):
    if residual is not None:
        orig_dtype = residual.dtype
        x = x + residual.to(x.dtype)
        residual = x.to(orig_dtype)
        x, _ = torch_npu.npu_rms_norm(x, weight, eps)
        return x, residual
    else:
        x, residual = torch_npu.npu_rms_norm(x, weight, eps)
        return x


def test_rmsnorm():
    bs = 128
    hidden_size = 256
    x = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    residual = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float16, device=device)

    g = graph.Graph("graph")
    g.build_rmsnorm(bs, hidden_size)

    session = graph.Session()
    session.add_graph(0, g)
    session.compile_graph(0)

    def fn(x, residual, weight):
        y = torch.empty_like(x)
        res_y = torch.empty_like(residual)
        session.run_async(0, [x, residual, weight], [y, res_y])
        return y, res_y

    y_ref, res_y_ref = rmsnorm(x, residual, weight)
    print(f"{y_ref.shape=} {res_y_ref.shape=}")
    y, res_y = fn(x, residual, weight)
    print("y_ref", y_ref)
    print("y", y)
    print("res_y_ref", res_y_ref)
    print("res_y", res_y)
    torch.testing.assert_close(y_ref, y)
    torch.testing.assert_close(res_y_ref, res_y)
    print("test_rmsnorm passed")


def test_attn():
    bs = 128
    hidden_size = 512
    num_heads = 16
    num_kv_heads = 8

    head_dim = hidden_size // num_heads
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    x = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    qkv_proj = torch.randn(
        q_size + 2 * kv_size, hidden_size, dtype=torch.float16, device=device
    )
    q_norm = torch.randn(head_dim, dtype=torch.float16, device=device)
    k_norm = torch.randn(head_dim, dtype=torch.float16, device=device)

    g = graph.Graph("graph")
    g.build_attn(bs, hidden_size, num_heads, num_kv_heads)

    session = graph.Session()
    session.add_graph(0, g)
    session.compile_graph(0)

    def fn(x):
        q = torch.empty(bs, q_size, dtype=torch.float16, device=device)
        k = torch.empty(bs, kv_size, dtype=torch.float16, device=device)
        session.run_async(0, [x, qkv_proj, q_norm, k_norm], [q, k])
        return q, k

    def fn_ref(x):
        qkv = x @ qkv_proj.t()
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q_by_head = q.view(bs, num_heads, head_dim)
        k_by_head = k.view(bs, num_kv_heads, head_dim)

        q_by_head = rmsnorm(q_by_head, None, q_norm)
        k_by_head = rmsnorm(k_by_head, None, k_norm)

        q = q_by_head.view_as(q)
        k = k_by_head.view_as(k)
        return q, k

    q_ref, k_ref = fn_ref(x)
    q, k = fn(x)
    print("q_ref", q_ref)
    print("q", q)
    print("k_ref", k_ref)
    print("k", k)
    torch.testing.assert_close(q_ref, q)
    torch.testing.assert_close(k_ref, k)
    print("test_attn passed")


# test_rmsnorm()
test_attn()
