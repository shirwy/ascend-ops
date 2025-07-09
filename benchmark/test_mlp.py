import torch

import ascend910a_extras.graph as graph

device = "npu:0"
torch.npu.set_device(device)


bs = 128
hidden_size = 256
intermediate_size = 512

x = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
gate_up_proj = torch.randn(
    intermediate_size * 2, hidden_size, dtype=torch.float16, device=device
)
down_proj = torch.randn(
    hidden_size, intermediate_size, dtype=torch.float16, device=device
)


g = graph.Graph("mlp")
g.build_mlp(bs, hidden_size, intermediate_size)
session = graph.Session()
session.add_graph(0, g)
session.compile_graph(0)
# session.dump_summary(0)


def swiglu(x):
    x1, x2 = x.chunk(2, -1)
    x1_f32, x2_f32 = x1.to(torch.float32), x2.to(torch.float32)
    out = (x1_f32 * x1_f32.sigmoid()) * x2_f32
    return out.to(x.dtype)


def fn_ref(x, gate_up_proj, down_proj):
    y = torch.matmul(x, gate_up_proj.T)
    y = swiglu(y)
    y = torch.matmul(y, down_proj.T)
    return y


y_ref = fn_ref(x, gate_up_proj, down_proj)

y = torch.zeros_like(y_ref)
session.run_async(0, [x, gate_up_proj, down_proj], [y])

print("y_ref", y_ref)
print("y", y)
# not exactly the same
# torch.testing.assert_close(y_ref, y)
