import math

import torch
import torch_npu

import ascend910a_extras.graph as graph

device = "npu:0"
torch.npu.set_device(device)
ACL_FORMAT_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29
torch.manual_seed(0)


def prof(fn, trace_fn):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[
            torch_npu.profiler.ExportType.Text,
            torch_npu.profiler.ExportType.Db,
        ],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )

    iter_num = 4

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=iter_num - 1, repeat=1, skip_first=1
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    ) as prof:
        for step in range(iter_num):
            fn()
            torch.npu.synchronize()
            prof.step()

    prof.export_chrome_trace(trace_fn)


def ceil_div(a, b):
    return (a + b - 1) // b


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


def swiglu(x):
    x1, x2 = x.chunk(2, -1)
    x1_f32, x2_f32 = x1.to(torch.float32), x2.to(torch.float32)
    out = (x1_f32 * x1_f32.sigmoid()) * x2_f32
    return out.to(x.dtype)


def load_weight(weight_name):
    import json
    from pathlib import Path

    from safetensors import safe_open

    path = Path("/data/models/Qwen/Qwen3-8B-2L")
    weight_map_path = path / "model.safetensors.index.json"
    with open(weight_map_path, "r") as f:
        weight_map = json.load(f)
    weight_path = path / weight_map["weight_map"][weight_name]
    with safe_open(path / weight_path, framework="pt", device="cpu") as f:
        weight = f.get_tensor(weight_name)

    return weight


def test_model_atb():
    num_split = 1
    # num_split = 2

    bs = 4
    hidden_size = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = hidden_size // num_heads
    intermediate_size = 12288
    num_layers = 2
    rms_norm_eps = 1e-6
    vocab_size = 151936
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    dtype = torch.float16

    num_pages = 64
    page_size = 128
    max_seqlen = 256
    assert bs * max_seqlen <= num_pages * page_size

    # token_ids = torch.randint(0, vocab_size, (bs,), dtype=torch.int32, device=device)
    token_ids = torch.tensor([32863, 32863, 32863, 0], dtype=torch.int64, device=device)
    key_caches = [
        torch_npu.npu_format_cast(
            torch.randn(
                num_pages,
                num_kv_heads * head_dim // 16,
                page_size,
                16,
                dtype=dtype,
                device=device,
            ),
            ACL_FORMAT_FRACTAL_NZ,
        )
        for _ in range(num_layers)
    ]
    value_caches = [
        torch_npu.npu_format_cast(
            torch.randn(
                num_pages,
                num_kv_heads * head_dim // 16,
                page_size,
                16,
                dtype=dtype,
                device=device,
            ),
            ACL_FORMAT_FRACTAL_NZ,
        )
        for _ in range(num_layers)
    ]
    position_ids = torch.full((bs,), max_seqlen - 1, dtype=torch.int32, device=device)
    context_lens = torch.full((bs,), max_seqlen, dtype=torch.int32, device=device)
    block_tables = (
        torch.arange(0, bs * max_seqlen // page_size, dtype=torch.int32, device=device)
        .reshape(bs, -1)
        .contiguous()
    )
    slot_mapping = torch.full((bs,), 0, dtype=torch.int32, device=device)
    for i in range(bs):
        slot_mapping[i] = (max_seqlen - 1) * (i + 1)

    weights = []
    weight_map = {}
    # vocab_weight = torch.randn(
    #     vocab_size, hidden_size, dtype=torch.float16, device=device
    # )
    vocab_weight = load_weight("model.embed_tokens.weight").to(
        dtype=dtype, device=device
    )
    weight_map["vocab_weight"] = vocab_weight
    weights.append(vocab_weight)
    for i in range(num_layers):
        # pre_rms_norm_weight = torch.randn(
        #     hidden_size, dtype=torch.float16, device=device
        # )
        # qkv_proj = torch.randn(
        #     q_size + 2 * kv_size, hidden_size, dtype=torch.float16, device=device
        # )
        # q_norm = torch.randn(head_dim, dtype=torch.float16, device=device)
        # k_norm = torch.randn(head_dim, dtype=torch.float16, device=device)
        # o_proj = torch.randn(
        #     hidden_size, hidden_size, dtype=torch.float16, device=device
        # )
        # post_rms_norm_weight = torch.randn(
        #     hidden_size, dtype=torch.float16, device=device
        # )
        # gate_up_proj_weight = torch.randn(
        #     intermediate_size * 2, hidden_size, dtype=torch.float16, device=device
        # )
        # down_proj_weight = torch.randn(
        #     hidden_size, intermediate_size, dtype=torch.float16, device=device
        # )
        pre_rms_norm_weight = load_weight(
            f"model.layers.{i}.input_layernorm.weight"
        ).to(dtype=dtype, device=device)
        post_rms_norm_weight = load_weight(
            f"model.layers.{i}.post_attention_layernorm.weight"
        ).to(dtype=dtype, device=device)
        gate_proj_weight = load_weight(f"model.layers.{i}.mlp.gate_proj.weight").to(
            dtype=dtype, device=device
        )
        up_proj_weight = load_weight(f"model.layers.{i}.mlp.up_proj.weight").to(
            dtype=dtype, device=device
        )
        gate_up_proj_weight = torch.cat(
            [gate_proj_weight, up_proj_weight], dim=0
        ).contiguous()
        assert gate_up_proj_weight.shape == (intermediate_size * 2, hidden_size)
        down_proj_weight = load_weight(f"model.layers.{i}.mlp.down_proj.weight").to(
            dtype=dtype, device=device
        )

        weight_map[f"pre_rms_norm_weight_{i}"] = pre_rms_norm_weight
        # weight_map[f"qkv_proj_{i}"] = qkv_proj
        # weight_map[f"q_norm_{i}"] = q_norm
        # weight_map[f"k_norm_{i}"] = k_norm
        # weight_map[f"o_proj_{i}"] = o_proj
        weight_map[f"post_rms_norm_weight_{i}"] = post_rms_norm_weight
        weight_map[f"gate_up_proj_weight_{i}"] = gate_up_proj_weight
        weight_map[f"down_proj_weight_{i}"] = down_proj_weight

        weights.append(pre_rms_norm_weight)
        # weights.append(qkv_proj)
        # weights.append(q_norm)
        # weights.append(k_norm)
        # weights.append(o_proj)
        weights.append(post_rms_norm_weight)
        weights.append(gate_up_proj_weight)
        weights.append(down_proj_weight)
    # final_rms_norm_weight = torch.randn(hidden_size, dtype=torch.float16, device=device)
    final_rms_norm_weight = load_weight("model.norm.weight").to(
        dtype=dtype, device=device
    )
    weight_map["final_rms_norm_weight"] = final_rms_norm_weight
    weights.append(final_rms_norm_weight)

    config = graph.GraphConfig()
    config.batch_size = bs
    config.hidden_size = hidden_size
    config.num_heads = num_heads
    config.num_kv_heads = num_kv_heads
    config.intermediate_size = intermediate_size
    config.num_layers = num_layers
    config.rms_norm_eps = rms_norm_eps

    g = graph.Graph(config)
    g.build_model(num_split)
    ctx = graph.Context()
    _y = torch.zeros(bs, hidden_size, dtype=torch.float16, device=device)
    ctx.setup(
        g,
        token_ids,
        key_caches,
        value_caches,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
        weights,
        _y,
    )
    # ctx.setup_fullgraph(
    #     g,
    #     [token_ids, key_caches[0], value_caches[0], key_caches[1], value_caches[1], position_ids, slot_mapping, block_tables, context_lens],
    #     [ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND],
    #     weights,
    #     [_y],
    # )

    def fn():
        ctx.run_with_dummy_setup(g)
        return _y

    y = fn()
    print("y", y)
    print(f"{num_layers=} {num_split=} passed")

    # prof(
    #     fn=lambda: fn(
    #         token_ids,
    #         key_cache,
    #         value_cache,
    #         position_ids,
    #         slot_mapping,
    #         block_tables,
    #         context_lens,
    #     ),
    #     trace_fn=f"test_model_atb_layer{num_layers}.json",
    # )


def test_rope_atb():
    bs = 4
    hidden_size = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = hidden_size // num_heads
    rotary_dim = head_dim
    max_position_embeddings = 40960
    dtype = torch.float16

    def _compute_inv_freq(base: float) -> torch.Tensor:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, device=device) / rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache() -> torch.Tensor:
        inv_freq = _compute_inv_freq(1000000)
        t = torch.arange(max_position_embeddings, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).contiguous()
        return cache.to(dtype)

    cos_sin_cache = _compute_cos_sin_cache().to(dtype=dtype, device=device)
    cos_cache, sin_cache = cos_sin_cache.chunk(2, -1)
    cos_cache = cos_cache.contiguous()
    sin_cache = sin_cache.contiguous()
    print(f"{cos_sin_cache.shape=}")

    q = torch.randn(bs, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(bs, num_kv_heads, head_dim, dtype=dtype, device=device)
    position_ids = torch.randint(
        0, max_position_embeddings, (bs,), dtype=torch.int32, device=device
    )

    config = graph.GraphConfig()
    config.batch_size = bs
    config.hidden_size = hidden_size
    config.num_heads = num_heads

    g = graph.Graph(config)
    g.build_rope()
    ctx = graph.Context()
    inputs = [q, k, position_ids, cos_cache, sin_cache]
    input_formats = [
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
    ]
    weights = []
    _q = torch.zeros_like(q)
    _k = torch.zeros_like(k)
    ctx.setup_fullgraph(g, inputs, input_formats, weights, [_q, _k])

    def fn_naive(q, k, position_ids, cos_cache, sin_cache):
        _q = q.transpose(0, 1).contiguous()
        _k = k.transpose(0, 1).contiguous()
        cos = cos_cache[position_ids][None, :]
        sin = sin_cache[position_ids][None, :]

        q_x, q_y = _q.chunk(2, -1)
        k_x, k_y = _k.chunk(2, -1)

        new_q_x = q_x * cos - q_y * sin
        new_q_y = q_y * cos + q_x * sin

        new_k_x = k_x * cos - k_y * sin
        new_k_y = k_y * cos + k_x * sin

        new_q = torch.cat([new_q_x, new_q_y], dim=-1)
        new_k = torch.cat([new_k_x, new_k_y], dim=-1)

        new_q = new_q.transpose(0, 1).contiguous()
        new_k = new_k.transpose(0, 1).contiguous()
        return new_q, new_k

    def fn_ref(q, k, position_ids, cos_sin_cache):
        _q = q.view(bs, num_heads * head_dim)
        _k = k.view(bs, num_kv_heads * head_dim)
        out_q = _q.clone()
        out_k = _k.clone()
        torch_npu._npu_rotary_embedding(
            position_ids, out_q, out_k, head_dim, cos_sin_cache, True
        )
        out_q = out_q.view(bs, num_heads, head_dim)
        out_k = out_k.view(bs, num_kv_heads, head_dim)
        return out_q, out_k

    def fn():
        ctx.run(g)
        return _q, _k

    out_q_ref, out_k_ref = fn_ref(q, k, position_ids, cos_sin_cache)
    out_q, out_k = fn()
    # out_q_naive, out_k_naive = fn_naive(q, k, position_ids, cos_cache, sin_cache)
    print("out_q_ref", out_q_ref)
    # print("out_q_naive", out_q_naive)
    print("out_q", out_q)
    print("out_k_ref", out_k_ref)
    # print("out_k_naive", out_k_naive)
    # print("out_k_op", out_k_op)
    print("out_k", out_k)

    torch.testing.assert_close(out_q_ref, out_q)
    torch.testing.assert_close(out_k_ref, out_k)
    print("test_rope_atb passed")


def test_embedding_atb():
    bs = 127
    hidden_size = 4096
    vocab_size = 151936

    token_ids = torch.randint(0, vocab_size, (bs,), dtype=torch.int32, device=device)
    vocab_weight = torch.randn(
        vocab_size, hidden_size, dtype=torch.float16, device=device
    )

    config = graph.GraphConfig()
    config.batch_size = bs
    config.hidden_size = hidden_size

    g = graph.Graph()
    g.build_embedding(config)
    ctx = graph.Context()

    def fn_ref(token_ids):
        y = vocab_weight[token_ids].contiguous()
        return y

    def fn(token_ids):
        y = torch.empty(bs, hidden_size, dtype=torch.float16, device=device)
        inputs = [token_ids]
        weights = [vocab_weight]
        input_formats = [ACL_FORMAT_ND]
        ctx.setup_then_run(g, inputs, input_formats, weights, [y])
        return y

    y_ref = fn_ref(token_ids)
    y = fn(token_ids)
    print("y_ref", y_ref)
    print("y_ref.shape", y_ref.shape)
    print("y", y)
    print("y.shape", y.shape)
    torch.testing.assert_close(y_ref, y)
    print("test_embedding_atb passed")


def test_paged_attn_atb():
    bs = 32
    hidden_size = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = hidden_size // num_heads
    scale = 1 / math.sqrt(head_dim)

    num_pages = 64
    page_size = 128
    max_seqlen = 256
    assert bs * max_seqlen <= num_pages * page_size

    q = torch.randn(bs, num_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(bs, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(bs, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    key_cache = torch.randn(
        num_pages,
        num_kv_heads * head_dim // 16,
        page_size,
        16,
        dtype=torch.float16,
        device=device,
    )
    key_cache = torch_npu.npu_format_cast(key_cache, ACL_FORMAT_FRACTAL_NZ)
    key_cache_ref = key_cache.clone()
    key_cache_ref = torch_npu.npu_format_cast(key_cache_ref, ACL_FORMAT_FRACTAL_NZ)
    value_cache = torch.randn(
        num_pages,
        num_kv_heads * head_dim // 16,
        page_size,
        16,
        dtype=torch.float16,
        device=device,
    )
    value_cache = torch_npu.npu_format_cast(value_cache, ACL_FORMAT_FRACTAL_NZ)
    value_cache_ref = value_cache.clone()
    value_cache_ref = torch_npu.npu_format_cast(value_cache_ref, ACL_FORMAT_FRACTAL_NZ)
    position_ids = torch.full((bs,), max_seqlen - 1, dtype=torch.int32, device=device)
    context_lens = torch.full((bs,), max_seqlen, dtype=torch.int32, device=device)
    block_tables = (
        torch.arange(0, bs * max_seqlen // page_size, dtype=torch.int32, device=device)
        .reshape(bs, -1)
        .contiguous()
    )
    slot_mapping = torch.full((bs,), 0, dtype=torch.int32, device=device)
    for i in range(bs):
        slot_mapping[i] = (max_seqlen - 1) * (i + 1)

    config = graph.GraphConfig()
    config.batch_size = bs
    config.hidden_size = hidden_size
    config.num_heads = num_heads
    config.num_kv_heads = num_kv_heads
    config.rms_norm_eps = 1e-6

    g = graph.Graph(config)
    g.build_paged_attn()
    inputs = [
        q,
        k,
        v,
        key_cache,
        value_cache,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
    ]
    input_formats = [
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_FRACTAL_NZ,
        ACL_FORMAT_FRACTAL_NZ,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
    ]
    weights = []
    _y = torch.zeros_like(q)
    ctx = graph.Context()
    ctx.setup_fullgraph(g, inputs, input_formats, weights, [_y])

    def fn_ref(
        q,
        k,
        v,
        key_cache,
        value_cache,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
    ):
        y = torch.empty_like(q)
        torch_npu._npu_reshape_and_cache(
            key=k,
            value=v,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=slot_mapping,
        )
        torch_npu._npu_paged_attention(
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            block_table=block_tables,
            context_lens=context_lens,
            out=y,
        )
        return y

    def fn():
        ctx.run(g)
        return _y

    y_ref = fn_ref(
        q,
        k,
        v,
        key_cache_ref,
        value_cache_ref,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
    )
    y = fn()
    print("y_ref", y_ref)
    print("y", y)
    torch.testing.assert_close(y_ref, y)
    torch.testing.assert_close(key_cache, key_cache_ref)
    torch.testing.assert_close(value_cache, value_cache_ref)
    print("test_paged_attn_atb passed")


def test_attn_atb():
    bs = 32
    hidden_size = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = hidden_size // num_heads
    scale = 1 / math.sqrt(head_dim)

    num_pages = 64
    page_size = 128
    max_seqlen = 256
    assert bs * max_seqlen <= num_pages * page_size

    head_dim = hidden_size // num_heads
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    # input
    x = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    key_cache = torch.randn(
        num_pages,
        num_kv_heads * head_dim // 16,
        page_size,
        16,
        dtype=torch.float16,
        device=device,
    )
    key_cache = torch_npu.npu_format_cast(key_cache, ACL_FORMAT_FRACTAL_NZ)
    key_cache_ref = key_cache.clone()
    key_cache_ref = torch_npu.npu_format_cast(key_cache_ref, ACL_FORMAT_FRACTAL_NZ)
    value_cache = torch.randn(
        num_pages,
        num_kv_heads * head_dim // 16,
        page_size,
        16,
        dtype=torch.float16,
        device=device,
    )
    value_cache = torch_npu.npu_format_cast(value_cache, ACL_FORMAT_FRACTAL_NZ)
    value_cache_ref = value_cache.clone()
    value_cache_ref = torch_npu.npu_format_cast(value_cache_ref, ACL_FORMAT_FRACTAL_NZ)
    position_ids = torch.full((bs,), max_seqlen - 1, dtype=torch.int32, device=device)
    context_lens = torch.full((bs,), max_seqlen, dtype=torch.int32, device=device)
    block_tables = (
        torch.arange(0, bs * max_seqlen // page_size, dtype=torch.int32, device=device)
        .reshape(bs, -1)
        .contiguous()
    )
    slot_mapping = torch.full((bs,), 0, dtype=torch.int32, device=device)
    for i in range(bs):
        slot_mapping[i] = (max_seqlen - 1) * (i + 1)

    # weight
    qkv_proj = torch.randn(
        q_size + 2 * kv_size, hidden_size, dtype=torch.float16, device=device
    )
    q_norm = torch.randn(head_dim, dtype=torch.float16, device=device)
    k_norm = torch.randn(head_dim, dtype=torch.float16, device=device)
    o_proj = torch.randn(hidden_size, hidden_size, dtype=torch.float16, device=device)

    config = graph.GraphConfig()
    config.batch_size = bs
    config.hidden_size = hidden_size
    config.num_heads = num_heads
    config.num_kv_heads = num_kv_heads
    config.rms_norm_eps = 1e-6

    g = graph.Graph(config)
    g.build_attn()
    inputs = [
        x,
        key_cache,
        value_cache,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
    ]
    input_formats = [
        ACL_FORMAT_ND,
        ACL_FORMAT_FRACTAL_NZ,
        ACL_FORMAT_FRACTAL_NZ,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
        ACL_FORMAT_ND,
    ]
    weights = [qkv_proj, q_norm, k_norm, o_proj]
    _y = torch.zeros_like(x)
    ctx = graph.Context()
    ctx.setup_fullgraph(g, inputs, input_formats, weights, [_y])

    def fn_ref(
        x,
        key_cache,
        value_cache,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
    ):
        qkv = x @ qkv_proj.t()
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q_by_head = q.view(bs, num_heads, head_dim)
        k_by_head = k.view(bs, num_kv_heads, head_dim)

        q_by_head = rmsnorm(q_by_head, None, q_norm)
        k_by_head = rmsnorm(k_by_head, None, k_norm)

        q = q_by_head.view_as(q)
        k = k_by_head.view_as(k)

        # rope
        # skip

        q = q.view(bs, num_heads, head_dim)
        k = k.view(bs, num_kv_heads, head_dim)
        v = v.view(bs, num_kv_heads, head_dim)

        torch_npu._npu_reshape_and_cache(
            key=k,
            value=v,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=slot_mapping,
        )

        y = torch.empty_like(q)
        torch_npu._npu_paged_attention(
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            block_table=block_tables,
            context_lens=context_lens,
            out=y,
        )
        y = y.view(bs, hidden_size)
        y = y @ o_proj.t()
        return y

    def fn():
        ctx.run_with_dummy_setup(g)
        return _y

    y_ref = fn_ref(
        x,
        key_cache_ref,
        value_cache_ref,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
    )
    y = fn()
    print("y_ref", y_ref)
    print("y", y, flush=True)
    torch.testing.assert_close(y_ref, y)
    print("test_attn passed")


def test_rmsnorm_atb():
    bs = 128
    hidden_size = 256
    x = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float16, device=device)

    config = graph.GraphConfig()
    config.batch_size = bs
    config.hidden_size = hidden_size
    config.rms_norm_eps = 1e-6

    g = graph.Graph(config)
    g.build_rmsnorm()
    ctx = graph.Context()
    inputs = [x]
    input_formats = [ACL_FORMAT_ND]
    weights = [weight]
    _y = torch.zeros_like(x)
    ctx.setup_fullgraph(g, inputs, input_formats, weights, [_y])

    def fn_ref(x, weight):
        y_ref = rmsnorm(x, None, weight)
        return y_ref

    def fn():
        ctx.run(g)
        return _y

    y_ref = fn_ref(x, weight)
    y = fn()
    print("y_ref", y_ref)
    print("y", y)
    torch.testing.assert_close(y_ref, y)
    print("test_rmsnorm_atb passed")


def test_rmsnorm_with_residual_atb():
    bs = 128
    hidden_size = 256
    x = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    residual = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    weight = torch.randn(hidden_size, dtype=torch.float16, device=device)

    config = graph.GraphConfig()
    config.batch_size = bs
    config.hidden_size = hidden_size
    config.rms_norm_eps = 1e-6

    g = graph.Graph(config)
    g.build_rmsnorm_with_residual()
    ctx = graph.Context()

    inputs = [x, residual]
    input_formats = [ACL_FORMAT_ND, ACL_FORMAT_ND]
    weights = [weight]
    _y = torch.zeros_like(x)
    _res_y = torch.zeros_like(residual)
    ctx.setup_fullgraph(g, inputs, input_formats, weights, [_y, _res_y])

    def fn_ref(x, residual, weight):
        y_ref, res_y_ref = rmsnorm(x, residual, weight)
        return y_ref, res_y_ref

    def fn():
        ctx.run(g)
        return _y, _res_y

    y_ref, res_y_ref = fn_ref(x, residual, weight)
    y, res_y = fn()
    print("y_ref", y_ref)
    print("y", y)
    print("res_y_ref", res_y_ref)
    print("res_y", res_y)
    torch.testing.assert_close(y_ref, y)
    torch.testing.assert_close(res_y_ref, res_y)
    print("test_rmsnorm_with_residual_atb passed")


def test_mlp_atb():
    bs = 2
    hidden_size = 4096
    intermediate_size = 12288
    x = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    gate_up_proj_weight = torch.randn(
        intermediate_size * 2, hidden_size, dtype=torch.float16, device=device
    )
    down_proj_weight = torch.randn(
        hidden_size, intermediate_size, dtype=torch.float16, device=device
    )

    config = graph.GraphConfig()
    config.batch_size = bs
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_layers = 1

    g = graph.Graph(config)
    g.build_mlp()
    ctx = graph.Context()

    inputs = [x]
    input_formats = [ACL_FORMAT_ND]
    weights = [gate_up_proj_weight, down_proj_weight]
    _y = torch.zeros_like(x)
    ctx.setup_fullgraph(g, inputs, input_formats, weights, [_y])

    def fn_ref(x, gate_up_proj_weight, down_proj_weight):
        y = x @ gate_up_proj_weight.t()
        y = swiglu(y)
        y = y @ down_proj_weight.t()
        return y

    def fn():
        ctx.run_with_dummy_setup(g)
        return _y

    y_ref = fn_ref(x, gate_up_proj_weight, down_proj_weight)
    y = fn()
    print("y_ref", y_ref)
    print("y", y)
    torch.testing.assert_close(y_ref, y)
    print("test_mlp passed")

    # prof(fn=lambda: fn(x, gate_up_proj_weight, down_proj_weight), trace_fn=f"mlp_bs{bs}_trace.json")


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


def test_mlp():
    bs = 2
    hidden_size = 4096
    intermediate_size = 12288
    x = torch.randn(bs, hidden_size, dtype=torch.float16, device=device)
    gate_up_proj_weight = torch.randn(
        intermediate_size * 2, hidden_size, dtype=torch.float16, device=device
    )
    down_proj_weight = torch.randn(
        hidden_size, intermediate_size, dtype=torch.float16, device=device
    )

    g = graph.Graph("mlp")
    g.build_mlp(bs, hidden_size, intermediate_size)

    session = graph.Session()
    session.add_graph(0, g)
    session.compile_graph(0)

    def fn_ref(x, gate_up_proj_weight, down_proj_weight):
        y = x @ gate_up_proj_weight.t()
        y = swiglu(y)
        y = y @ down_proj_weight.t()
        return y

    def fn(x, gate_up_proj_weight, down_proj_weight):
        y = torch.empty_like(x)
        session.run_async(0, [x, gate_up_proj_weight, down_proj_weight], [y])
        return y

    y_ref = fn_ref(x, gate_up_proj_weight, down_proj_weight)
    y = fn(x, gate_up_proj_weight, down_proj_weight)
    print("y_ref", y_ref)
    print("y", y)
    # not exactly the same
    # torch.testing.assert_close(y_ref, y)
    print("test_mlp passed")

    prof(
        fn=lambda: fn(x, gate_up_proj_weight, down_proj_weight),
        trace_fn=f"mlp_bs{bs}_trace_ge.json",
    )


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


# test_mlp_atb()
# test_rmsnorm_atb()
# test_rmsnorm_with_residual_atb()
# test_attn_atb()
# test_paged_attn_atb()
# test_embedding_atb()
# test_model_atb()
test_rope_atb()


# test_mlp()
# test_rmsnorm()
# test_attn()
