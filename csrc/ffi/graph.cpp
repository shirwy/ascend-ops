// FIXME: must be included first.
// #include "ge/ge_api.h"
// #include "all_ops.h"
// #include "op_proto.h"

#include <atb/atb_infer.h>

#include "all_ops.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "adaptor.h"

#include "dbg/dbg.h"


namespace py = pybind11;

namespace native {

struct GraphConfig {
  int batch_size = 1;
  int hidden_size = 4096;
  int num_heads = 32;
  int num_kv_heads = 8;
  int intermediate_size = 12288;
  int num_layers = 1;
  float rms_norm_eps = 1e-6;

  void display() {
    printf("GraphConfig: batch_size=%d, hidden_size=%d, num_heads=%d, num_kv_heads=%d, intermediate_size=%d, num_layers=%d, rms_norm_eps=%f\n",
      batch_size, hidden_size, num_heads, num_kv_heads, intermediate_size, num_layers, rms_norm_eps);
  }
};

class GraphBuilder {
public:
  GraphConfig config;

  atb::GraphOpBuilder* builder;
  uint32_t tensor_num;
  std::vector<uint32_t> in_ids;
  std::vector<uint32_t> internal_ids;
  std::vector<uint32_t> out_ids;
  std::map<uint32_t, uint32_t> id_map;

  atb::GraphParam graph_param;
  atb::ReshapeFunc identity_reshape_func = [](const atb::Dims& old_shape, atb::Dims& new_shape) {
    new_shape = old_shape;
  };

  GraphBuilder(GraphConfig config) : config(config) {
    builder = nullptr;
    clear();
  }

  void clear() {
    if (builder) {
      CHECK_ATB(atb::DestroyGraphOpBuilder(builder));
      builder = nullptr;
    }
    tensor_num = 0;
    in_ids.clear();
    internal_ids.clear();
    out_ids.clear();
    id_map.clear();
    graph_param.nodes.clear();
    graph_param.inferShapeFunc = nullptr;
  }

  atb::Operation* build(
    const char* name,
    int input_num,
    std::function<std::vector<uint32_t>(std::vector<uint32_t>)> build_fn
  ) {
    clear();
    CHECK_ATB(atb::CreateGraphOpBuilder(&builder));
    assert(builder != nullptr && "builder should not be nullptr");
    std::vector<uint32_t> xs(input_num);
    for (int i = 0; i < input_num; i++) {
      uint32_t x = tensor_num++;
      in_ids.push_back(x);
      xs[i] = x;
    }

    // build graph
    auto ys = build_fn(xs);

    // remap ids
    // dbg(in_ids, internal_ids, out_ids);
    // dbg(xs, ys);
    assert(std::all_of(ys.begin(), ys.end(), [&](uint32_t y) {
      return std::count(ys.begin(), ys.end(), y) == 1;
    }));
    assert(out_ids.size() == 0);
    std::set<uint32_t> ys_set(ys.begin(), ys.end());
    for (auto& y: ys_set) {
      internal_ids.erase(std::find(internal_ids.begin(), internal_ids.end(), y));
      out_ids.push_back(y);
    }
    remap();

    // create graph
    graph_param.name = name;
    graph_param.inTensorNum = in_ids.size();
    graph_param.outTensorNum = out_ids.size();
    graph_param.internalTensorNum = internal_ids.size();

    for (auto& node: graph_param.nodes) {
      for (int i = 0; i < node.inTensorIds.size(); i++) {
        node.inTensorIds[i] = id_map[node.inTensorIds[i]];
      }
      for (int i = 0; i < node.outTensorIds.size(); i++) {
        node.outTensorIds[i] = id_map[node.outTensorIds[i]];
      }
    }

    atb::Operation* graph = nullptr;
    CHECK_ATB(atb::CreateOperation(graph_param, &graph));
    assert(graph != nullptr && "graph should not be nullptr");
    dbg(name, input_num, in_ids.size(), internal_ids.size(), out_ids.size());
    CHECK_ATB(atb::DestroyGraphOpBuilder(builder));
    return graph;
  }


  void remap() {
    uint32_t in_tensor_id = 0;
    for (auto& id: in_ids) {
      id_map[id] = in_tensor_id++;
    }
    uint32_t out_tensor_id = in_tensor_id;
    for (auto& id: out_ids) {
      id_map[id] = out_tensor_id++;
    }
    uint32_t internal_tensor_id = out_tensor_id;
    for (auto& id: internal_ids) {
      id_map[id] = internal_tensor_id++;
    }
  }

  std::vector<uint32_t> add_decoder_layer(
    uint32_t x,
    std::optional<uint32_t> residual,
    uint32_t key_cache,
    uint32_t value_cache,
    uint32_t position_ids,
    uint32_t slot_mapping,
    uint32_t block_tables,
    uint32_t context_lens,
    uint32_t cos_cache,
    uint32_t sin_cache
  ) {
    float rms_norm_eps = config.rms_norm_eps;
    uint32_t hidden_states = uint32_t(-1);
    uint32_t res = uint32_t(-1);
    if (residual.has_value()) {
      auto x_and_residual = add_rmsnorm(x, residual.value(), rms_norm_eps, identity_reshape_func);
      assert(x_and_residual.size() == 2);
      hidden_states = x_and_residual[0];
      res = x_and_residual[1];
    } else {
      res = x;
      hidden_states = add_rmsnorm(x, std::nullopt, rms_norm_eps, identity_reshape_func)[0];
    }

    hidden_states = add_attn(
      {hidden_states, key_cache, value_cache, position_ids, slot_mapping, block_tables, context_lens, cos_cache, sin_cache}
    );

    auto hidden_states_and_res = add_rmsnorm(hidden_states, res, rms_norm_eps, identity_reshape_func);
    assert(hidden_states_and_res.size() == 2);
    hidden_states = hidden_states_and_res[0];
    res = hidden_states_and_res[1];

    auto y = add_mlp(hidden_states);
    return {y, res};
  }


  uint32_t add_embedding(uint32_t token_ids) {
    // dbg(token_ids);
    atb::Node node;
    atb::infer::GatherParam param;
    param.axis = 0;
    CHECK_ATB(atb::CreateOperation(param, &node.operation));
    uint32_t vocab_weight = tensor_num++;
    uint32_t y = tensor_num++;
    node.inTensorIds = {vocab_weight, token_ids};
    node.outTensorIds = {y};
    graph_param.nodes.push_back(node);
    in_ids.push_back(vocab_weight);
    internal_ids.push_back(y);
    return y;
  }

  uint32_t add_mlp(uint32_t x) {
    auto y = add_linear(x, false, true, identity_reshape_func);
    y = add_swiglu(y);
    y = add_linear(y, false, true, identity_reshape_func);
    return y;
  }

  uint32_t add_attn(std::vector<uint32_t> xs) {
    // dbg(xs);
    int num_heads = config.num_heads;
    int num_kv_heads = config.num_kv_heads;
    int head_dim = config.hidden_size / num_heads;
    float rms_norm_eps = config.rms_norm_eps;

    assert(xs.size() == 9);
    auto x = xs[0];
    auto key_cache = xs[1];
    auto value_cache = xs[2];
    auto position_ids = xs[3];
    auto slot_mapping = xs[4];
    auto block_tables = xs[5];
    auto context_lens = xs[6];
    auto cos_cache = xs[7];
    auto sin_cache = xs[8];

    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;
    int hidden_size = num_heads * head_dim;

    auto qkv_proj = add_linear(x, false, true, identity_reshape_func);
    auto split = add_split(qkv_proj, 1, {q_size, kv_size, kv_size});
    auto q = split[0];
    auto k = split[1];
    auto v = split[2];
    auto q_reshape_func = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      // q: [bs, q_size] -> [bs, num_heads, head_dim]
      assert(old_shape.dimNum == 2);
      assert(old_shape.dims[1] == q_size);
      new_shape.dimNum = 3;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = num_heads;
      new_shape.dims[2] = head_dim;
    };
    auto kv_reshape_func = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      // kv: [bs, kv_size] -> [bs, num_kv_heads, head_dim]
      assert(old_shape.dimNum == 2);
      assert(old_shape.dims[1] == kv_size);
      new_shape.dimNum = 3;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = num_kv_heads;
      new_shape.dims[2] = head_dim;
    };
    auto q_norm = add_rmsnorm(q, std::nullopt, rms_norm_eps, q_reshape_func);
    auto k_norm = add_rmsnorm(k, std::nullopt, rms_norm_eps, kv_reshape_func);

    auto qk_rope = add_rope(q_norm[0], k_norm[0], position_ids, cos_cache, sin_cache, identity_reshape_func, identity_reshape_func);

    auto attn_out = add_paged_attn(
      qk_rope[0],
      qk_rope[1],
      v,
      key_cache,
      value_cache,
      position_ids,
      slot_mapping,
      block_tables,
      context_lens,
      identity_reshape_func,
      identity_reshape_func,
      kv_reshape_func
    );

    auto x_reshape_back_func = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      // x: [bs, num_heads, head_dim] -> [bs, hidden_size]
      assert(old_shape.dimNum == 3);
      assert(old_shape.dims[1] == num_heads);
      assert(old_shape.dims[2] == head_dim);
      new_shape.dimNum = 2;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = hidden_size;
    };
    auto y = add_linear(attn_out, false, true, x_reshape_back_func);
    return y;
  }

  std::vector<uint32_t> add_rope(uint32_t q, uint32_t k, uint32_t position_ids, uint32_t cos_cache, uint32_t sin_cache, atb::ReshapeFunc q_reshape_func, atb::ReshapeFunc k_reshape_func) {
    // dbg(q, k, position_ids, cos_cache, sin_cache);
    uint32_t out_q = tensor_num++;
    uint32_t out_k = tensor_num++;
    atb::Node node;
    node.operation = new RopeEx();
    node.inTensorIds = {q, k, position_ids, cos_cache, sin_cache};
    node.outTensorIds = {out_q, out_k};
    node.inTensorReshapeFuncs = {q_reshape_func, k_reshape_func, identity_reshape_func, identity_reshape_func, identity_reshape_func};
    graph_param.nodes.push_back(node);
    internal_ids.push_back(out_q);
    internal_ids.push_back(out_k);
    return {out_q, out_k};
  }


  uint32_t add_paged_attn(
    uint32_t q,
    uint32_t k,
    uint32_t v,
    uint32_t key_cache,
    uint32_t value_cache,
    uint32_t position_ids,
    uint32_t slot_mapping,
    uint32_t block_tables,
    uint32_t context_lens,
    atb::ReshapeFunc q_reshape_func,
    atb::ReshapeFunc k_reshape_func,
    atb::ReshapeFunc v_reshape_func
  ) {
    // dbg(q, k, v, key_cache, value_cache, position_ids, slot_mapping, block_tables, context_lens);
    int num_heads = config.num_heads;
    int num_kv_heads = config.num_kv_heads;
    int head_dim = config.hidden_size / num_heads;
    float scale_value = 1.0f / std::sqrt(head_dim);

    atb::Node cache_node;
    atb::infer::ReshapeAndCacheParam cache_param;
    cache_param.compressType = atb::infer::ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED;
    // cache_param.kvCacheCfg = atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE;
    cache_param.kvCacheCfg = atb::infer::ReshapeAndCacheParam::K_CACHE_V_CACHE_NZ;
    CHECK_ATB(atb::CreateOperation(cache_param, &cache_node.operation));
    cache_node.inTensorIds = {k, v, key_cache, value_cache, slot_mapping};
    cache_node.outTensorIds = {key_cache, value_cache};
    cache_node.inTensorReshapeFuncs = {
      k_reshape_func,
      v_reshape_func,
      identity_reshape_func,
      identity_reshape_func,
      identity_reshape_func
    };
    graph_param.nodes.push_back(cache_node);


    // paged attn
    uint32_t y = tensor_num++;
    atb::Node paged_attn_node;
    atb::infer::PagedAttentionParam paged_attn_param;
    paged_attn_param.headNum = num_heads;
    paged_attn_param.qkScale = scale_value;
    paged_attn_param.kvHeadNum = num_kv_heads;
    CHECK_ATB(atb::CreateOperation(paged_attn_param, &paged_attn_node.operation));
    paged_attn_node.inTensorIds = {q, key_cache, value_cache, block_tables, context_lens};
    paged_attn_node.outTensorIds = {y};
    paged_attn_node.inTensorReshapeFuncs = {
      q_reshape_func,
      identity_reshape_func,
      identity_reshape_func,
      identity_reshape_func,
      identity_reshape_func
    };
    graph_param.nodes.push_back(paged_attn_node);
    internal_ids.push_back(y);
    return y;
  }

  std::vector<uint32_t> add_split(uint32_t x, int dim, std::vector<int> sizes) {
    // dbg(x, dim, sizes);
    atb::Node node;
    atb::infer::SplitParam param;
    param.splitDim = dim;
    param.splitNum = (int32_t)sizes.size();
    atb::SVector<int> split_sizes;
    for (auto& size: sizes) {
      split_sizes.push_back(size);
    }
    param.splitSizes = split_sizes;
    CHECK_ATB(atb::CreateOperation(param, &node.operation));

    std::vector<uint32_t> ys(sizes.size());
    atb::SVector<uint32_t> ys_vec;
    for (int i = 0; i < sizes.size(); i++) {
      ys[i] = tensor_num++;
      ys_vec.push_back(ys[i]);
    }
    node.inTensorIds = {x};
    node.outTensorIds = ys_vec;
    graph_param.nodes.push_back(node);
    for (auto& y: ys) {
      internal_ids.push_back(y);
    }
    return ys;
  }

  std::vector<uint32_t> add_rmsnorm(uint32_t x, std::optional<uint32_t> residual, float eps, atb::ReshapeFunc x_reshape_func) {
    // dbg(x, residual, eps);
    if (residual.has_value()) {
      uint32_t y_add = tensor_num++;
      atb::Node add_node;
      atb::infer::ElewiseParam add_param;
      add_param.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;
      CHECK_ATB(atb::CreateOperation(add_param, &add_node.operation));
      add_node.inTensorIds = {x, residual.value()};
      add_node.outTensorIds = {y_add};
      add_node.inTensorReshapeFuncs = {x_reshape_func, identity_reshape_func};
      graph_param.nodes.push_back(add_node);
      internal_ids.push_back(y_add);

      uint32_t rmsnorm_w = tensor_num++;
      uint32_t y = tensor_num++;
      atb::Node rmsnorm_node;
      atb::infer::RmsNormParam rmsnorm_param;
      rmsnorm_param.layerType = atb::infer::RmsNormParam::RMS_NORM_NORM;
      rmsnorm_param.normParam.epsilon = eps;
      CHECK_ATB(atb::CreateOperation(rmsnorm_param, &rmsnorm_node.operation));
      rmsnorm_node.inTensorIds = {y_add, rmsnorm_w};
      rmsnorm_node.outTensorIds = {y};
      graph_param.nodes.push_back(rmsnorm_node);
      in_ids.push_back(rmsnorm_w);
      internal_ids.push_back(y);

      return {y, y_add};
    } else {
      uint32_t rmsnorm_w = tensor_num++;
      uint32_t y = tensor_num++;
      atb::Node node;
      atb::infer::RmsNormParam param;
      param.layerType = atb::infer::RmsNormParam::RMS_NORM_NORM;
      param.normParam.epsilon = eps;
      CHECK_ATB(atb::CreateOperation(param, &node.operation));
      node.inTensorIds = {x, rmsnorm_w};
      node.outTensorIds = {y};
      node.inTensorReshapeFuncs = {x_reshape_func, identity_reshape_func};
      graph_param.nodes.push_back(node);
      in_ids.push_back(rmsnorm_w);
      internal_ids.push_back(y);
      return {y};
    }
  }

  uint32_t add_swiglu(uint32_t x) {
    // dbg(x);
    uint32_t y = tensor_num++;
    atb::Node node;
    // FIXME: maybe memory leak
    node.operation = new SwiGluEx();
    node.inTensorIds = {x};
    node.outTensorIds = {y};
    graph_param.nodes.push_back(node);

    internal_ids.push_back(y);
    return y;
  }

  uint32_t add_linear(uint32_t x, bool trans_a, bool trans_b, atb::ReshapeFunc x_reshape_func) {
    // dbg(x, trans_a, trans_b);
    atb::Node node;

    atb::infer::LinearParam param;
    param.transposeA = trans_a;
    param.transposeB = trans_b;
    param.hasBias = false;
    param.outDataType = ACL_DT_UNDEFINED;

    CHECK_ATB(atb::CreateOperation(param, &node.operation));

    uint32_t w = tensor_num++;
    uint32_t y = tensor_num++;

    node.inTensorIds = {x, w};
    node.outTensorIds = {y};
    node.inTensorReshapeFuncs = {x_reshape_func, identity_reshape_func};
    graph_param.nodes.push_back(node);

    in_ids.push_back(w);
    internal_ids.push_back(y);

    return y;
  }


};

class Graph {
public:
  std::vector<atb::Operation*> ops;
  std::vector<uint32_t> in_tensor_nums;
  std::vector<uint32_t> weight_nums;
  std::vector<uint32_t> out_tensor_nums;

  GraphConfig config;

  Graph(GraphConfig config) : config(config) {
    config.display();
  }

  ~Graph() {
    for (auto& op: ops) {
      if (op) {
        assert(atb::DestroyOperation(op) == atb::NO_ERROR);
      }
    }
  }

  void build_model(int num_split) {
    int num_layers = config.num_layers;
    assert(num_layers >= num_split);
    int num_layers_per_split = (num_layers + num_split - 1) / num_split;
    assert(num_layers_per_split > 0);
    dbg(num_layers_per_split);
    for (int split_id = 0; split_id < num_split; split_id++) {
      int start_layer = split_id * num_layers_per_split;
      int end_layer = std::min(start_layer + num_layers_per_split, num_layers);
      dbg(start_layer, end_layer);

      GraphBuilder builder(config);
      int input_num = 0;
      if (start_layer == 0) {
        // input: token_ids, [key_cache, value_cache] * layer_num, position_ids, slot_mapping, block_tables, context_lens, cos_cache, sin_cache
        input_num = 1 + 2 * (end_layer - start_layer) + 4 + 2;
      } else {
        // input: hidden_states, residual, [key_cache, value_cache] * layer_num, position_ids, slot_mapping, block_tables, context_lens, cos_cache, sin_cache
        input_num = 1 + 1 + 2 * (end_layer - start_layer) + 4 + 2;
      }

      std::string name = "split_" + std::to_string(split_id);
      auto op = builder.build(name.c_str(), input_num, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
        assert(xs.size() == input_num);
        uint32_t token_ids = uint32_t(-1);
        uint32_t hidden_states = uint32_t(-1);
        std::optional<uint32_t> residual = std::nullopt;
        std::vector<uint32_t> key_caches;
        std::vector<uint32_t> value_caches;
        uint32_t position_ids = uint32_t(-1);
        uint32_t slot_mapping = uint32_t(-1);
        uint32_t block_tables = uint32_t(-1);
        uint32_t context_lens = uint32_t(-1);
        uint32_t cos_cache = uint32_t(-1);
        uint32_t sin_cache = uint32_t(-1);
        if (start_layer == 0) {
          token_ids = xs[0];
          for (int i = 0; i < end_layer - start_layer; i++) {
            key_caches.push_back(xs[1 + i * 2]);
            value_caches.push_back(xs[2 + i * 2]);
          }
          position_ids = xs[1 + 2 * (end_layer - start_layer)];
          slot_mapping = xs[2 + 2 * (end_layer - start_layer)];
          block_tables = xs[3 + 2 * (end_layer - start_layer)];
          context_lens = xs[4 + 2 * (end_layer - start_layer)];
          cos_cache = xs[5 + 2 * (end_layer - start_layer)];
          sin_cache = xs[6 + 2 * (end_layer - start_layer)];

          // build pre-layer
          hidden_states = builder.add_embedding(token_ids);
        } else {
          hidden_states = xs[0];
          residual = xs[1];
          for (int i = 0; i < end_layer - start_layer; i++) {
            key_caches.push_back(xs[2 + i * 2]);
            value_caches.push_back(xs[3 + i * 2]);
          }
          position_ids = xs[2 + 2 * (end_layer - start_layer)];
          slot_mapping = xs[3 + 2 * (end_layer - start_layer)];
          block_tables = xs[4 + 2 * (end_layer - start_layer)];
          context_lens = xs[5 + 2 * (end_layer - start_layer)];
          cos_cache = xs[6 + 2 * (end_layer - start_layer)];
          sin_cache = xs[7 + 2 * (end_layer - start_layer)];
        }

        // build
        for (int i = 0; i < end_layer - start_layer; i++) {
          auto hidden_states_and_residual = builder.add_decoder_layer(
            hidden_states,
            residual,
            key_caches[i],
            value_caches[i],
            position_ids,
            slot_mapping,
            block_tables,
            context_lens,
            cos_cache,
            sin_cache
          );
          assert(hidden_states_and_residual.size() == 2);
          hidden_states = hidden_states_and_residual[0];
          residual = hidden_states_and_residual[1];
        }

        // build post-layer
        if (end_layer == config.num_layers) {
          auto y_and_residual = builder.add_rmsnorm(hidden_states, residual, config.rms_norm_eps, builder.identity_reshape_func);
          assert(y_and_residual.size() == 2);
          auto y = y_and_residual[0];
          return {y};
        } else {
          return {hidden_states, residual.value()};
        }
      });

      ops.push_back(op);
      in_tensor_nums.push_back(input_num);
      weight_nums.push_back(builder.in_ids.size() - input_num);
      out_tensor_nums.push_back(builder.out_ids.size());
      dbg(split_id, in_tensor_nums.back(), weight_nums.back(), out_tensor_nums.back());
    }
  }

  void build_embedding() {
    GraphBuilder builder(config);
    auto op = builder.build("embedding", 1, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 1);
      auto y = builder.add_embedding(xs[0]);
      return {y};
    });

    ops.push_back(op);
    in_tensor_nums.push_back(1);
    weight_nums.push_back(builder.in_ids.size() - 1);
    out_tensor_nums.push_back(builder.out_ids.size());
    dbg(in_tensor_nums.back(), weight_nums.back(), out_tensor_nums.back());
  }

  void build_mlp() {
    GraphBuilder builder(config);
    auto op = builder.build("mlp", 1, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 1);
      auto y = builder.add_mlp(xs[0]);
      return {y};
    });

    ops.push_back(op);
    in_tensor_nums.push_back(1);
    weight_nums.push_back(builder.in_ids.size() - 1);
    out_tensor_nums.push_back(builder.out_ids.size());
    dbg(in_tensor_nums.back(), weight_nums.back(), out_tensor_nums.back());
  }

  void build_attn() {
    GraphBuilder builder(config);
    auto op = builder.build("attn", 9, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      // x, key_cache, value_cache, position_ids, slot_mapping, block_tables, context_lens
      assert(xs.size() == 9);
      auto y = builder.add_attn(xs);
      return {y};
    });

    ops.push_back(op);
    in_tensor_nums.push_back(9);
    weight_nums.push_back(builder.in_ids.size() - 9);
    out_tensor_nums.push_back(builder.out_ids.size());
    dbg(in_tensor_nums.back(), weight_nums.back(), out_tensor_nums.back());
  }

  void build_paged_attn() {
    GraphBuilder builder(config);
    auto op = builder.build("paged_attn", 9, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 9);
      auto q = xs[0];
      auto k = xs[1];
      auto v = xs[2];
      auto key_cache = xs[3];
      auto value_cache = xs[4];
      auto position_ids = xs[5];
      auto slot_mapping = xs[6];
      auto block_tables = xs[7];
      auto context_lens = xs[8];

      auto attn_out = builder.add_paged_attn(
        q,
        k,
        v,
        key_cache,
        value_cache,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
        builder.identity_reshape_func,
        builder.identity_reshape_func,
        builder.identity_reshape_func
      );

      return {attn_out};
    });

    ops.push_back(op);
    in_tensor_nums.push_back(9);
    weight_nums.push_back(builder.in_ids.size() - 9);
    out_tensor_nums.push_back(builder.out_ids.size());
    dbg(in_tensor_nums.back(), weight_nums.back(), out_tensor_nums.back());
  }

  void build_rope() {
    GraphBuilder builder(config);
    auto op = builder.build("rope", 5, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 5);
      auto ys = builder.add_rope(xs[0], xs[1], xs[2], xs[3], xs[4], builder.identity_reshape_func, builder.identity_reshape_func);
      return ys;
    });

    ops.push_back(op);
    in_tensor_nums.push_back(5);
    weight_nums.push_back(builder.in_ids.size() - 5);
    out_tensor_nums.push_back(builder.out_ids.size());
    dbg(in_tensor_nums.back(), weight_nums.back(), out_tensor_nums.back());
  }


  void build_rmsnorm() {
    GraphBuilder builder(config);
    auto op = builder.build("rmsnorm", 1, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 1);
      auto ys = builder.add_rmsnorm(xs[0], std::nullopt, builder.config.rms_norm_eps, builder.identity_reshape_func);
      return ys;
    });

    ops.push_back(op);
    in_tensor_nums.push_back(1);
    weight_nums.push_back(builder.in_ids.size() - 1);
    out_tensor_nums.push_back(builder.out_ids.size());
    dbg(in_tensor_nums.back(), weight_nums.back(), out_tensor_nums.back());
  }

  void build_rmsnorm_with_residual() {
    GraphBuilder builder(config);
    auto op = builder.build("rmsnorm_with_residual", 2, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 2);
      auto ys = builder.add_rmsnorm(xs[0], xs[1], builder.config.rms_norm_eps, builder.identity_reshape_func);
      return ys;
    });

    ops.push_back(op);
    in_tensor_nums.push_back(2);
    weight_nums.push_back(builder.in_ids.size() - 2);
    out_tensor_nums.push_back(builder.out_ids.size());
    dbg(in_tensor_nums.back(), weight_nums.back(), out_tensor_nums.back());
  }

};


class Context {
public:
  std::shared_ptr<atb::Context> ctx;
  std::vector<atb::VariantPack> packs;

  std::vector<std::vector<at::Tensor>> intermediate_tensors; // for splited graph
  std::vector<uint64_t> workspace_sizes;
  uint64_t max_workspace_size = 0;

  Context() {
    atb::Context* raw = nullptr;
    CHECK_ATB(atb::CreateContext(&raw));
    printf("Context created %p\n", raw);
    ctx.reset(raw, [](atb::Context* p) {
      if (p) {
        printf("Context destroyed %p\n", p);
        atb::DestroyContext(p);
      }
    });
  }
};

void init_ffi_graph(py::module_ &&m) {
  py::class_<GraphConfig>(m, "GraphConfig")
    .def(py::init<>())
    .def_readwrite("batch_size", &GraphConfig::batch_size)
    .def_readwrite("hidden_size", &GraphConfig::hidden_size)
    .def_readwrite("num_heads", &GraphConfig::num_heads)
    .def_readwrite("num_kv_heads", &GraphConfig::num_kv_heads)
    .def_readwrite("intermediate_size", &GraphConfig::intermediate_size)
    .def_readwrite("num_layers", &GraphConfig::num_layers)
    .def_readwrite("rms_norm_eps", &GraphConfig::rms_norm_eps);

  py::class_<Graph>(m, "Graph")
    .def(py::init<GraphConfig>())
    .def("build_model", &Graph::build_model)
    .def("build_embedding", &Graph::build_embedding)
    .def("build_mlp", &Graph::build_mlp)
    .def("build_paged_attn", &Graph::build_paged_attn)
    .def("build_rope", &Graph::build_rope)
    .def("build_rmsnorm", &Graph::build_rmsnorm)
    .def("build_rmsnorm_with_residual", &Graph::build_rmsnorm_with_residual)
    .def("build_attn", &Graph::build_attn);


  py::class_<Context>(m, "Context")
    .def(py::init<>())
    .def("setup", [](
      Context& self,
      Graph& graph,
      at::Tensor token_ids,
      std::vector<at::Tensor> key_caches,
      std::vector<at::Tensor> value_caches,
      at::Tensor position_ids,
      at::Tensor slot_mapping,
      at::Tensor block_tables,
      at::Tensor context_lens,
      at::Tensor cos_cache,
      at::Tensor sin_cache,
      std::vector<at::Tensor> weights,
      at::Tensor out
    ) -> uint64_t {
      pybind11::gil_scoped_release gil_release;
      aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
      dbg(stream);
      CHECK_ATB(self.ctx->SetExecuteStream(stream));
      if (key_caches.size() != graph.config.num_layers) {
        std::stringstream ss;
        ss << "key_caches size mismatch, expected " << graph.config.num_layers << ", got " << key_caches.size();
        throw std::runtime_error(ss.str());
      }
      if (value_caches.size() != graph.config.num_layers) {
        std::stringstream ss;
        ss << "value_caches size mismatch, expected " << graph.config.num_layers << ", got " << value_caches.size();
        throw std::runtime_error(ss.str());
      }
      if (position_ids.size(0) != graph.config.batch_size) {
        std::stringstream ss;
        ss << "position_ids size mismatch, expected " << graph.config.batch_size << ", got " << position_ids.size(0);
        throw std::runtime_error(ss.str());
      }
      if (slot_mapping.size(0) != graph.config.batch_size) {
        std::stringstream ss;
        ss << "slot_mapping size mismatch, expected " << graph.config.batch_size << ", got " << slot_mapping.size(0);
        throw std::runtime_error(ss.str());
      }
      if (block_tables.size(0) != graph.config.batch_size) {
        std::stringstream ss;
        ss << "block_tables size mismatch, expected " << graph.config.batch_size << ", got " << block_tables.size(0);
        throw std::runtime_error(ss.str());
      }
      if (context_lens.size(0) != graph.config.batch_size) {
        std::stringstream ss;
        ss << "context_lens size mismatch, expected " << graph.config.batch_size << ", got " << context_lens.size(0);
        throw std::runtime_error(ss.str());
      }

      int total_weight_num = std::accumulate(graph.weight_nums.begin(), graph.weight_nums.end(), 0);
      if (weights.size() != total_weight_num) {
        std::stringstream ss;
        ss << "weights size mismatch, expected " << total_weight_num << ", got " << weights.size();
        throw std::runtime_error(ss.str());
      }

      int num_split = graph.ops.size();
      int num_layers_per_split = (graph.config.num_layers + num_split - 1) / num_split;
      dbg(num_split, num_layers_per_split);
      std::map<torch::ScalarType, aclDataType> dtype_map = {
        {torch::kFloat16, ACL_FLOAT16},
        {torch::kInt32, ACL_INT32},
        {torch::kInt64, ACL_INT64}
      };
      auto to_atb_tensor = [&](at::Tensor& x, aclFormat format) -> atb::Tensor {
        atb::Tensor y;
        y.desc.dtype = dtype_map[x.scalar_type()];
        y.desc.format = format;
        y.desc.shape.dimNum = x.dim();
        for (int i = 0; i < x.dim(); i++) {
          y.desc.shape.dims[i] = x.size(i);
        }
        y.dataSize = x.numel() * x.element_size();
        y.deviceData = x.data_ptr();
        return y;
      };

      self.packs.clear();
      self.intermediate_tensors.clear();
      self.workspace_sizes.clear();
      int weight_num_offset = 0;
      for (int split_id = 0; split_id < num_split; split_id++) {
        int start_layer = split_id * num_layers_per_split;
        int end_layer = std::min(start_layer + num_layers_per_split, graph.config.num_layers);

        atb::VariantPack pack;
        pack.inTensors.reserve(graph.in_tensor_nums[split_id] + graph.weight_nums[split_id]);
        if (split_id == 0) {
          pack.inTensors.push_back(to_atb_tensor(token_ids, ACL_FORMAT_ND));
        } else {
          assert(self.intermediate_tensors[split_id - 1].size() == 2);
          // hidden states
          pack.inTensors.push_back(to_atb_tensor(self.intermediate_tensors[split_id - 1][0], ACL_FORMAT_ND));
          // residual
          pack.inTensors.push_back(to_atb_tensor(self.intermediate_tensors[split_id - 1][1], ACL_FORMAT_ND));

        }
        for (int i = 0; i < end_layer - start_layer; i++) {
          pack.inTensors.push_back(to_atb_tensor(key_caches[start_layer + i], ACL_FORMAT_FRACTAL_NZ));
          pack.inTensors.push_back(to_atb_tensor(value_caches[start_layer + i], ACL_FORMAT_FRACTAL_NZ));
        }
        pack.inTensors.push_back(to_atb_tensor(position_ids, ACL_FORMAT_ND));
        pack.inTensors.push_back(to_atb_tensor(slot_mapping, ACL_FORMAT_ND));
        pack.inTensors.push_back(to_atb_tensor(block_tables, ACL_FORMAT_ND));
        pack.inTensors.push_back(to_atb_tensor(context_lens, ACL_FORMAT_ND));
        pack.inTensors.push_back(to_atb_tensor(cos_cache, ACL_FORMAT_ND));
        pack.inTensors.push_back(to_atb_tensor(sin_cache, ACL_FORMAT_ND));
        assert(pack.inTensors.size() == graph.in_tensor_nums[split_id]);

        // weight
        int weight_num = graph.weight_nums[split_id];
        dbg(weight_num, weight_num_offset, pack.inTensors.size());
        for (int i = weight_num_offset; i < weight_num_offset + weight_num; i++) {
          assert(i < weights.size());
          pack.inTensors.push_back(to_atb_tensor(weights[i], ACL_FORMAT_ND));
        }
        weight_num_offset += weight_num;
        assert(pack.inTensors.size() == graph.in_tensor_nums[split_id] + graph.weight_nums[split_id]);

        // output
        if (split_id < num_split - 1) {
          auto options = at::TensorOptions().dtype(key_caches[0].scalar_type()).device(key_caches[0].device());
          auto y = at::empty({graph.config.batch_size, graph.config.hidden_size}, options);
          auto y_residual = at::empty({graph.config.batch_size, graph.config.hidden_size}, options);
          self.intermediate_tensors.push_back({y, y_residual});
          pack.outTensors.push_back(to_atb_tensor(y, ACL_FORMAT_ND));
          pack.outTensors.push_back(to_atb_tensor(y_residual, ACL_FORMAT_ND));
        } else {
          pack.outTensors.push_back(to_atb_tensor(out, ACL_FORMAT_ND));
        }
        assert(pack.outTensors.size() == graph.out_tensor_nums[split_id]);

        uint64_t workspace_size = 0;
        CHECK_ATB(graph.ops[split_id]->Setup(pack, workspace_size, self.ctx.get()));
        dbg(split_id, workspace_size);
        self.workspace_sizes.push_back(workspace_size);
        self.packs.push_back(pack);
      }
      assert(weight_num_offset == total_weight_num);

      if (self.workspace_sizes.size() != num_split) {
        std::stringstream ss;
        ss << "workspace_sizes size mismatch, expected " << num_split << ", got " << self.workspace_sizes.size();
        throw std::runtime_error(ss.str());
      }
      auto _max_workspace_size = std::max_element(self.workspace_sizes.begin(), self.workspace_sizes.end());
      self.max_workspace_size = *_max_workspace_size;
      dbg(self.max_workspace_size);
      if (self.packs.size() != num_split) {
        std::stringstream ss;
        ss << "packs size mismatch, expected " << num_split << ", got " << self.packs.size();
        throw std::runtime_error(ss.str());
      }
      return self.max_workspace_size;
    })
    .def("setup_fullgraph", [](Context& self, Graph& graph, std::vector<at::Tensor>& inputs, std::vector<int> input_formats, std::vector<at::Tensor>& weights, std::vector<at::Tensor>& outputs) -> uint64_t {
      pybind11::gil_scoped_release gil_release;
      aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
      dbg(stream);
      CHECK_ATB(self.ctx->SetExecuteStream(stream));
      if (graph.ops.size() != 1) {
        std::stringstream ss;
        ss << "graph ops size mismatch, expected 1, got " << graph.ops.size();
        throw std::runtime_error(ss.str());
      }
      if (graph.in_tensor_nums[0] != inputs.size()) {
        std::stringstream ss;
        ss << "graph inputs size mismatch, expected " << graph.in_tensor_nums[0] << ", got " << inputs.size();
        throw std::runtime_error(ss.str());
      }
      if (graph.weight_nums[0] != weights.size()) {
        std::stringstream ss;
        ss << "graph weights size mismatch, expected " << graph.weight_nums[0] << ", got " << weights.size();
        throw std::runtime_error(ss.str());
      }
      if (graph.out_tensor_nums[0] != outputs.size()) {
        std::stringstream ss;
        ss << "graph outputs size mismatch, expected " << graph.out_tensor_nums[0] << ", got " << outputs.size();
        throw std::runtime_error(ss.str());
      }

      std::map<torch::ScalarType, aclDataType> dtype_map = {
        {torch::kFloat16, ACL_FLOAT16},
        {torch::kInt32, ACL_INT32},
        {torch::kInt64, ACL_INT64}
      };

      self.packs.clear();
      self.workspace_sizes.clear();

      assert(input_formats.size() == inputs.size());
      atb::VariantPack pack;
      for (int i = 0; i < inputs.size(); i++) {
        auto &input = inputs[i];
        atb::Tensor x;
        x.desc.dtype = dtype_map[input.scalar_type()];
        x.desc.format = (aclFormat)input_formats[i];
        x.desc.shape.dimNum = input.dim();
        for (int i = 0; i < input.dim(); i++) {
          x.desc.shape.dims[i] = input.size(i);
        }
        x.dataSize = input.numel() * input.element_size();
        x.deviceData = input.data_ptr();
        pack.inTensors.push_back(x);
      }
      for (auto &weight: weights) {
        atb::Tensor x;
        x.desc.dtype = dtype_map[weight.scalar_type()];
        x.desc.format = ACL_FORMAT_ND;
        x.desc.shape.dimNum = weight.dim();
        for (int i = 0; i < weight.dim(); i++) {
          x.desc.shape.dims[i] = weight.size(i);
        }
        x.dataSize = weight.numel() * weight.element_size();
        x.deviceData = weight.data_ptr();
        pack.inTensors.push_back(x);
      }
      for (auto &output: outputs) {
        atb::Tensor x;
        x.desc.dtype = dtype_map[output.scalar_type()];
        x.desc.format = ACL_FORMAT_ND;
        x.desc.shape.dimNum = output.dim();
        for (int i = 0; i < output.dim(); i++) {
          x.desc.shape.dims[i] = output.size(i);
        }
        x.dataSize = output.numel() * output.element_size();
        x.deviceData = output.data_ptr();
        pack.outTensors.push_back(x);
      }
      uint64_t workspace_size = 0;
      CHECK_ATB(graph.ops[0]->Setup(pack, workspace_size, self.ctx.get()));
      dbg(workspace_size);
      self.max_workspace_size = workspace_size;
      dbg(self.max_workspace_size);

      self.workspace_sizes.push_back(workspace_size);
      self.packs.push_back(pack);

      return self.max_workspace_size;
    })
    .def("run", [](Context& self, Graph& graph, at::Tensor workspace) {
      // FIXME: it will cause out of bound error when re-run with the same graph
      // So we need to use run_with_dummy_setup
      dbg("[warning] use run_with_dummy_setup instead to support re-run");
      pybind11::gil_scoped_release gil_release;
      for (int i = 0; i < graph.ops.size(); i++) {
        CHECK_ATB(graph.ops[i]->Execute(self.packs[i], workspace.data_ptr<uint8_t>(), self.workspace_sizes[i], self.ctx.get()));
      }
    })
    .def("run_with_dummy_setup", [](Context& self, Graph& graph, at::Tensor workspace) {
      pybind11::gil_scoped_release gil_release;
      dbg("run_with_dummy_setup resetup");
      aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
      CHECK_ATB(self.ctx->SetExecuteStream(stream));
      assert(graph.ops.size() == self.packs.size());
      assert(graph.ops.size() == self.workspace_sizes.size());
      for (int i = 0; i < graph.ops.size(); i++) {
        uint64_t workspace_size = 0;
        CHECK_ATB(graph.ops[i]->Setup(self.packs[i], workspace_size, self.ctx.get()));
        if (workspace_size != self.workspace_sizes[i]) {
          std::stringstream ss;
          ss << "workspace_sizes size mismatch, expected " << self.workspace_sizes[i] << ", got " << workspace_size;
          throw std::runtime_error(ss.str());
        }
      }
      dbg("run_with_dummy_setup execute");
      for (int i = 0; i < graph.ops.size(); i++) {
        CHECK_ATB(graph.ops[i]->Execute(self.packs[i], workspace.data_ptr<uint8_t>(), self.workspace_sizes[i], self.ctx.get()));
      }
    });

}

}
