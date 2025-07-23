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

class Graph {
public:
  atb::GraphOpBuilder* builder;
  atb::Operation* graph;
  atb::GraphParam graph_param;
  std::vector<std::unique_ptr<AclnnOp>> aclnn_ops;

  uint32_t tensor_num;
  std::vector<uint32_t> in_ids;
  std::vector<uint32_t> internal_ids;
  std::vector<uint32_t> out_ids;
  std::map<uint32_t, uint32_t> id_map;

  atb::ReshapeFunc identity_reshape_func = [](const atb::Dims& old_shape, atb::Dims& new_shape) {
    new_shape = old_shape;
  };

  Graph() : builder(nullptr), graph(nullptr), tensor_num(0) {}

  void clear() {
    in_ids.clear();
    internal_ids.clear();
    out_ids.clear();
    tensor_num = 0;

    if (builder) {
      CHECK_ATB(atb::DestroyGraphOpBuilder(builder));
      builder = nullptr;
    }

    for (auto& node: graph_param.nodes) {
      if (node.operation) {
        CHECK_ATB(atb::DestroyOperation(node.operation));
      }
    }
    aclnn_ops.clear();
    graph_param.nodes.clear();
    graph_param.inferShapeFunc = nullptr;
    if (graph) {
      CHECK_ATB(atb::DestroyOperation(graph));
      graph = nullptr;
    }
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

  void build_model(GraphConfig config) {
    int num_layers = config.num_layers;
    build(7, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      // token_ids, key_cache, value_cache, position_ids, slot_mapping, block_tables, context_lens
      auto token_ids = xs[0];
      auto key_cache = xs[1];
      auto value_cache = xs[2];
      auto position_ids = xs[3];
      auto slot_mapping = xs[4];
      auto block_tables = xs[5];
      auto context_lens = xs[6];

      int head_dim = config.hidden_size / config.num_heads;
      float scale_value = 1.0f / std::sqrt(head_dim);

      auto x = add_embedding(token_ids);
      uint32_t residual = uint32_t(-1);
      for (int i = 0; i < num_layers; i++) {
        if (i == 0) {
          residual = x;
          x = add_rmsnorm(x, std::nullopt, config.rms_norm_eps, identity_reshape_func)[0];
        } else {
          auto x_and_residual = add_rmsnorm(x, residual, config.rms_norm_eps, identity_reshape_func);
          assert(x_and_residual.size() == 2);
          x = x_and_residual[0];
          residual = x_and_residual[1];
        }

        x = add_attn(
          {x, key_cache, value_cache, position_ids, slot_mapping, block_tables, context_lens},
          config.num_heads,
          config.num_kv_heads,
          head_dim,
          scale_value,
          config.rms_norm_eps
        );

        auto x_and_residual = add_rmsnorm(x, residual, config.rms_norm_eps, identity_reshape_func);
        assert(x_and_residual.size() == 2);
        x = x_and_residual[0];
        residual = x_and_residual[1];

        x = add_mlp(x);
      }

      // final norm
      auto x_and_residual = add_rmsnorm(x, residual, config.rms_norm_eps, identity_reshape_func);
      assert(x_and_residual.size() == 2);
      auto y = x_and_residual[0];
      return {y};
    });
  }

  void build_embedding(GraphConfig config) {
    build(1, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 1);
      auto y = add_embedding(xs[0]);
      return {y};
    });
  }

  void build_mlp(GraphConfig config) {
    build(1, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 1);
      auto y = add_mlp(xs[0]);
      return {y};
    });
  }

  void build_attn(GraphConfig config) {
    build(7, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      // x, key_cache, value_cache, position_ids, slot_mapping, block_tables, context_lens
      assert(xs.size() == 7);
      int head_dim = config.hidden_size / config.num_heads;
      float scale_value = 1.0f / std::sqrt(head_dim);
      auto y = add_attn(xs, config.num_heads, config.num_kv_heads, head_dim, scale_value, config.rms_norm_eps);
      return {y};
    });
  }

  void build_paged_attn(GraphConfig config) {
    build(9, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
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

      int head_dim = config.hidden_size / config.num_heads;
      int q_size = config.num_heads * head_dim;
      int kv_size = config.num_kv_heads * head_dim;
      float scale_value = 1.0f / std::sqrt(head_dim);

      auto attn_out = add_paged_attn(
        q,
        k,
        v,
        key_cache,
        value_cache,
        position_ids,
        slot_mapping,
        block_tables,
        context_lens,
        identity_reshape_func,
        identity_reshape_func,
        identity_reshape_func,
        config.num_heads,
        config.num_kv_heads,
        scale_value
      );

      return {attn_out};
    });
  }


  void build_rmsnorm(GraphConfig config) {
    build(1, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 1);
      auto ys = add_rmsnorm(xs[0], std::nullopt, config.rms_norm_eps, identity_reshape_func);
      return ys;
    });
  }

  void build_rmsnorm_with_residual(GraphConfig config) {
    build(2, [&](std::vector<uint32_t> xs) -> std::vector<uint32_t> {
      assert(xs.size() == 2);
      auto ys = add_rmsnorm(xs[0], xs[1], config.rms_norm_eps, identity_reshape_func);
      return ys;
    });
  }

  void build(int input_num, std::function<std::vector<uint32_t>(std::vector<uint32_t>)> build_fn) {
    clear();
    assert(builder == nullptr && "builder should be nullptr before build");
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
    dbg(in_ids, internal_ids, out_ids);
    dbg(xs, ys);
    assert(std::all_of(ys.begin(), ys.end(), [&](uint32_t y) {
      return std::find(internal_ids.end() - ys.size(), internal_ids.end(), y) != internal_ids.end();
    }));
    for (auto& y: ys) {
      internal_ids.pop_back();
      out_ids.push_back(y);
    }
    remap();

    graph_param.name = "graph";
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

    CHECK_ATB(atb::CreateOperation(graph_param, &graph));
    assert(graph != nullptr && "graph should not be nullptr");
    dbg(in_ids.size(), internal_ids.size(), out_ids.size());
  }

  uint32_t add_embedding(uint32_t token_ids) {
    dbg(token_ids);
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

  uint32_t add_attn(std::vector<uint32_t> xs, int num_heads, int num_kv_heads, int head_dim, float scale_value, float rms_norm_eps) {
    dbg(xs, num_heads, num_kv_heads, head_dim, scale_value, rms_norm_eps);
    assert(xs.size() == 7);
    auto x = xs[0];
    auto key_cache = xs[1];
    auto value_cache = xs[2];
    auto position_ids = xs[3];
    auto slot_mapping = xs[4];
    auto block_tables = xs[5];
    auto context_lens = xs[6];

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

    auto attn_out = add_paged_attn(
      q_norm[0],
      k_norm[0],
      v,
      key_cache,
      value_cache,
      position_ids,
      slot_mapping,
      block_tables,
      context_lens,
      identity_reshape_func,
      identity_reshape_func,
      kv_reshape_func,
      num_heads,
      num_kv_heads,
      scale_value
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
    atb::ReshapeFunc v_reshape_func,
    int num_heads,
    int num_kv_heads,
    float scale_value
  ) {
    dbg(q, k, v, key_cache, value_cache, position_ids, slot_mapping, block_tables, context_lens);
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
    dbg(x, dim, sizes);
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
    dbg(x, residual, eps);
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
    dbg(x);
    uint32_t y = tensor_num++;
    atb::Node node;
    aclnn_ops.push_back(std::make_unique<SwiGluEx>());
    node.operation = aclnn_ops.back().get();
    node.inTensorIds = {x};
    node.outTensorIds = {y};
    graph_param.nodes.push_back(node);

    internal_ids.push_back(y);
    return y;
  }

  uint32_t add_linear(uint32_t x, bool trans_a, bool trans_b, atb::ReshapeFunc x_reshape_func) {
    dbg(x, trans_a, trans_b);
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


class Context {
public:
  std::shared_ptr<atb::Context> ctx;
  uint64_t workspace_size = 0;
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
    .def(py::init<>())
    .def("build_model", &Graph::build_model)
    .def("build_embedding", &Graph::build_embedding)
    .def("build_mlp", &Graph::build_mlp)
    .def("build_paged_attn", &Graph::build_paged_attn)
    .def("build_rmsnorm", &Graph::build_rmsnorm)
    .def("build_rmsnorm_with_residual", &Graph::build_rmsnorm_with_residual)
    .def("build_attn", &Graph::build_attn);


  py::class_<Context>(m, "Context")
    .def(py::init<>())
    .def("setup", [](Context& self, Graph& graph, std::vector<at::Tensor>& inputs, std::vector<at::Tensor>& outputs) {
      pybind11::gil_scoped_release gil_release;
      aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
      CHECK_ATB(self.ctx->SetExecuteStream(stream));
      dbg(inputs.size(), outputs.size());

      atb::VariantPack pack;
      for (auto &input: inputs) {
        atb::Tensor x;
        x.desc.dtype = ACL_FLOAT16;
        x.desc.format = ACL_FORMAT_ND;
        x.desc.shape.dimNum = input.dim();
        for (int i = 0; i < input.dim(); i++) {
          x.desc.shape.dims[i] = input.size(i);
        }
        x.dataSize = input.numel() * input.element_size();
        x.deviceData = input.data_ptr();
        pack.inTensors.push_back(x);
      }
      for (auto &output: outputs) {
        atb::Tensor x;
        x.desc.dtype = ACL_FLOAT16;
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
      CHECK_ATB(graph.graph->Setup(pack, workspace_size, self.ctx.get()));
      dbg(workspace_size);
      self.workspace_size = workspace_size;
    })
    .def("run", [](Context& self, Graph& graph, std::vector<at::Tensor>& inputs, std::vector<at::Tensor>& outputs) {
      pybind11::gil_scoped_release gil_release;
      atb::VariantPack pack;
      for (auto &input: inputs) {
        atb::Tensor x;
        x.desc.dtype = ACL_FLOAT16;
        x.desc.format = ACL_FORMAT_ND;
        x.desc.shape.dimNum = input.dim();
        for (int i = 0; i < input.dim(); i++) {
          x.desc.shape.dims[i] = input.size(i);
        }
        x.dataSize = input.numel() * input.element_size();
        x.deviceData = input.data_ptr();
        pack.inTensors.push_back(x);
      }
      for (auto &output: outputs) {
        atb::Tensor x;
        x.desc.dtype = ACL_FLOAT16;
        x.desc.format = ACL_FORMAT_ND;
        x.desc.shape.dimNum = output.dim();
        for (int i = 0; i < output.dim(); i++) {
          x.desc.shape.dims[i] = output.size(i);
        }
        x.dataSize = output.numel() * output.element_size();
        x.deviceData = output.data_ptr();
        pack.outTensors.push_back(x);
      }
      auto options = at::TensorOptions().dtype(torch::kUInt8).device(inputs[0].device());
      auto workspace = at::empty({(int64_t)self.workspace_size}, options);

      CHECK_ATB(graph.graph->Execute(pack, workspace.data_ptr<uint8_t>(), self.workspace_size, self.ctx.get()));
    })
    .def("setup_then_run", [](Context& self, Graph& graph, std::vector<at::Tensor>& inputs, std::vector<int> input_formats, std::vector<at::Tensor>& weights, std::vector<at::Tensor>& outputs) {
      pybind11::gil_scoped_release gil_release;
      aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
      CHECK_ATB(self.ctx->SetExecuteStream(stream));
      if (inputs.size() + weights.size() != graph.in_ids.size()) {
        throw std::runtime_error("graph inputs size mismatch");
      }
      if (outputs.size() != graph.out_ids.size()) {
        throw std::runtime_error("graph outputs size mismatch");
      }

      std::map<torch::ScalarType, aclDataType> dtype_map = {
        {torch::kFloat16, ACL_FLOAT16},
        {torch::kInt32, ACL_INT32},
      };

      atb::VariantPack pack;
      assert(input_formats.size() == inputs.size());
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
      CHECK_ATB(graph.graph->Setup(pack, workspace_size, self.ctx.get()));
      auto options = at::TensorOptions().dtype(torch::kUInt8).device(inputs[0].device());
      auto workspace = at::empty({(int64_t)workspace_size}, options);
      dbg(workspace_size);
      CHECK_ATB(graph.graph->Execute(pack, workspace.data_ptr<uint8_t>(), workspace_size, self.ctx.get()));
    });
}

}
