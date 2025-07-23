// FIXME: must be included first.
#include "ge/ge_api.h"
#include "all_ops.h"
#include "op_proto.h"


#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <optional>

namespace py = pybind11;

namespace native {

void print_shape(ge::TensorDesc desc, const char* name, const char* end = "") {
  auto shape = desc.GetShape().GetDims();
  printf(" %s[", name);
  for (int i = 0; i < shape.size(); i++) {
    printf("%d", shape[i]);
    if (i < shape.size() - 1) {
      printf(",");
    } else {
      printf("] ");
    }
  }
  printf("%s", end);
}

class GraphBuilder {
public:
  GraphBuilder(ge::Graph& graph) : graph(graph) {
    inputs.clear();
    outputs.clear();
  }
  ge::Graph& graph;
  std::vector<ge::Operator> inputs;
  std::vector<ge::Operator> outputs;

  void build_attn(int bs, int hidden_size, int num_heads, int num_kv_heads) {
    printf("build_attn: bs: %d, hidden_size: %d, num_heads: %d, num_kv_heads: %d\n", bs, hidden_size, num_heads, num_kv_heads);
    auto x_desc = ge::TensorDesc(ge::Shape({bs, hidden_size}), ge::FORMAT_ND, ge::DT_FLOAT16);
    x_desc.SetPlacement(ge::Placement::kPlacementDevice);
    auto x_data = ge::op::Data("x").set_attr_index(inputs.size());
    x_data.update_input_desc_x(x_desc);
    x_data.update_output_desc_y(x_desc);
    inputs.push_back(x_data);

    auto outputs = add_attn({x_data, 0}, num_heads, num_kv_heads, "<layer0>");

    graph.SetInputs(inputs);
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs_with_idx;
    for (int i = 0; i < outputs.size(); i++) {
      outputs_with_idx.push_back({outputs[i], {0}});
    }
    graph.SetOutputs(outputs_with_idx);
  }

  void build_rmsnorm(int bs, int hidden_size) {
    printf("build_rmsnorm: bs: %d, hidden_size: %d\n", bs, hidden_size);
    std::vector<int64_t> x_shape = {bs, hidden_size};
    ge::TensorDesc x_desc(ge::Shape(x_shape), ge::FORMAT_ND, ge::DT_FLOAT16);
    x_desc.SetPlacement(ge::Placement::kPlacementDevice);
    auto x_data = ge::op::Data("x").set_attr_index(inputs.size());
    x_data.update_input_desc_x(x_desc);
    x_data.update_output_desc_y(x_desc);
    inputs.push_back(x_data);

    auto residual_desc = x_desc;
    residual_desc.SetShape(ge::Shape({bs, hidden_size}));
    auto residual_data = ge::op::Data("residual").set_attr_index(inputs.size());
    residual_data.update_input_desc_x(residual_desc);
    residual_data.update_output_desc_y(residual_desc);
    inputs.push_back(residual_data);

    auto outputs = add_rmsnorm({x_data, 0}, std::make_optional(std::make_tuple(residual_data, 0)), 1e-6, "<layer0>");

    graph.SetInputs(inputs);
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs_with_idx;
    for (int i = 0; i < outputs.size(); i++) {
      outputs_with_idx.push_back({outputs[i], {0}});
    }
    graph.SetOutputs(outputs_with_idx);
  }

  std::vector<ge::Operator> add_rmsnorm(std::tuple<ge::Operator, int64_t> x, std::optional<std::tuple<ge::Operator, int64_t>> residual, float eps, std::string name_suffix = "") {
    auto x_op = std::get<0>(x);
    auto x_op_idx = std::get<1>(x);
    auto x_desc = x_op.GetOutputDesc(x_op_idx);
    int dim = x_desc.GetShape().GetDim(x_desc.GetShape().GetDimNum() - 1);
    printf("add_rmsnorm_%s[dim: %d, eps: %f]: ", name_suffix.c_str(), dim, eps);
    print_shape(x_desc, "x");
    if (residual) {
      auto residual_op = std::get<0>(*residual);
      auto residual_op_idx = std::get<1>(*residual);
      auto residual_desc = residual_op.GetOutputDesc(residual_op_idx);
      print_shape(residual_desc, "res");
    }
    printf("\n");

    ge::TensorDesc weight_desc(ge::Shape({dim}), ge::FORMAT_ND, ge::DT_FLOAT16);
    weight_desc.SetPlacement(ge::Placement::kPlacementDevice);
    auto weight_name = std::string("rmsnorm_weight_") + name_suffix;
    auto weight_data = ge::op::Data(weight_name.c_str());
    weight_data.set_attr_index(inputs.size());
    weight_data.update_input_desc_x(weight_desc);
    weight_data.update_output_desc_y(weight_desc);
    inputs.push_back(weight_data);

    if (residual.has_value()) {
      auto add_name = std::string("rmsnorm_add_") + name_suffix;
      auto add = ge::op::Add(add_name.c_str());
      add.set_input_x1(x_op, x_op_idx);
      auto residual_op = std::get<0>(*residual);
      auto residual_op_idx = std::get<1>(*residual);
      add.set_input_x2(residual_op, residual_op_idx);
      add.update_output_desc_y(x_desc);

      auto rmsnorm_name = std::string("rmsnorm_") + name_suffix;
      auto rmsnorm = ge::op::RmsNorm(rmsnorm_name.c_str());
      rmsnorm.set_input_x(add);
      rmsnorm.set_input_gamma(weight_data);
      rmsnorm.set_attr_epsilon(eps);
      rmsnorm.update_output_desc_y(x_desc);

      graph.AddOp(add);
      graph.AddOp(rmsnorm);

      return {rmsnorm, add};
    } else {
      auto rmsnorm_name = std::string("rmsnorm_") + name_suffix;
      auto rmsnorm = ge::op::RmsNorm(rmsnorm_name.c_str());
      rmsnorm.set_input_x(x_op, x_op_idx);
      rmsnorm.set_input_gamma(weight_data);
      rmsnorm.set_attr_epsilon(eps);
      rmsnorm.update_output_desc_y(x_desc);

      graph.AddOp(rmsnorm);

      return {rmsnorm};
    }
  }

  ge::Operator add_swiglu(std::tuple<ge::Operator, int64_t> x, std::string name_suffix) {
    auto x_op = std::get<0>(x);
    auto x_op_idx = std::get<1>(x);
    auto x_desc = x_op.GetOutputDesc(x_op_idx);
    printf("add_swiglu_%s: ", name_suffix.c_str());
    print_shape(x_desc, "x", "\n");

    auto swiglu_name = std::string("swiglu_") + name_suffix;
    auto swiglu = ge::op::SwiGluEx(swiglu_name.c_str());
    swiglu.set_input_x(x_op, x_op_idx);
    swiglu.update_output_desc_y(x_desc);

    graph.AddOp(swiglu);

    return swiglu;
  }

  ge::Operator add_linear(std::tuple<ge::Operator, int64_t> x, int out_dim, std::string name_suffix) {
    auto x_op = std::get<0>(x);
    auto x_op_idx = std::get<1>(x);
    auto x_desc = x_op.GetOutputDesc(x_op_idx);
    int in_dim = x_desc.GetShape().GetDim(x_desc.GetShape().GetDimNum() - 1);
    printf("add_linear_%s[in_dim: %d, out_dim: %d]: ", name_suffix.c_str(), in_dim, out_dim);
    print_shape(x_desc, "x", "\n");

    ge::Shape y_shape = x_desc.GetShape();
    y_shape.SetDim(y_shape.GetDimNum() - 1, out_dim);
    ge::TensorDesc y_desc(y_shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    y_desc.SetPlacement(ge::Placement::kPlacementDevice);

    ge::TensorDesc weight_desc(ge::Shape({out_dim, in_dim}), ge::FORMAT_ND, ge::DT_FLOAT16);
    weight_desc.SetPlacement(ge::Placement::kPlacementDevice);
    auto weight_name = std::string("linear_weight_") + name_suffix;
    auto weight_data = ge::op::Data(weight_name.c_str());
    weight_data.set_attr_index(inputs.size());
    weight_data.update_input_desc_x(weight_desc);
    weight_data.update_output_desc_y(weight_desc);
    inputs.push_back(weight_data);

    auto linear_name = std::string("linear_") + name_suffix;
    auto linear = ge::op::MatMulV2(linear_name.c_str());
    linear.set_input_x1(x_op, x_op_idx);
    linear.set_input_x2(weight_data, 0);
    linear.set_attr_transpose_x2(true);
    linear.update_output_desc_y(y_desc);

    graph.AddOp(linear);

    return linear;
  }

  ge::Operator add_split(std::tuple<ge::Operator, int64_t> x, std::vector<int64_t> sizes, int64_t dim, std::string name_suffix) {
    auto x_op = std::get<0>(x);
    auto x_op_idx = std::get<1>(x);
    auto x_desc = x_op.GetOutputDesc(x_op_idx);
    printf("add_split_%s[dim: %d]: ", name_suffix.c_str(), dim);
    print_shape(x_desc, "x", "\n");

    std::vector<ge::TensorDesc> output_descs;
    int64_t real_dim = dim < 0 ? dim + x_desc.GetShape().GetDimNum() : dim;
    for (int i = 0; i < sizes.size(); i++) {
      auto output_shape = x_desc.GetShape().GetDims();
      output_shape[real_dim] = sizes[i];
      output_descs.push_back(ge::TensorDesc(ge::Shape(output_shape), ge::FORMAT_ND, ge::DT_FLOAT16));
    }

    ge::TensorDesc size_splits_desc(ge::Shape({(int64_t)sizes.size()}), ge::FORMAT_ND, ge::DT_INT64);
    size_splits_desc.SetPlacement(ge::Placement::kPlacementHost);
    auto size_splits_name = std::string("size_splits_const_") + name_suffix;
    auto size_splits_data_op = ge::op::Const(size_splits_name.c_str());
    size_splits_data_op.update_output_desc_y(size_splits_desc);
    ge::Tensor size_splits_tensor(size_splits_desc, reinterpret_cast<uint8_t*>(sizes.data()), sizes.size() * sizeof(int64_t));
    size_splits_data_op.set_attr_value(size_splits_tensor);

    std::vector<int64_t> split_dim_data = {dim};
    ge::TensorDesc split_dim_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
    split_dim_desc.SetPlacement(ge::Placement::kPlacementHost);
    auto split_dim_name = std::string("split_dim_const_") + name_suffix;
    auto split_dim_data_op = ge::op::Const(split_dim_name.c_str());
    split_dim_data_op.update_output_desc_y(split_dim_desc);
    ge::Tensor split_dim_tensor(split_dim_desc, reinterpret_cast<uint8_t*>(split_dim_data.data()), split_dim_data.size() * sizeof(int64_t));
    split_dim_data_op.set_attr_value(split_dim_tensor);

    auto split_name = std::string("split_") + name_suffix;
    auto split = ge::op::SplitV(split_name.c_str());
    split.set_input_x(x_op, x_op_idx);
    split.set_input_size_splits(size_splits_data_op);
    split.set_input_split_dim(split_dim_data_op);
    split.set_attr_num_split(sizes.size());
    split.create_dynamic_output_y(sizes.size());
    for (int i = 0; i < sizes.size(); i++) {
      split.UpdateDynamicOutputDesc("y", i, output_descs[i]);
    }

    graph.AddOp(size_splits_data_op);
    graph.AddOp(split_dim_data_op);
    graph.AddOp(split);

    return split;
  }

  ge::Operator add_reshape(std::tuple<ge::Operator, int64_t> x, std::vector<int64_t> shape, std::string name_suffix) {
    auto x_op = std::get<0>(x);
    auto x_op_idx = std::get<1>(x);
    auto x_desc = x_op.GetOutputDesc(x_op_idx);
    printf("add_reshape_%s: ", name_suffix.c_str());
    ge::TensorDesc y_desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT16);
    y_desc.SetPlacement(ge::Placement::kPlacementDevice);
    print_shape(x_desc, "x");
    print_shape(y_desc, "y", "\n");

    ge::TensorDesc shape_desc(ge::Shape({(int64_t)shape.size()}), ge::FORMAT_ND, ge::DT_INT64);
    // ge::TensorDesc shape_desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_INT64);
    shape_desc.SetPlacement(ge::Placement::kPlacementHost);
    auto shape_name = std::string("shape_const_") + name_suffix;
    auto shape_data_op = ge::op::Const(shape_name.c_str());
    shape_data_op.update_output_desc_y(shape_desc);
    std::vector<int64_t> shape_data = shape;
    ge::Tensor shape_tensor(shape_desc, reinterpret_cast<uint8_t*>(shape_data.data()), shape_data.size() * sizeof(int64_t));
    shape_data_op.set_attr_value(shape_tensor);

    auto reshape_name = std::string("reshape_") + name_suffix;
    auto reshape = ge::op::Reshape(reshape_name.c_str());
    reshape.set_input_x(x_op, x_op_idx);
    reshape.set_input_shape(shape_data_op);
    reshape.set_attr_axis(0);
    reshape.update_output_desc_y(y_desc);

    graph.AddOp(shape_data_op);
    graph.AddOp(reshape);

    return reshape;
  }

  void add_rotary_emb(std::tuple<ge::Operator, int64_t> q, std::tuple<ge::Operator, int64_t> k, std::tuple<ge::Operator, int64_t> pos, std::tuple<ge::Operator, int64_t> cos_sin_cache, std::string name_suffix) {
    auto q_op = std::get<0>(q);
    auto q_op_idx = std::get<1>(q);
    auto q_desc = q_op.GetOutputDesc(q_op_idx);
    auto k_op = std::get<0>(k);
    auto k_op_idx = std::get<1>(k);
    auto k_desc = k_op.GetOutputDesc(k_op_idx);
    auto pos_op = std::get<0>(pos);
    auto pos_op_idx = std::get<1>(pos);
    auto pos_desc = pos_op.GetOutputDesc(pos_op_idx);
    auto cos_sin_cache_op = std::get<0>(cos_sin_cache);
    auto cos_sin_cache_op_idx = std::get<1>(cos_sin_cache);
    auto cos_sin_cache_desc = cos_sin_cache_op.GetOutputDesc(cos_sin_cache_op_idx);

    printf("add_rotary_emb_%s: ", name_suffix.c_str());
    print_shape(q_desc, "q");
    print_shape(k_desc, "k");
    print_shape(pos_desc, "pos");
    print_shape(cos_sin_cache_desc, "cos_sin_cache", "\n");
  }

  void add_core_attn(std::tuple<ge::Operator, int64_t> q, std::tuple<ge::Operator, int64_t> k, std::tuple<ge::Operator, int64_t> v, std::string name_suffix) {
  }

  std::vector<ge::Operator> add_attn(std::tuple<ge::Operator, int64_t> x, int num_heads, int num_kv_heads, std::string name_suffix) {
    auto x_op = std::get<0>(x);
    auto x_op_idx = std::get<1>(x);
    auto x_desc = x_op.GetOutputDesc(x_op_idx);
    int hidden_size = x_desc.GetShape().GetDim(x_desc.GetShape().GetDimNum() - 1);
    int head_dim = hidden_size / num_heads;
    printf("attn_%s[heads: %d, kv_heads: %d, head_dim: %d]: ", name_suffix.c_str(), num_heads, num_kv_heads, head_dim);
    print_shape(x_desc, "x", "\n");

    int bs = x_desc.GetShape().GetDim(0);

    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;


    auto qkv_proj = add_linear({x_op, x_op_idx}, q_size + 2 * kv_size, name_suffix + "<qkv_proj>");
    auto split = add_split({qkv_proj, 0}, {q_size, kv_size, kv_size}, -1, name_suffix + "<split>");
    auto q_reshape = add_reshape({split, 0}, {bs, num_heads, head_dim}, name_suffix + "<q_reshape>");
    auto k_reshape = add_reshape({split, 1}, {bs, num_kv_heads, head_dim}, name_suffix + "<k_reshape>");

    auto q_norm = add_rmsnorm({q_reshape, 0}, std::nullopt, 1e-6, name_suffix + "<q_norm>")[0];
    auto k_norm = add_rmsnorm({k_reshape, 0}, std::nullopt, 1e-6, name_suffix + "<k_norm>")[0];

    auto q_reshape_back = add_reshape({q_norm, 0}, {bs, q_size}, name_suffix + "<q_reshape_back>");
    auto k_reshape_back = add_reshape({k_norm, 0}, {bs, kv_size}, name_suffix + "<k_reshape_back>");

    return {q_reshape_back, k_reshape_back};
  }

  ge::Operator add_mlp(std::tuple<ge::Operator, int64_t> x, int intermediate_size, std::string name_suffix) {
    auto x_op = std::get<0>(x);
    auto x_op_idx = std::get<1>(x);
    auto x_desc = x_op.GetOutputDesc(x_op_idx);
    int hidden_size = x_desc.GetShape().GetDim(x_desc.GetShape().GetDimNum() - 1);
    printf("mlp_%s[intermediate_size: %d]: ", name_suffix.c_str(), intermediate_size);
    print_shape(x_desc, "x", "\n");

    auto gate_up_proj = add_linear({x_op, x_op_idx}, intermediate_size * 2, name_suffix + "<gate_up_proj>");
    auto swiglu = add_swiglu({gate_up_proj, 0}, name_suffix + "<swiglu>");
    auto down_proj = add_linear({swiglu, 0}, hidden_size, name_suffix + "<down_proj>");

    return down_proj;
  }

};

void print_compiled_graph_summary(ge::Session& session, int id) {
  auto summary = session.GetCompiledGraphSummary(id);
  printf("is_static: %d\n", summary->IsStatic());
  size_t const_memory_size = 0;
  size_t feature_memory_size = 0;
  bool feature_memory_base_refreshable = false;
  size_t stream_num = 0;
  size_t event_num = 0;
  std::vector<ge::Shape> output_shapes;
  std::vector<ge::DataType> output_types;
  size_t refreshable_feature_memory_size = 0;
  size_t fixed_feature_memory_size = 0;

  summary->GetConstMemorySize(const_memory_size);
  summary->GetFeatureMemorySize(feature_memory_size);
  summary->GetFeatureMemoryBaseRefreshable(feature_memory_base_refreshable);
  summary->GetStreamNum(stream_num);
  summary->GetEventNum(event_num);
  summary->GetOutputShapes(output_shapes);
  summary->GetOutputDtypes(output_types);
  summary->GetRefreshableFeatureMemorySize(refreshable_feature_memory_size);
  summary->GetFixedFeatureMemorySize(fixed_feature_memory_size);

  printf("const_memory_size: %ld\n", const_memory_size);
  printf("feature_memory_size: %ld\n", feature_memory_size);
  printf("feature_memory_base_refreshable: %d\n", feature_memory_base_refreshable);
  printf("stream_num: %ld\n", stream_num);
  printf("event_num: %ld\n", event_num);
  printf("output_shapes: %ld\n", output_shapes.size());
  printf("output_types: %ld\n", output_types.size());
  for (int i = 0; i < output_shapes.size(); i++) {
    auto shape = output_shapes[i].GetDims();
    printf("shape: ");
    for (int j = 0; j < shape.size(); j++) {
      printf("%d ", shape[j]);
    }
    printf("\n");
  }
  for (int i = 0; i < output_types.size(); i++) {
    printf("dtype: %d\n", output_types[i]);
  }
  printf("refreshable_feature_memory_size: %ld\n", refreshable_feature_memory_size);
  printf("fixed_feature_memory_size: %ld\n", fixed_feature_memory_size);
}

ge::Graph& build_attn(ge::Graph& graph, int bs, int hidden_size, int num_heads, int num_kv_heads) {
  int head_dim = hidden_size / num_heads;
  int q_size = num_heads * head_dim;
  int kv_size = num_kv_heads * head_dim;
  printf("build_attn: bs: %d, hidden_size: %d, num_heads: %d, num_kv_heads: %d, head_dim: %d, q_size: %d, kv_size: %d\n", bs, hidden_size, num_heads, num_kv_heads, head_dim, q_size, kv_size);

  ge::TensorDesc x_desc(ge::Shape({bs, hidden_size}), ge::FORMAT_ND, ge::DT_FLOAT16);
  x_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto x_data = ge::op::Data("x").set_attr_index(0);
  x_data.update_input_desc_x(x_desc);
  x_data.update_output_desc_y(x_desc);

  ge::TensorDesc qkv_proj_desc(ge::Shape({(num_heads + 2 * num_kv_heads) * head_dim, hidden_size}), ge::FORMAT_ND, ge::DT_FLOAT16);
  qkv_proj_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto qkv_proj_data = ge::op::Data("qkv_proj").set_attr_index(1);
  qkv_proj_data.update_input_desc_x(qkv_proj_desc);
  qkv_proj_data.update_output_desc_y(qkv_proj_desc);

  auto qkv_proj = ge::op::MatMulV2("qkv_proj_mul");
  qkv_proj.set_input_x1(x_data);
  qkv_proj.set_input_x2(qkv_proj_data);
  qkv_proj.set_attr_transpose_x2(true);

  auto split = ge::op::SplitV("split");
  split.set_input_x(qkv_proj);

  std::vector<int64_t> size_splits_data = {q_size, kv_size, kv_size};
  ge::TensorDesc size_splits_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
  size_splits_desc.SetPlacement(ge::Placement::kPlacementHost);
  auto size_splits_data_op = ge::op::Const("size_splits_const");
  size_splits_data_op.update_output_desc_y(size_splits_desc);
  ge::Tensor size_splits_tensor(size_splits_desc, reinterpret_cast<uint8_t*>(size_splits_data.data()), size_splits_data.size() * sizeof(int64_t));
  size_splits_data_op.set_attr_value(size_splits_tensor);

  std::vector<int64_t> split_dim_data = {-1};
  ge::TensorDesc split_dim_desc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT64);
  split_dim_desc.SetPlacement(ge::Placement::kPlacementHost);
  auto split_dim_data_op = ge::op::Const("split_dim_const");
  split_dim_data_op.update_output_desc_y(split_dim_desc);
  ge::Tensor split_dim_tensor(split_dim_desc, reinterpret_cast<uint8_t*>(split_dim_data.data()), split_dim_data.size() * sizeof(int64_t));
  split_dim_data_op.set_attr_value(split_dim_tensor);

  split.set_input_size_splits(size_splits_data_op);
  split.set_input_split_dim(split_dim_data_op);
  split.set_attr_num_split(3);
  split.create_dynamic_output_y(3);

  graph.AddOp(qkv_proj);
  graph.AddOp(size_splits_data_op);
  graph.AddOp(split_dim_data_op);
  graph.AddOp(split);
  graph.SetInputs({x_data, qkv_proj_data});
  graph.SetOutputs({split});

  return graph;
}

ge::Graph& build_mlp(ge::Graph& graph, int bs, int hidden_size, int intermediate_size) {
  printf("build_mlp: bs: %d, hidden_size: %d, intermediate_size: %d\n", bs, hidden_size, intermediate_size);

  ge::TensorDesc x_desc(ge::Shape({bs, hidden_size}), ge::FORMAT_ND, ge::DT_FLOAT16);
  x_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto x_data = ge::op::Data("x").set_attr_index(0);
  x_data.update_input_desc_x(x_desc);
  x_data.update_output_desc_y(x_desc);

  ge::TensorDesc gate_up_proj_desc(ge::Shape({intermediate_size * 2, hidden_size}), ge::FORMAT_ND, ge::DT_FLOAT16);
  gate_up_proj_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto gate_up_proj_data = ge::op::Data("gate_up_proj").set_attr_index(1);
  gate_up_proj_data.update_input_desc_x(gate_up_proj_desc);
  gate_up_proj_data.update_output_desc_y(gate_up_proj_desc);

  ge::TensorDesc down_proj_desc(ge::Shape({hidden_size, intermediate_size}), ge::FORMAT_ND, ge::DT_FLOAT16);
  down_proj_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto down_proj_data = ge::op::Data("down_proj").set_attr_index(2);
  down_proj_data.update_input_desc_x(down_proj_desc);
  down_proj_data.update_output_desc_y(down_proj_desc);

  auto gate_up = ge::op::MatMulV2("gate_up_proj_mul");
  gate_up.set_input_x1(x_data);
  gate_up.set_input_x2(gate_up_proj_data);
  gate_up.set_attr_transpose_x2(true);

  auto swiglu = ge::op::SwiGluEx("swiglu");
  swiglu.set_input_x(gate_up);

  auto down = ge::op::MatMulV2("down_proj_mul");
  down.set_input_x1(swiglu);
  down.set_input_x2(down_proj_data);
  down.set_attr_transpose_x2(true);

  graph.AddOp(gate_up);
  graph.AddOp(swiglu);
  graph.AddOp(down);
  graph.SetInputs({x_data, gate_up_proj_data, down_proj_data});
  graph.SetOutputs({down});

  return graph;
}

void init_ffi_graph(py::module_ &&m) {
  py::class_<ge::Graph>(m, "Graph")
    .def(py::init<const std::string&>())
    .def("build_rmsnorm", [](ge::Graph& self, int bs, int hidden_size) {
      GraphBuilder builder(self);
      builder.build_rmsnorm(bs, hidden_size);
    })
    .def("build_attn", [](ge::Graph& self, int bs, int hidden_size, int num_heads, int num_kv_heads) {
      GraphBuilder builder(self);
      builder.build_attn(bs, hidden_size, num_heads, num_kv_heads);
    })
    .def("build_mlp", &build_mlp);

  py::class_<ge::Session>(m, "Session")
    .def(py::init([]() {
      std::map<ge::AscendString, ge::AscendString> options;
      return std::make_unique<ge::Session>(options);
    }))
    .def("add_graph", [](ge::Session &self, int id, ge::Graph &graph) {
      pybind11::gil_scoped_release gil_release;
      if (self.AddGraph(id, graph) != 0) {
        throw std::runtime_error("Failed to add graph to session");
      }
    })
    .def("compile_graph", [](ge::Session &self, int id) {
      pybind11::gil_scoped_release gil_release;
      if (self.CompileGraph(id) != 0) {
        throw std::runtime_error("Failed to compile graph");
      }
    })
    .def("dump_summary", [](ge::Session &self, int id) {
      print_compiled_graph_summary(self, id);
    })
    .def("run_async", [](ge::Session &self, int id, std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs) {
      pybind11::gil_scoped_release gil_release;
      aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
      std::vector<ge::Tensor> ge_inputs;
      std::vector<ge::Tensor> ge_outputs;
      std::map<at::ScalarType, ge::DataType> dtype_map = {
        {torch::kFloat16, ge::DT_FLOAT16},
        {torch::kFloat, ge::DT_FLOAT}
      };

      for (auto &input : inputs) {
        ge::TensorDesc desc(ge::Shape(input.sizes().vec()), ge::FORMAT_ND, dtype_map[input.scalar_type()]);
        // print_shape(desc, "input", "\n");
        desc.SetPlacement(ge::Placement::kPlacementDevice);
        ge::Tensor tensor(desc);
        auto ptr = reinterpret_cast<uint8_t*>(input.data_ptr());
        tensor.SetData(ptr, input.numel() * input.element_size(), [](uint8_t*) {});
        ge_inputs.push_back(tensor);
      }
      for (auto &output : outputs) {
        ge::TensorDesc desc(ge::Shape(output.sizes().vec()), ge::FORMAT_ND, dtype_map[output.scalar_type()]);
        // print_shape(desc, "output", "\n");
        desc.SetPlacement(ge::Placement::kPlacementDevice);
        ge::Tensor tensor(desc);
        auto ptr = reinterpret_cast<uint8_t*>(output.data_ptr());
        tensor.SetData(ptr, output.numel() * output.element_size(), [](uint8_t*) {});
        ge_outputs.push_back(tensor);
      }

      if (self.RunGraphWithStreamAsync(id, stream, ge_inputs, ge_outputs) != 0) {
        throw std::runtime_error("Failed to run graph");
      }
    });
}

}
