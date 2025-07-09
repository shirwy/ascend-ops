#include <vector>
#include <map>
// FIXME: must be included first, or char_t wont be found. Don't know why.
#include "graph/types.h"
#include "ge/ge_api.h"
#include "all_ops.h"

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include <pybind11/pybind11.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclnn/opdev/platform.h"

namespace native {

/*
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

at::Tensor mlp(at::Tensor x, at::Tensor gate_up_proj, at::Tensor down_proj) {
  pybind11::gil_scoped_release gil_release;
  int bs = x.size(0);
  int hidden_size = x.size(1);
  int intermediate_size = down_proj.size(1);
  printf("bs: %d, hidden_size: %d, intermediate_size: %d\n", bs, hidden_size, intermediate_size);
  at::Tensor y = at::empty_like(x);

  printf("y bufsize: %ld\n", y.numel() * y.element_size());
  printf("x bufsize: %ld\n", x.numel() * x.element_size());
  printf("gate_up_proj bufsize: %ld\n", gate_up_proj.numel() * gate_up_proj.element_size());
  printf("down_proj bufsize: %ld\n", down_proj.numel() * down_proj.element_size());

  uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x.data_ptr());
  uint8_t* gate_up_proj_ptr = reinterpret_cast<uint8_t*>(gate_up_proj.data_ptr());
  uint8_t* down_proj_ptr = reinterpret_cast<uint8_t*>(down_proj.data_ptr());
  uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y.data_ptr());

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  // std::map<ge::AscendString, ge::AscendString> config = {
  //   {"ge.exec.deviceId", "0"},
  //   {"ge.graphRunMode", "0"},
  //   {"ge::exec.precision_mode", "allow_fp32_to_fp16"}
  // };


  // // build graph
  // ge::GEInitialize(config);
  ge::Graph graph("mlp_graph");

  ge::TensorDesc x_desc(ge::Shape(x.sizes().vec()), ge::FORMAT_ND, ge::DT_FLOAT16);
  x_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto x_data = ge::op::Data("x").set_attr_index(0);
  x_data.update_input_desc_x(x_desc);
  x_data.update_output_desc_y(x_desc);
  ge::TensorDesc gate_up_proj_desc(ge::Shape(gate_up_proj.sizes().vec()), ge::FORMAT_ND, ge::DT_FLOAT16);
  gate_up_proj_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto gate_up_proj_data = ge::op::Data("gate_up_proj").set_attr_index(1);
  gate_up_proj_data.update_input_desc_x(gate_up_proj_desc);
  gate_up_proj_data.update_output_desc_y(gate_up_proj_desc);
  ge::TensorDesc down_proj_desc(ge::Shape(down_proj.sizes().vec()), ge::FORMAT_ND, ge::DT_FLOAT16);
  down_proj_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto down_proj_data = ge::op::Data("down_proj").set_attr_index(2);
  down_proj_data.update_input_desc_x(down_proj_desc);
  down_proj_data.update_output_desc_y(down_proj_desc);

  auto gate_up = ge::op::MatMulV2("gate_up_proj");
  gate_up.set_input_x1(x_data);
  gate_up.set_input_x2(gate_up_proj_data);
  auto down = ge::op::MatMulV2("down_proj");
  down.set_input_x1(gate_up);
  down.set_input_x2(down_proj_data);

  graph.AddOp(gate_up);
  graph.AddOp(down);
  graph.SetInputs({x_data, gate_up_proj_data, down_proj_data});
  graph.SetOutputs({down});

  // prepare input/output
  ge::Tensor x_tensor(x_desc);
  x_tensor.SetData(x_ptr, x.numel() * x.element_size(), [](uint8_t*) {});
  ge::Tensor gate_up_proj_tensor(gate_up_proj_desc);
  gate_up_proj_tensor.SetData(gate_up_proj_ptr, gate_up_proj.numel() * gate_up_proj.element_size(), [](uint8_t*) {});
  ge::Tensor down_proj_tensor(down_proj_desc);
  down_proj_tensor.SetData(down_proj_ptr, down_proj.numel() * down_proj.element_size(), [](uint8_t*) {});
  ge::TensorDesc y_desc(ge::Shape(y.sizes().vec()), ge::FORMAT_ND, ge::DT_FLOAT16);
  y_desc.SetPlacement(ge::Placement::kPlacementDevice);
  ge::Tensor y_tensor(y_desc);
  y_tensor.SetData(y_ptr, y.numel() * y.element_size(), [](uint8_t*) {});

  std::vector<ge::Tensor> inputs = {x_tensor, gate_up_proj_tensor, down_proj_tensor};
  std::vector<ge::Tensor> outputs = {y_tensor};

  // run graph
  std::map<ge::AscendString, ge::AscendString> options;
  ge::Session session(options);
  if (session.AddGraph(0, graph) != 0) {
    throw std::runtime_error("Failed to add graph to session");
  }
  if (session.CompileGraph(0) != 0) {
    throw std::runtime_error("Failed to compile graph");
  }

  print_compiled_graph_summary(session, 0);

  if (session.RunGraphWithStreamAsync(0, stream, inputs, outputs) != 0) {
    throw std::runtime_error("Failed to run graph");
  }
  // if (ge::GEFinalize() != 0) {
  //   throw std::runtime_error("Failed to finalize GE");
  // }

  return y;
}
*/

/*
at::Tensor graph_run(const at::Tensor& x, const at::Tensor& w) {
  pybind11::gil_scoped_release gil_release;
  int m = x.size(0);
  int k = x.size(1);
  int n = w.size(1);
  at::Tensor y = at::empty({m, n}, x.options());
  uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x.data_ptr());
  uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr());
  uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y.data_ptr());
  // void* x_ptr = x.data_ptr();
  // void* w_ptr = w.data_ptr();
  // void* y_ptr = y.data_ptr();


  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  std::map<ge::AscendString, ge::AscendString> config = {
    {"ge.exec.deviceId", "0"},
    {"ge.graphRunMode", "0"},
    {"ge::exec.precision_mode", "allow_fp32_to_fp16"}
  };
  ge::GEInitialize(config);
  ge::Graph graph("matmul_graph");


  ge::TensorDesc x_desc(ge::Shape(x.sizes().vec()), ge::FORMAT_ND, ge::DT_FLOAT16);
  x_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto x_data = ge::op::Data("x").set_attr_index(0);
  x_data.update_input_desc_x(x_desc);
  x_data.update_output_desc_y(x_desc);

  ge::TensorDesc w_desc(ge::Shape(w.sizes().vec()), ge::FORMAT_ND, ge::DT_FLOAT16);
  w_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto w_shape = w.sizes().vec();
  auto w_data = ge::op::Data("w").set_attr_index(1);
  w_data.update_input_desc_x(w_desc);
  w_data.update_output_desc_y(w_desc);
  printf("w_data done\n");

  auto matmul = ge::op::MatMulV2("matmul_v2");
  matmul.set_input_x1(x_data);
  matmul.set_input_x2(w_data);
  printf("matmul done\n");

  graph.AddOp(matmul);
  graph.SetInputs({x_data, w_data});
  graph.SetOutputs({matmul});


  ge::Tensor x_tensor(x_desc);
  x_tensor.SetData(x_ptr, x.numel() * x.element_size(), [](uint8_t*) {});
  ge::Tensor w_tensor(w_desc);
  w_tensor.SetData(w_ptr, w.numel() * w.element_size(), [](uint8_t*) {});
  ge::TensorDesc y_desc(ge::Shape(y.sizes().vec()), ge::FORMAT_ND, ge::DT_FLOAT16);
  y_desc.SetPlacement(ge::Placement::kPlacementDevice);
  ge::Tensor y_tensor(y_desc);
  y_tensor.SetData(y_ptr, y.numel() * y.element_size(), [](uint8_t*) {});


  std::vector<ge::Tensor> inputs = {x_tensor, w_tensor};
  std::vector<ge::Tensor> outputs = {y_tensor};

  std::map<ge::AscendString, ge::AscendString> options;
  ge::Session session(options);
  if (session.AddGraph(0, graph) != 0) {
    throw std::runtime_error("Failed to add graph to session");
  }
  printf("AddGraph done\n");

  // if (session.RunGraph(0, inputs, outputs) != 0) {
  if (session.RunGraphWithStreamAsync(0, stream, inputs, outputs) != 0) {
    throw std::runtime_error("Failed to run graph");
  }
  // printf("RunGraphWithStreamAsync done\n");
  // aclrtSynchronizeStream(stream);
  // printf("SynchronizeStream done\n");

  if (ge::GEFinalize() != 0) {
    throw std::runtime_error("Failed to finalize GE");
  }

  return y;
}
*/

}
