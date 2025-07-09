#include <vector>
#include <map>
// FIXME: must be included first, or char_t wont be found. Don't know why.
#include "graph/types.h"
#include "ge/ge_api.h"
#include "all_ops.h"
#include "op_proto.h"

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


#include "kernels/type.h"

namespace native {

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


extern void swiglu_impl(ScalarType dtype, void *stream, uint8_t *input, uint8_t *output, int dim, int64_t stride, int64_t out_stride, int64_t num_tokens, uint32_t aiv_num);
extern void grouped_matmul_impl(ScalarType dtype, void *stream, uint8_t *x, uint8_t *w, uint8_t *group_list, uint8_t *y,
                                int num_tokens, int dim, int num_exports, int inner_dim, uint32_t aic_num);
extern void matmul_impl(void* stream, uint8_t* x, uint8_t* w, uint8_t* y,
                 int m, int n, int k);


// extern at::Tensor mlp(at::Tensor x, at::Tensor gate_up_proj, at::Tensor down_proj);

at::Tensor graph_swiglu(at::Tensor x) {
  pybind11::gil_scoped_release gil_release;
  auto x_sizes = x.sizes();
  std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
  y_sizes.back() = y_sizes.back() / 2;
  at::Tensor y = at::empty(y_sizes, x.options());
  uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x.data_ptr());
  uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y.data_ptr());


  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  ge::Graph graph("swiglu_graph");
  ge::TensorDesc x_desc(ge::Shape(x_sizes.vec()), ge::FORMAT_ND, ge::DT_FLOAT16);
  x_desc.SetPlacement(ge::Placement::kPlacementDevice);
  auto x_data = ge::op::Data("x").set_attr_index(0);
  x_data.update_input_desc_x(x_desc);
  x_data.update_output_desc_y(x_desc);

  auto swiglu = ge::op::Swiglu("swiglu");
  swiglu.set_input_x(x_data);

  graph.AddOp(swiglu);
  graph.SetInputs({x_data});
  graph.SetOutputs({swiglu});

  ge::Tensor x_tensor(x_desc);
  x_tensor.SetData(x_ptr, x.numel() * x.element_size(), [](uint8_t*) {});
  ge::TensorDesc y_desc(ge::Shape(y_sizes), ge::FORMAT_ND, ge::DT_FLOAT16);
  y_desc.SetPlacement(ge::Placement::kPlacementDevice);
  ge::Tensor y_tensor(y_desc);
  y_tensor.SetData(y_ptr, y.numel() * y.element_size(), [](uint8_t*) {});

  std::vector<ge::Tensor> inputs = {x_tensor};
  std::vector<ge::Tensor> outputs = {y_tensor};

  std::map<ge::AscendString, ge::AscendString> options;
  ge::Session session(options);
  if (session.AddGraph(0, graph) != 0) {
    throw std::runtime_error("Failed to add graph to session");
  }
  if (session.CompileGraph(0) != 0) {
    throw std::runtime_error("Failed to compile graph");
  }
  if (session.RunGraphWithStreamAsync(0, stream, inputs, outputs) != 0) {
    throw std::runtime_error("Failed to run graph");
  }
  aclrtSynchronizeStream(stream);

  return y;
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

  auto gate_up = ge::op::MatMulV2("gate_up_proj_mul");
  gate_up.set_input_x1(x_data);
  gate_up.set_input_x2(gate_up_proj_data);
  gate_up.set_attr_transpose_x2(true);
  auto down = ge::op::MatMulV2("down_proj_mul");
  down.set_input_x1(gate_up);
  down.set_input_x2(down_proj_data);
  down.set_attr_transpose_x2(true);

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
  aclrtSynchronizeStream(stream);

  // if (ge::GEFinalize() != 0) {
  //   throw std::runtime_error("Failed to finalize GE");
  // }

  return y;
}

// extern at::Tensor graph_run(const at::Tensor& x, const at::Tensor& w);
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
  // if (ge::GEInitialize(config) != 0) {
  //   throw std::runtime_error("Failed to initialize GE");
  // }
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
  printf("RunGraphWithStreamAsync done\n");
  aclrtSynchronizeStream(stream);
  printf("SynchronizeStream done\n");

  // if (ge::GEFinalize() != 0) {
  //   throw std::runtime_error("Failed to finalize GE");
  // }

  return y;
}

ScalarType get_dtype_from_torch(at::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat16:
      return ScalarType::FP16;
    case torch::kBFloat16:
      return ScalarType::BF16;
    case torch::kFloat:
      return ScalarType::FP32;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}


at::Tensor swiglu(const at::Tensor& x) {
  TORCH_CHECK(x.dim() == 2,
              "swiglu: input tensor must be 2D, got ", x.dim(), "D tensor");
  TORCH_CHECK(x.size(-1) >= 64 && x.size(-1) % 64 == 0,
              "swiglu: last dimension must be a multiple of 64, got ", x.size(-1));
  TORCH_CHECK(x.is_contiguous(),
              "swiglu: input tensor must be contiguous");

  at::ScalarType scalar_type = x.scalar_type();
  auto x_sizes = x.sizes();
  std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
  y_sizes.back() = y_sizes.back() / 2;
  at::Tensor y = at::empty(y_sizes, x.options());

  int64_t num_tokens = x.numel() / x.size(-1);
  uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x.data_ptr());
  uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y.data_ptr());
  int dim = x.size(-1) / 2;
  int64_t stride = x.stride(-2);
  int64_t out_stride = y.stride(-2);
  int device_id = c10_npu::getCurrentNPUStream().device_index();
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("swiglu");
  cmd.SetCustomHandler([scalar_type, device_id, stream, x_ptr, y_ptr, num_tokens, dim, stride, out_stride]() -> int {
      auto dtype = get_dtype_from_torch(scalar_type);
      fe::PlatFormInfos platform_infos;
      fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
      uint32_t aiv_num = platform_infos.GetCoreNumByType("aiv");
      swiglu_impl(dtype, stream, x_ptr, y_ptr, dim, stride, out_stride, num_tokens, aiv_num);
      return 0;
  });
  cmd.Run();
  return y;
}

at::Tensor grouped_matmul(const at::Tensor& x, const at::Tensor& w, const at::Tensor& group_list) {
  TORCH_CHECK(x.dim() == 2 && w.dim() == 3 && group_list.dim() == 1,
              "grouped_matmul: input tensors must be 2D and group_list must be 1D");
  at::ScalarType scalar_type = x.scalar_type();

  int num_tokens = x.size(0);
  int dim = x.size(1);
  int num_exports = w.size(0);
  int inner_dim = w.size(2);

  TORCH_CHECK(dim == w.size(1),
              "grouped_matmul: last dimension of x must match second dimension of w, got ", dim, " and ", w.size(1));
  TORCH_CHECK(x.is_contiguous() && group_list.is_contiguous(),
              "grouped_matmul: x and group_list must be contiguous tensors");
  TORCH_CHECK(w.stride(1) == 1 && w.stride(2) == dim,
              "grouped_matmul: w must be K-major order, got strides ", w.strides());
  TORCH_CHECK(w.size(1) % 64 == 0 && w.size(2) % 64 == 0,
              "grouped_matmul: second and third dimensions of w must be multiples of 64, got ", w.size(1), " and ", w.size(2));

  uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x.data_ptr());
  uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr());
  uint8_t* group_list_ptr = reinterpret_cast<uint8_t*>(group_list.data_ptr());
  at::Tensor y = at::empty({num_tokens, inner_dim}, x.options());
  uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y.data_ptr());

  int device_id = c10_npu::getCurrentNPUStream().device_index();
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("grouped_matmul");
  cmd.SetCustomHandler([scalar_type, device_id, stream, x_ptr, w_ptr, group_list_ptr, y_ptr, num_tokens, dim, num_exports, inner_dim]() -> int {
      auto dtype = get_dtype_from_torch(scalar_type);
      fe::PlatFormInfos platform_infos;
      fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
      uint32_t aic_num = platform_infos.GetCoreNumByType("aic");
      grouped_matmul_impl(dtype, stream, x_ptr, w_ptr, group_list_ptr, y_ptr,
                          num_tokens, dim, num_exports, inner_dim, aic_num);
      return 0;
  });
  cmd.Run();
  return y;
}

at::Tensor matmul(const at::Tensor& x, const at::Tensor& w) {
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2,
              "matmul: input tensors must be 2D");
  TORCH_CHECK(x.size(1) == w.size(1),
              "matmul: last dimension of x must match last dimension of w, got ", x.size(1), " and ", w.size(1));
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "matmul: x and w must be contiguous tensors");

  at::ScalarType scalar_type = x.scalar_type();
  TORCH_CHECK(scalar_type == torch::kFloat16,
              "matmul: only float16 is supported, got ", scalar_type);
  int m = x.size(0);
  int k = x.size(1);
  int n = w.size(0);

  // TORCH_CHECK(k % 64 == 0 && n % 64 == 0,
  //             "matmul: k and n must be multiples of 64, got k=", k, " and n=", n);

  at::Tensor y = at::empty({m, n}, x.options());
  uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x.data_ptr());
  uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr());
  uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y.data_ptr());

  int device_id = c10_npu::getCurrentNPUStream().device_index();
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("matmul");
  cmd.SetCustomHandler([scalar_type, device_id, stream, x_ptr, w_ptr, y_ptr, m, n, k]() -> int {
      matmul_impl(stream, x_ptr, w_ptr, y_ptr, m, n, k);
      return 0;
  });
  cmd.Run();

  return y;
}

int print_info(int device_id) {
  fe::PlatFormInfos platform_infos;
  fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
  uint32_t aic_num = platform_infos.GetCoreNumByType("aic");
  uint32_t aiv_num = platform_infos.GetCoreNumByType("aiv");
  uint64_t l0_a_size, l0_b_size, l0_c_size, l1_size, l2_size, ub_size, hbm_size;
  platform_infos.GetLocalMemSize(fe::LocalMemType::L0_A, l0_a_size);
  platform_infos.GetLocalMemSize(fe::LocalMemType::L0_B, l0_b_size);
  platform_infos.GetLocalMemSize(fe::LocalMemType::L0_C, l0_c_size);
  platform_infos.GetLocalMemSize(fe::LocalMemType::L1, l1_size);
  platform_infos.GetLocalMemSize(fe::LocalMemType::L2, l2_size);
  platform_infos.GetLocalMemSize(fe::LocalMemType::UB, ub_size);
  platform_infos.GetLocalMemSize(fe::LocalMemType::HBM, hbm_size);
  uint64_t l0_a_bw, l0_b_bw, l0_c_bw, l1_bw, l2_bw, ub_bw, hbm_bw;
  platform_infos.GetLocalMemBw(fe::LocalMemType::L0_A, l0_a_bw);
  platform_infos.GetLocalMemBw(fe::LocalMemType::L0_B, l0_b_bw);
  platform_infos.GetLocalMemBw(fe::LocalMemType::L0_C, l0_c_bw);
  platform_infos.GetLocalMemBw(fe::LocalMemType::L1, l1_bw);
  platform_infos.GetLocalMemBw(fe::LocalMemType::L2, l2_bw);
  platform_infos.GetLocalMemBw(fe::LocalMemType::UB, ub_bw);
  platform_infos.GetLocalMemBw(fe::LocalMemType::HBM, hbm_bw);
  printf("L0_A: %ld, L0_B: %ld, L0_C: %ld, L1: %ld, L2: %ld, UB: %ld, HBM: %ld\n", l0_a_size, l0_b_size, l0_c_size, l1_size, l2_size, ub_size, hbm_size);
  printf("L0_A_BW: %ld, L0_B_BW: %ld, L0_C_BW: %ld, L1_BW: %ld, L2_BW: %ld, UB_BW: %ld, HBM_BW: %ld\n", l0_a_bw, l0_b_bw, l0_c_bw, l1_bw, l2_bw, ub_bw, hbm_bw);
  printf("AIC: %d, AIV: %d\n", aic_num, aiv_num);
  return 0;
}

} // namespace native

TORCH_LIBRARY(ascend910a, m) {
  m.def("swiglu(Tensor x) -> Tensor");
  m.impl("swiglu", torch::kPrivateUse1, &native::swiglu);

  m.def("grouped_matmul(Tensor x, Tensor w, Tensor group_list) -> Tensor");
  m.impl("grouped_matmul", torch::kPrivateUse1, &native::grouped_matmul);

  m.def("matmul(Tensor x, Tensor w) -> Tensor");
  m.impl("matmul", torch::kPrivateUse1, &native::matmul);
}

PYBIND11_MODULE(ascend910a_extras_C, m) {
  m.def("print_info", &native::print_info, "Print info about the device");
  m.def("graph_run", &native::graph_run, "Run graph");
  m.def("mlp", &native::mlp, "Qwen3 MLP");
  m.def("graph_swiglu", &native::graph_swiglu, "Graph swiglu");
}
