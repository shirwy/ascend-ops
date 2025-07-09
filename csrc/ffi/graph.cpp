// FIXME: must be included first.
#include "ge/ge_api.h"
#include "all_ops.h"
#include "op_proto.h"


#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace py = pybind11;

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
        desc.SetPlacement(ge::Placement::kPlacementDevice);
        ge::Tensor tensor(desc);
        auto ptr = reinterpret_cast<uint8_t*>(input.data_ptr());
        tensor.SetData(ptr, input.numel() * input.element_size(), [](uint8_t*) {});
        ge_inputs.push_back(tensor);
      }
      for (auto &output : outputs) {
        ge::TensorDesc desc(ge::Shape(output.sizes().vec()), ge::FORMAT_ND, dtype_map[output.scalar_type()]);
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
