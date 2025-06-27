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
#include <vector>

#include "kernels/type.h"

namespace native {

extern void swiglu_impl(ScalarType dtype, void *stream, uint8_t *input, uint8_t *output, int dim, int64_t stride, int64_t out_stride, int64_t num_tokens, uint32_t aiv_num);


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
  TORCH_CHECK(x.size(-1) >= 256 && x.size(-1) % 256 == 0,
              "swiglu: last dimension must be a multiple of 256, got ", x.size(-1));

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

  printf("num_tokens: %d, dim: %d, num_exports: %d, inner_dim: %d\n",
         num_tokens, dim, num_exports, inner_dim);

  at::Tensor y = at::empty({num_tokens, inner_dim}, x.options());
  return y;
}

} // namespace native

TORCH_LIBRARY(ascend910a, m) {
  m.def("swiglu(Tensor x) -> Tensor");
  m.impl("swiglu", torch::kPrivateUse1, &native::swiglu);

  m.def("grouped_matmul(Tensor x, Tensor w, Tensor group_list) -> Tensor");
  m.impl("grouped_matmul", torch::kPrivateUse1, &native::grouped_matmul);
}

PYBIND11_MODULE(ascend910a_extras_C, m) {

}
