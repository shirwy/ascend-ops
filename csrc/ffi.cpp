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

namespace native {

extern void swiglu_impl(void *stream, uint8_t *input, uint8_t *output, int dim, int64_t stride, int64_t out_stride, int64_t num_tokens, uint32_t aiv_num);


at::Tensor swiglu(const at::Tensor& x) {
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
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("swiglu");
  cmd.SetCustomHandler([scalar_type, stream, x_ptr, y_ptr, num_tokens, dim, stride, out_stride]() -> int {
      // auto dtype_num = get_dtype_from_torch(scalar_type);
      fe::PlatFormInfos platform_infos;
      int device_id = 0;
      fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
      uint32_t aiv_num = platform_infos.GetCoreNumByType("aiv");
      swiglu_impl(stream, x_ptr, y_ptr, dim, stride, out_stride, num_tokens, aiv_num);
      return 0;
  });
  cmd.Run();
  return y;
}

} // namespace native

TORCH_LIBRARY(ascend910a, m) {
  m.def("swiglu(Tensor x) -> Tensor");
  m.impl("swiglu", torch::kPrivateUse1, &native::swiglu);
}

PYBIND11_MODULE(ascend910a_extras_C, m) {

}
