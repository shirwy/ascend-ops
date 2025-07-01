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
extern void grouped_matmul_impl(ScalarType dtype, void *stream, uint8_t *x, uint8_t *w, uint8_t *group_list, uint8_t *y,
                                int num_tokens, int dim, int num_exports, int inner_dim, uint32_t aic_num);
extern void matmul_impl(void* stream, uint8_t* x, uint8_t* w, uint8_t* y,
                 int m, int k, int n);

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
  TORCH_CHECK(x.size(1) == w.size(0),
              "matmul: last dimension of x must match first dimension of w, got ", x.size(1), " and ", w.size(0));
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
              "matmul: x and w must be contiguous tensors");

  at::ScalarType scalar_type = x.scalar_type();
  TORCH_CHECK(scalar_type == torch::kFloat16,
              "matmul: only float16 is supported, got ", scalar_type);
  int m = x.size(0);
  int k = x.size(1);
  int n = w.size(0);

  TORCH_CHECK(k % 128 == 0 && n % 128 == 0,
              "matmul: k and n must be multiples of 128, got k=", k, " and n=", n);

  at::Tensor y = at::empty_like(x);
  uint8_t* x_ptr = reinterpret_cast<uint8_t*>(x.data_ptr());
  uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr());
  uint8_t* y_ptr = reinterpret_cast<uint8_t*>(y.data_ptr());

  int device_id = c10_npu::getCurrentNPUStream().device_index();
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("matmul");
  cmd.SetCustomHandler([scalar_type, device_id, stream, x_ptr, w_ptr, y_ptr, m, k, n]() -> int {
      matmul_impl(stream, x_ptr, w_ptr, y_ptr, m, k, n);
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
}
