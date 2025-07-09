#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "aclnn_swi_glu_ex.h"


namespace native {

at::Tensor swiglu(at::Tensor x) {
  TORCH_CHECK(x.dim() == 2,
              "swiglu: input tensor must be 2D, got ", x.dim(), "D tensor");
  TORCH_CHECK(x.size(-1) >= 64 && x.size(-1) % 64 == 0,
              "swiglu: last dimension must be a multiple of 64, got ", x.size(-1));
  TORCH_CHECK(x.is_contiguous(),
              "swiglu: input tensor must be contiguous");

  at::ScalarType scalar_type = x.scalar_type();
  auto x_sizes = x.sizes();
  auto x_strides = x.strides();
  std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
  y_sizes.back() = y_sizes.back() / 2;
  at::Tensor y = at::empty(y_sizes, x.options());
  auto y_strides = y.strides();

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  aclTensor *x_acl = aclCreateTensor(x_sizes.data(), x.dim(), ACL_FLOAT16, x_strides.data(), 0, ACL_FORMAT_ND, x_sizes.data(), x.dim(), x.data_ptr());
  if (x_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor");
  }
  aclTensor *y_acl = aclCreateTensor(y_sizes.data(), y.dim(), ACL_FLOAT16, y_strides.data(), 0, ACL_FORMAT_ND, y_sizes.data(), y.dim(), y.data_ptr());
  if (y_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor");
  }

  uint64_t workspace_size = 0;
  aclOpExecutor* handle = nullptr;
  if (aclnnSwiGluExGetWorkspaceSize(x_acl, y_acl, &workspace_size, &handle) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to get workspace size");
  }
  auto options = at::TensorOptions().dtype(torch::kUInt8).device(x.device());
  auto workspace_tensor = at::empty({(int64_t)workspace_size}, options);
  if (aclnnSwiGluEx(workspace_tensor.data_ptr(), workspace_size, handle, stream) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to execute swiglu");
  }

  if (aclDestroyTensor(x_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for x");
  }
  if (aclDestroyTensor(y_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for y");
  }
  return y;
}

}
