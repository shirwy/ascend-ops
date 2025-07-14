#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "aclnn_swi_glu_ex.h"
#include "aclnn_grouped_mat_mul_ex.h"
#include "aclnn_add_rms_norm_ex.h"
#include <tuple>

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


at::Tensor grouped_matmul(at::Tensor x, at::Tensor w, at::Tensor group_list) {
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

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  auto x_sizes = x.sizes();
  auto x_strides = x.strides();
  aclTensor* x_acl = aclCreateTensor(x_sizes.data(), x.dim(), ACL_FLOAT16, x_strides.data(), 0, ACL_FORMAT_ND, x_sizes.data(), x.dim(), x.data_ptr());
  if (x_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for x");
  }
  // auto w_sizes = w.sizes();
  auto w_strides = w.strides();
  std::vector<int64_t> w_storage_sizes({w.size(0), w.size(2), w.size(1)});
  std::vector<int64_t> w_storage_strides({w_strides[0], w_strides[2], w_strides[1]});
  aclTensor* w_acl = aclCreateTensor(w_storage_sizes.data(), w.dim(), ACL_FLOAT16, w_storage_strides.data(), 0, ACL_FORMAT_ND, w_storage_sizes.data(), w.dim(), w.data_ptr());
  if (w_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for w");
  }
  auto group_list_sizes = group_list.sizes();
  auto group_list_strides = group_list.strides();
  aclTensor* group_list_acl = aclCreateTensor(group_list_sizes.data(), group_list.dim(), ACL_INT64, group_list_strides.data(), 0, ACL_FORMAT_ND, group_list_sizes.data(), group_list.dim(), group_list.data_ptr());
  if (group_list_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for group_list");
  }
  auto y_sizes = y.sizes();
  auto y_strides = y.strides();
  aclTensor* y_acl = aclCreateTensor(y_sizes.data(), y.dim(), ACL_FLOAT16, y_strides.data(), 0, ACL_FORMAT_ND, y_sizes.data(), y.dim(), y.data_ptr());
  if (y_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for y");
  }
  uint64_t workspace_size = 0;
  aclOpExecutor* handle = nullptr;
  if (aclnnGroupedMatMulExGetWorkspaceSize(x_acl, w_acl, group_list_acl, y_acl, &workspace_size, &handle) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to get workspace size");
  }
  auto options = at::TensorOptions().dtype(torch::kUInt8).device(x.device());
  auto workspace_tensor = at::empty({(int64_t)workspace_size}, options);
  if (aclnnGroupedMatMulEx(workspace_tensor.data_ptr(), workspace_size, handle, stream) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to execute grouped_matmul");
  }

  if (aclDestroyTensor(x_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for x");
  }
  if (aclDestroyTensor(w_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for w");
  }
  if (aclDestroyTensor(group_list_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for group_list");
  }
  if (aclDestroyTensor(y_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for y");
  }
  return y;
}


std::tuple<at::Tensor, at::Tensor> add_rms_norm(at::Tensor x, at::Tensor residual, at::Tensor weight, float epsilon) {
  TORCH_CHECK(x.dim() == 2 && residual.dim() == 2 && weight.dim() == 1,
              "add_rms_norm: x and residual must be 2D, weight must be 1D");
  TORCH_CHECK(x.size(1) == weight.size(0),
              "add_rms_norm: last dimension of x must match weight size, got ", x.size(1), " and ", weight.size(0));
  TORCH_CHECK(x.size(0) == residual.size(0) && x.size(1) == residual.size(1),
              "add_rms_norm: x and residual must have the same shape, got ", x.sizes(), " and ", residual.sizes());
  TORCH_CHECK(x.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
              "add_rms_norm: all input tensors must be contiguous");
  TORCH_CHECK(x.size(1) % 64 == 0,
              "add_rms_norm: last dimension must be a multiple of 64, got ", x.size(1));

  int num_tokens = x.size(0);
  int dim = x.size(1);
  at::Tensor y = at::empty({num_tokens, dim}, x.options());
  at::Tensor residual_output = at::empty({num_tokens, dim}, x.options());

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  
  // Create ACL tensors
  auto x_sizes = x.sizes();
  auto x_strides = x.strides();
  aclTensor* x_acl = aclCreateTensor(x_sizes.data(), x.dim(), ACL_FLOAT16, x_strides.data(), 0, ACL_FORMAT_ND, x_sizes.data(), x.dim(), x.data_ptr());
  if (x_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for x");
  }
  
  auto residual_sizes = residual.sizes();
  auto residual_strides = residual.strides();
  aclTensor* residual_acl = aclCreateTensor(residual_sizes.data(), residual.dim(), ACL_FLOAT16, residual_strides.data(), 0, ACL_FORMAT_ND, residual_sizes.data(), residual.dim(), residual.data_ptr());
  if (residual_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for residual");
  }
  
  auto weight_sizes = weight.sizes();
  auto weight_strides = weight.strides();
  aclTensor* weight_acl = aclCreateTensor(weight_sizes.data(), weight.dim(), ACL_FLOAT16, weight_strides.data(), 0, ACL_FORMAT_ND, weight_sizes.data(), weight.dim(), weight.data_ptr());
  if (weight_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for weight");
  }
  
  // Create epsilon tensor (optional parameter)
  at::Tensor epsilon_tensor = at::tensor({epsilon}, at::TensorOptions().dtype(torch::kFloat32).device(x.device()));
  auto epsilon_sizes = epsilon_tensor.sizes();
  auto epsilon_strides = epsilon_tensor.strides();
  aclTensor* epsilon_acl = aclCreateTensor(epsilon_sizes.data(), epsilon_tensor.dim(), ACL_FLOAT, epsilon_strides.data(), 0, ACL_FORMAT_ND, epsilon_sizes.data(), epsilon_tensor.dim(), epsilon_tensor.data_ptr());
  if (epsilon_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for epsilon");
  }
  
  auto y_sizes = y.sizes();
  auto y_strides = y.strides();
  aclTensor* y_acl = aclCreateTensor(y_sizes.data(), y.dim(), ACL_FLOAT16, y_strides.data(), 0, ACL_FORMAT_ND, y_sizes.data(), y.dim(), y.data_ptr());
  if (y_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for y");
  }

  auto residual_output_sizes = residual_output.sizes();
  auto residual_output_strides = residual_output.strides();
  aclTensor* residual_output_acl = aclCreateTensor(residual_output_sizes.data(), residual_output.dim(), ACL_FLOAT16, residual_output_strides.data(), 0, ACL_FORMAT_ND, residual_output_sizes.data(), residual_output.dim(), residual_output.data_ptr());
  if (residual_output_acl == nullptr) {
    throw std::runtime_error("Failed to create ACL tensor for residual_output");
  }

  // Get workspace size and execute
  uint64_t workspace_size = 0;
  aclOpExecutor* handle = nullptr;
  if (aclnnAddRMSNormExGetWorkspaceSize(x_acl, residual_acl, weight_acl, epsilon_acl, y_acl, residual_output_acl, &workspace_size, &handle) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to get workspace size");
  }
  auto options = at::TensorOptions().dtype(torch::kUInt8).device(x.device());
  auto workspace_tensor = at::empty({(int64_t)workspace_size}, options);
  if (aclnnAddRMSNormEx(workspace_tensor.data_ptr(), workspace_size, handle, stream) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to execute add_rms_norm");
  }

  // Clean up ACL tensors
  if (aclDestroyTensor(x_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for x");
  }
  if (aclDestroyTensor(residual_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for residual");
  }
  if (aclDestroyTensor(weight_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for weight");
  }
  if (aclDestroyTensor(epsilon_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for epsilon");
  }
  if (aclDestroyTensor(y_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for y");
  }
  if (aclDestroyTensor(residual_output_acl) != ACL_SUCCESS) {
    throw std::runtime_error("Failed to destroy ACL tensor for residual_output");
  }

  return std::make_tuple(y, residual_output);
}

void init_ffi_ops(py::module_ &&m) {
  m.def("swiglu", &swiglu, "Swiglu");
  m.def("grouped_matmul", &grouped_matmul, "GroupedMatMul");
  m.def("add_rms_norm", &add_rms_norm, "AddRMSNorm");
}

}
