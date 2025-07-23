#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/operation.h>

#include "aclnn_swi_glu_ex.h"
#include "dbg/dbg.h"

namespace native {

#define CHECK_ATB(condition) \
  do { \
    auto status = (condition); \
    if (status != atb::NO_ERROR) { \
      std::stringstream ss; \
      ss << "atb failed with error: " << status << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(ss.str()); \
    } \
  } while (0)


inline aclTensor* create_acl_tensor(atb::Tensor& src) {
  std::vector<int64_t> strides(src.desc.shape.dimNum, 1);
  for (int i = src.desc.shape.dimNum - 2; i >= 0; i--) {
    strides[i] = src.desc.shape.dims[i + 1] * strides[i + 1];
  }

  aclTensor* dst = aclCreateTensor(
    src.desc.shape.dims,
    src.desc.shape.dimNum,
    src.desc.dtype,
    strides.data(),
    0,
    src.desc.format,
    src.desc.shape.dims,
    src.desc.shape.dimNum,
    src.deviceData
  );

  return dst;
}

struct AclnnTensor {
  atb::Tensor atb_tensor;
  aclTensor *acl_tensor = nullptr;
  int idx = -1; // aclTensor index in aclExecutor
};

class AclnnOp: public atb::Operation {
public:
  explicit AclnnOp(const std::string& name): op_name(name) {}

  std::string GetName() const override {
    return op_name;
  }

  atb::Status Setup(const atb::VariantPack &pack, uint64_t &workspace_size, atb::Context *ctx) override {
    in_tensors.resize(this->GetInputNum());
    for (int i = 0; i < in_tensors.size(); ++i) {
      in_tensors[i] = std::make_shared<AclnnTensor>();
      in_tensors[i]->atb_tensor = pack.inTensors[i];
      in_tensors[i]->acl_tensor = create_acl_tensor(in_tensors[i]->atb_tensor);
      if (in_tensors[i]->acl_tensor == nullptr) {
        throw std::runtime_error("Failed to create ACL tensor for input " + std::to_string(i) + " for " + op_name);
      }
      in_tensors[i]->idx = i;
    }
    out_tensors.resize(this->GetOutputNum());
    for (int i = 0; i < out_tensors.size(); ++i) {
      out_tensors[i] = std::make_shared<AclnnTensor>();
      out_tensors[i]->atb_tensor = pack.outTensors.at(i);
      out_tensors[i]->acl_tensor = create_acl_tensor(out_tensors[i]->atb_tensor);
      if (out_tensors[i]->acl_tensor == nullptr) {
        throw std::runtime_error("Failed to create ACL tensor for output " + std::to_string(i) + " for " + op_name);
      }
      out_tensors[i]->idx = i;
    }
    CHECK_ATB(SetAclnnWorkspaceAndExecutor());
    workspace_size = this->workspace_size;
    return atb::NO_ERROR;
  }

  atb::Status Execute(const atb::VariantPack &pack, uint8_t *workspace, uint64_t workspace_size, atb::Context *ctx) override {
    aclrtStream stream = ctx->GetExecuteStream();
    assert(stream != nullptr);
    CHECK_ATB(UpdateAclnnTensors(pack));
    CHECK_ATB(ExecuteAclnnOp(workspace, stream));
    return atb::NO_ERROR;
  }

  atb::Status UpdateAclnnTensors(const atb::VariantPack &pack) {
    for (int i = 0; i < in_tensors.size(); ++i) {
      in_tensors[i]->atb_tensor = pack.inTensors[i];
      if (aclSetInputTensorAddr(acl_executor, in_tensors[i]->idx, in_tensors[i]->acl_tensor, in_tensors[i]->atb_tensor.deviceData) != ACL_SUCCESS) {
        throw std::runtime_error("Failed to update input tensor " + std::to_string(i) + " address for " + op_name);
      }
    }
    for (int i = 0; i < out_tensors.size(); ++i) {
      out_tensors[i]->atb_tensor = pack.outTensors[i];
      if (aclSetOutputTensorAddr(acl_executor, out_tensors[i]->idx, out_tensors[i]->acl_tensor, out_tensors[i]->atb_tensor.deviceData) != ACL_SUCCESS) {
        throw std::runtime_error("Failed to update output tensor " + std::to_string(i) + " address for " + op_name);
      }
    }
    return atb::NO_ERROR;
  }

  virtual atb::Status SetAclnnWorkspaceAndExecutor() = 0;
  virtual atb::Status ExecuteAclnnOp(uint8_t *workspace, aclrtStream stream) = 0;


  std::string op_name;
  aclOpExecutor *acl_executor = nullptr;
  uint64_t workspace_size = 0;
  std::vector<std::shared_ptr<AclnnTensor>> in_tensors;
  std::vector<std::shared_ptr<AclnnTensor>> out_tensors;
};

class SwiGluEx: public AclnnOp {
public:
  SwiGluEx(const std::string& name = "SwiGluEx"): AclnnOp(name) {}
  atb::Status InferShape(const atb::SVector<atb::TensorDesc> &in_tensor_descs, atb::SVector<atb::TensorDesc> &out_tensor_descs) const override {
    out_tensor_descs[0] = in_tensor_descs[0];
    out_tensor_descs[0].shape.dims[out_tensor_descs[0].shape.dimNum - 1] = out_tensor_descs[0].shape.dims[out_tensor_descs[0].shape.dimNum - 1] / 2;
    return atb::NO_ERROR;
  }
  uint32_t GetInputNum() const override {
    return 1;
  }
  uint32_t GetOutputNum() const override {
    return 1;
  }

  atb::Status SetAclnnWorkspaceAndExecutor() override {
    if (aclnnSwiGluExGetWorkspaceSize(in_tensors[0]->acl_tensor, out_tensors[0]->acl_tensor, &workspace_size, &acl_executor) != ACL_SUCCESS) {
      throw std::runtime_error("Failed to get workspace size for SwiGluEx");
    }
    if (aclSetAclOpExecutorRepeatable(acl_executor) != ACL_SUCCESS) {
      throw std::runtime_error("Failed to set ACL op executor repeatable for SwiGluEx");
    }
    return atb::NO_ERROR;
  };

  atb::Status ExecuteAclnnOp(uint8_t *workspace, aclrtStream stream) override {
    if (aclnnSwiGluEx(workspace, this->workspace_size, acl_executor, stream) != ACL_SUCCESS) {
      throw std::runtime_error("Failed to execute SwiGluEx");
    }
    return atb::NO_ERROR;
  }
};


}
