
#include <vector>
#include <map>
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

extern void init_ffi_graph(py::module_ &&m);
extern void init_ffi_ops(py::module_ &&m);

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


}

PYBIND11_MODULE(ascend910a_extras_C, m) {
  m.def("print_info", &native::print_info, "Print info about the device");

  native::init_ffi_ops(m.def_submodule("ops"));
  native::init_ffi_graph(m.def_submodule("graph"));
}
