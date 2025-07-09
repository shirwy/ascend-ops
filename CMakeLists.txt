cmake_minimum_required(VERSION 3.16.0)
project(ascend910a_extras)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RUN_MODE "npu" CACHE STRING "run mode: npu")

message("TORCH_NPU_PATH: ${TORCH_NPU_PATH}")
message("ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
message("SOC_VERSION: ${SOC_VERSION}")


set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)


include(csrc/opdev/cmake/func.cmake)
get_system_info(SYSTEM_INFO)
# FIXME: build_out is a hardcoded path in build.sh
set(OPP_RUN_PKG "${CMAKE_SOURCE_DIR}/csrc/opdev/build_out/custom_opp_${SYSTEM_INFO}.run")
set(OPP_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/opp_install")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "OPP_RUN_PKG: ${OPP_RUN_PKG}")
message(STATUS "OPP_INSTALL_DIR: ${OPP_INSTALL_DIR}")

add_custom_command(
  OUTPUT ${OPP_RUN_PKG}
  COMMAND bash ${CMAKE_SOURCE_DIR}/csrc/opdev/build.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/csrc/opdev
  DEPENDS ${CMAKE_SOURCE_DIR}/csrc/opdev/build.sh
  COMMENT "Building custom OPP .run package"
)

add_custom_command(
  OUTPUT ${OPP_INSTALL_DIR}/install.done
  COMMAND bash ${OPP_RUN_PKG} --install-path=${OPP_INSTALL_DIR}
  COMMAND touch ${OPP_INSTALL_DIR}/install.done
  DEPENDS ${OPP_RUN_PKG}
)

add_custom_target(custom_opp ALL
  DEPENDS ${OPP_INSTALL_DIR}/install.done
)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/ffi/*.cpp)
message(STATUS "SRCS: ${SRCS}")
pybind11_add_module(ascend910a_extras_C ${SRCS})

add_dependencies(ascend910a_extras_C custom_opp)


target_include_directories(
  ascend910a_extras_C
  PRIVATE

  ${pybind11_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc

  # custom opp
  ${OPP_INSTALL_DIR}/vendors/customize/op_proto/inc
  ${OPP_INSTALL_DIR}/vendors/customize/op_api/include
)

target_link_directories(
  ascend910a_extras_C
  PRIVATE

  ${TORCH_NPU_PATH}/lib
  ${ASCEND_HOME_PATH}/lib64
  ${ASCEND_HOME_PATH}/compiler/lib64/stub

  # custom opp
  ${OPP_INSTALL_DIR}/vendors/customize/op_api/lib
)

target_link_libraries(
  ascend910a_extras_C
  PUBLIC
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
  platform
  graph
  ge_runner

  # custom opp
  cust_opapi
)


install(TARGETS ascend910a_extras_C
        LIBRARY DESTINATION ascend910a_extras
        ARCHIVE DESTINATION ascend910a_extras
        RUNTIME DESTINATION ascend910a_extras)

install(DIRECTORY ${OPP_INSTALL_DIR}/
        DESTINATION ascend910a_extras/opp_install)
