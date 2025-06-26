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


file(GLOB KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp)

message(STATUS "kernels: ${KERNELS}")
message(STATUS "csrcs: ${SRCS}")

include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_PATH}/include
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/aarch64-linux/include/experiment/platform
  ${ASCEND_HOME_PATH}/x86_64-linux/include/experiment/platform
)


ascendc_library(kernels STATIC ${KERNELS})
pybind11_add_module(ascend910a_extras_C ${SRCS})

target_link_directories(
  ascend910a_extras_C
  PRIVATE
  ${TORCH_NPU_PATH}/lib
  ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(
  ascend910a_extras_C
  PUBLIC
  ${TORCH_LIBRARIES}
  torch_npu
  kernels
  ascendcl
  platform
)