cmake_minimum_required(VERSION 3.3)

project(mluops VERSION ${BUILD_VERSION})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/test")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fPIC -std=c++11 -pthread -pipe")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -fPIC")

################################################################################
# Neuware Evironment and BANG Setup
################################################################################

if(${MLUOP_BUILD_COVERAGE_TEST} MATCHES "ON")
  message("-- MLU_OP_COVERAGE_TEST=${MLUOP_BUILD_COVERAGE_TEST}")
  set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "-u__llvm_profile_runtime ${NEUWARE_HOME}/lib/clang/11.0.0/lib/linux/libclang_rt.profile-x86_64.a")
  set(CNRT_DUMP_PGO 1)
  set(CNRT_PGO_OUTPUT_DIR=output)
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fprofile-instr-generate -fcoverage-mapping -D COVERAGE")
endif()

################################################################################
# ASAN Check and memcheck
################################################################################
# -- leak detector
if(${MLUOP_BUILD_ASAN_CHECK} MATCHES "ON")
  message("--Address sanitizer enabled")
  set(CMAKE_ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_ASAN_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_ASAN_FLAGS}")
endif()

# check `NEUWARE_HOME` env
message(${NEUWARE_HOME})
if(EXISTS ${NEUWARE_HOME})
  include_directories("${NEUWARE_HOME}/include")
  link_directories("${NEUWARE_HOME}/lib64")
  link_directories("${NEUWARE_HOME}/lib")
  set(NEUWARE_ROOT_DIR "${NEUWARE_HOME}")
else()
  message(FATAL_ERROR "NEUWARE directory cannot be found, refer README.md to prepare NEUWARE_HOME environment.")
endif()

# setup cmake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_SOURCE_DIR}/cmake"
  "${NEUWARE_HOME}/cmake"
  "${NEUWARE_HOME}/cmake/modules"
)

# include FindBANG.cmake and check cncc
find_package(BANG)
if(NOT BANG_FOUND)
  message(FATAL_ERROR "BANG cannot be found.")
elseif (NOT BANG_CNCC_EXECUTABLE)
  message(FATAL_ERROR "cncc not found, please ensure cncc is in your PATH env or set variable BANG_CNCC_EXECUTABLE from cmake. Otherwise you should check path used by find_program(BANG_CNCC_EXECUTABLE) in FindBANG.cmake")
endif()

# setup cncc flags
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fPIC -Wall -Werror -std=c++11 -pthread")
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_220"
                                         "--bang-mlu-arch=mtp_270"
                                         "--bang-mlu-arch=mtp_290"
                                         "--bang-mlu-arch=mtp_372"
)

file(GLOB_RECURSE src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.mlu" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp")
file(GLOB_RECURSE obj_files ${obj_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*/${MLUOPS_TARGET_CPU_ARCH}/*.o")
# set(src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp")

bang_add_library(mluops SHARED ${src_files})
target_link_libraries(mluops cnrt cndrv)
target_link_libraries(mluops ${obj_files})
set_target_properties(mluops PROPERTIES
  OUTPUT_NAME "mluops"
  PREFIX      "lib"
  VERSION     "${BUILD_VERSION}"
  SOVERSION   "${MAJOR_VERSION}"
)

################################################################################
# Build MLUOP GTEST
################################################################################
option(MLU_OP_BUILD_GTEST "Build mlu-ops gtest" ON)
message("-- MLU_OP_BUILD_GTEST=${MLU_OP_BUILD_GTEST}")
if(${MLU_OP_BUILD_GTEST} MATCHES "ON")
  message("-- Build MLUOP Gtest")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test/mlu_op_gtest" "mlu_op_gtest")
endif()
