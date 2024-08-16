cmake_minimum_required(VERSION 3.5)

project(mluops VERSION ${BUILD_VERSION})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/test")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wdeprecated-declarations -fPIC -std=c++11 -pthread -pipe")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Wl,--gc-sections -fPIC")

################################################################################
# Build Environment
################################################################################
set(MLUOP_BUILD_SPECIFIC_OP ${MLUOP_BUILD_SPECIFIC_OP})
message("-- MLUOP_BUILD_SPECIFIC_OP=${MLUOP_BUILD_SPECIFIC_OP}")
set(MLUOP_MLU_ARCH_LIST ${MLUOP_MLU_ARCH_LIST})
message("-- MLUOP_MLU_ARCH_LIST=${MLUOP_MLU_ARCH_LIST}")
set(MLUOP_BUILD_PERF ${MLUOP_BUILD_PERF})
message("-- MLUOP_BUILD_PERF=${MLUOP_BUILD_PERF}")

################################################################################
# Environment and BANG Setup
################################################################################
if (NOT CMAKE_BUILD_TYPE)
  set(_CMAKE_BUILD_TYPE_LOWER "release")
else()
  string(TOLOWER ${CMAKE_BUILD_TYPE} _CMAKE_BUILD_TYPE_LOWER)
endif()

if(${MLUOP_BUILD_COVERAGE_TEST} MATCHES "ON")
  message("-- MLU_OP_COVERAGE_TEST=${MLUOP_BUILD_COVERAGE_TEST}")
  set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "-u__llvm_profile_runtime ${NEUWARE_HOME}/lib/clang/11.1.0/lib/linux/libclang_rt.profile-x86_64.a")
  set(CNRT_DUMP_PGO 1)
  set(CNRT_PGO_OUTPUT_DIR=output)
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fprofile-instr-generate -fcoverage-mapping -D COVERAGE")
endif()

################################################################################
# Add readperf
################################################################################
if(MLUOP_BUILD_PERF)
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -fbang-instrument-kernels")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -DMLUOP_BANG_PERF")
  add_compile_definitions(MLUOP_BANG_PERF)
endif()

################################################################################
# ASAN Check and memcheck
################################################################################
# -- leak detector
if(${MLUOP_BUILD_ASAN_CHECK} MATCHES "ON")
  message("-- Address sanitizer enabled")
  set(CMAKE_ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_ASAN_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_ASAN_FLAGS}")
endif()

# -- BANG memcheck
if(${MLUOP_BUILD_BANG_MEMCHECK} MATCHES "ON")
  message("-- BANG memcheck enabled")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -mllvm -enable-mlisa-sanitizer")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Xbang-cnas -O1") # XXX to reduce cnas compilation time
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -g") # to show file line number
endif()

# check `NEUWARE_HOME` env
message("-- NEUWARE_HOME=${NEUWARE_HOME}")
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
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Werror -Wdeprecated-declarations -Wall -std=c++11 -fPIC -pthread --neuware-path=${NEUWARE_HOME}")
if(${_CMAKE_BUILD_TYPE_LOWER} MATCHES "debug")
  message(STATUS "build debug version")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -g3 -O0")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")
elseif(${_CMAKE_BUILD_TYPE_LOWER} MATCHES "release")
  message(STATUS "build release version")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

if (NOT MLUOP_MLU_ARCH_LIST)
  message(STATUS "build all arch")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "--bang-mlu-arch=mtp_372"
                                           "--bang-mlu-arch=mtp_592")
else()
  foreach (arch ${MLUOP_MLU_ARCH_LIST})
    set(CNCC_FLAGS_ARCH ${CNCC_FLAGS_ARCH} "--bang-mlu-arch=${arch}" )
  endforeach ()
  message(STATUS "build specific arch:${CNCC_FLAGS_ARCH}")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}" "${CNCC_FLAGS_ARCH}")
endif()

message(STATUS "BANG_CNCC_FLAGS:${BANG_CNCC_FLAGS}")

if (${MLUOP_PACKAGE_INFO_SET} MATCHES "ON")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_package_info.cmake)
endif()

# resolve kernel dependency in MLUOP_BUILD_SPECIFIC_OP
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/mluop_kernel_depends.cmake)
# populate_op is provided by mluop_kernel_depends.cmake
if (MLUOP_BUILD_SPECIFIC_OP)
  populate_op(MLUOP_BUILD_SPECIFIC_OP SPECIFIC_OP ${MLUOP_BUILD_SPECIFIC_OP})
  message(STATUS "MLUOP_BUILD_SPECIFIC_OP (populated): ${MLUOP_BUILD_SPECIFIC_OP}")
endif()

if (NOT MLUOP_BUILD_SPECIFIC_OP)
  message(STATUS "Build all kernels")
  file(GLOB all_kernels "${CMAKE_CURRENT_LIST_DIR}/kernels/*")
  foreach (o ${all_kernels})
    if (IS_DIRECTORY ${o})
      get_filename_component(kernelname ${o} NAME)
      set(build_kernel ${kernelname} ${build_kernel})
    endif()
  endforeach ()
else()
  set(build_kernel ${MLUOP_BUILD_SPECIFIC_OP})
endif()

list(SORT build_kernel)
message(STATUS "build_kernel:[${build_kernel}]")
 
file(GLOB all_kernels "${CMAKE_CURRENT_LIST_DIR}/kernels/*")
foreach(kernel ${build_kernel} )
  set(kernel_parent_dir '')
  foreach (o ${all_kernels})
    if (IS_DIRECTORY ${o})
      get_filename_component(kernelname ${o} NAME)
      if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/kernels/${kernel} OR
         IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/kernels/${kernelname}/${kernel})
        set(kernel_parent_dir ${kernelname})
      endif()
    endif()
  endforeach ()
  if (IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/kernels/${kernel}")
    file(GLOB_RECURSE src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}/*.mlu")
  elseif(IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/kernels/${kernel_parent_dir}/${kernel}")
    file(GLOB_RECURSE src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel_parent_dir}/${kernel}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel_parent_dir}/${kernel}/*.mlu")
  else()
    message(WARNING "kernel/${kernel} is not a directory, ${kernel} is an alias")
    continue()
  endif()
endforeach()
 
file(GLOB_RECURSE src_helper_files ${src_helper_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/utils/cnnl_helper.cpp")
file(GLOB_RECURSE core_src_files ${core_src_files} "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp")
# set(src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp")

if(EXISTS ${CMAKE_BINARY_DIR}/${MLUOP_SYMBOL_VIS_FILE})
  message(STATUS "${MLUOP_SYMBOL_VIS_FILE} exists.")
else()
  message(FATAL_ERROR "${MLUOP_SYMBOL_VIS_FILE} doesn't exist.")
endif()

set(LINK_FLAGS "-Wl,--version-script=${CMAKE_BINARY_DIR}/${MLUOP_SYMBOL_VIS_FILE}")
message(STATUS "LINK_FLAGS:${LINK_FLAGS}")
add_library(mluopscore STATIC ${core_src_files})

target_link_libraries(mluopscore cnnl cnrt cndrv)
bang_add_library(mluops SHARED ${src_files} ${src_helper_files})

if(${MLUOP_BUILD_STATIC} MATCHES "ON")
  message("-- Build MLUOP static")
  bang_add_library(mluops_static STATIC ${src_files} ${src_helper_files} ${core_src_files})
endif()

target_link_libraries(mluops
  -Wl,--start-group
  ${arch_binary_files}
  -Wl,--whole-archive ${archive_binary_files} -Wl,--no-whole-archive
  mluopscore
  cnnl
  cnrt cndrv dl
  -Wl,--end-group
)
target_link_libraries(mluops ${LINK_FLAGS})
set_target_properties(mluops PROPERTIES
  OUTPUT_NAME "mluops"
  PREFIX      "lib"
  VERSION     "${BUILD_VERSION}"
  SOVERSION   "${MAJOR_VERSION}"
)

if(${MLUOP_BUILD_STATIC} MATCHES "ON")
  set_target_properties(mluops_static PROPERTIES
    OUTPUT_NAME "mluops"
    PREFIX      "lib"
    VERSION     "${BUILD_VERSION}"
    SOVERSION   "${MAJOR_VERSION}"
  )
endif()

################################################################################
# Build MLUOP GTEST
################################################################################
option(MLUOP_BUILD_GTEST "Build mlu-ops gtest" ON)
message("-- MLUOP_BUILD_GTEST=${MLUOP_BUILD_GTEST}")
if(${MLUOP_BUILD_GTEST} MATCHES "ON")
  message("-- Build MLUOP Gtest")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test/mlu_op_gtest" "mlu_op_gtest")
endif()
