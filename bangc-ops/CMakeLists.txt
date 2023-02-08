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
  message("-- Address sanitizer enabled")
  set(CMAKE_ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_ASAN_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_ASAN_FLAGS}")
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
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Werror -Wdeprecated-declarations -Wall -std=c++11 -fPIC -pthread")
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

foreach(kernel ${build_kernel})
  if (NOT IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/kernels/${kernel}")
    message(WARNING "kernel/${kernel} is not a directory, ${kernel} is an alias")
    continue()
  endif()
  file(GLOB_RECURSE src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}/*.mlu")
  file(GLOB_RECURSE arch_binary_files ${arch_binary_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${kernel}/${MLUOP_TARGET_CPU_ARCH}/*.o")
endforeach()

file(GLOB_RECURSE archive_binary_files ${archive_binary_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/lib/*.a")

file(GLOB_RECURSE core_src_files ${core_src_files} "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp")
# set(src_files ${src_files} "${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${MLUOP_SYMBOL_VIS_FILE})
  message(STATUS "${MLUOP_SYMBOL_VIS_FILE} exists.")
else()
  message(FATAL_ERROR "${MLUOP_SYMBOL_VIS_FILE} doesn't exist.")
endif()

set(LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${MLUOP_SYMBOL_VIS_FILE}")
message(STATUS "LINK_FLAGS:${LINK_FLAGS}")
add_library(mluopscore STATIC ${core_src_files})

if (archive_binary_files MATCHES ".a")
  if ("${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/wrapper.cpp" IN_LIST src_files)
    list(REMOVE_ITEM src_files "${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/wrapper.cpp")
  endif()
  message(STATUS "Build MLUOP with static object file:${archive_binary_files}")
else()
  file(GLOB_RECURSE wrapper_binary_files ${wrapper_binary_files} "${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/lib/*.o")
  message(STATUS "Build External lib with object files.")
  execute_process(COMMAND g++ -c -std=c++11 -fPIC -I ${CMAKE_CURRENT_SOURCE_DIR}
                                 -I ${NEUWARE_HOME}/include
                                 ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/wrapper.cpp
                                 -o ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/lib/wrapper.cpp.o)
  execute_process(COMMAND ar -rc ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/lib/libextops.a
                                  ${wrapper_binary_files} ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/lib/wrapper.cpp.o)
  execute_process(COMMAND rm ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_wrapper/lib/wrapper.cpp.o)
endif()

bang_add_library(mluops SHARED ${src_files})
target_link_libraries(mluops
  -Wl,--start-group
  ${arch_binary_files} ${wrapper_binary_files}
  -Wl,--whole-archive ${archive_binary_files} mluopscore -Wl,--no-whole-archive
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

################################################################################
# Build MLUOP GTEST
################################################################################
option(MLU_OP_BUILD_GTEST "Build mlu-ops gtest" ON)
message("-- MLU_OP_BUILD_GTEST=${MLU_OP_BUILD_GTEST}")
if(${MLU_OP_BUILD_GTEST} MATCHES "ON")
  message("-- Build MLUOP Gtest")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/test/mlu_op_gtest" "mlu_op_gtest")
endif()
