cmake_minimum_required(VERSION 3.3)
cmake_policy(SET CMP0048 NEW) # Use project(... VERSION ...)
project(cntest)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# check
message("[CNNL GTEST STYLE]: ")
set(CNNL_BUILD_SPECIFIC_GTEST_OP ${CNNL_BUILD_SPECIFIC_OP} ${CNNL_BUILD_SPECIFIC_GTEST_OP})
if ( CNNL_BUILD_SPECIFIC_GTEST_OP )
  list(REMOVE_DUPLICATES CNNL_BUILD_SPECIFIC_GTEST_OP)
endif ()
execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/gtest_style.py ${CNNL_BUILD_SPECIFIC_GTEST_OP} RESULT_VARIABLE STYLE_CHECK)
if (STYLE_CHECK)
  message(FATAL_ERROR "-- Pleace check gtest code manually.")
endif()


message("[CNNL GTEST REGISTER OP]: ")
file(GLOB CASE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/gtest/*case_list.cpp")
execute_process(COMMAND rm ${CASE_LIST})
execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/gtest/register_op.py ${CNNL_BUILD_SPECIFIC_GTEST_OP})



## Try to find libxml2 and protobuf
# @ref cmake wiki: How-To-Find-Libraries
find_package(PkgConfig)
# XXX If we install cmake-modules, we could use find_package(LibXML2) directly
pkg_check_modules(PC_LIBXML QUIET libxml-2.0)
set(LIBXML2_DEFINITIONS ${PC_LIBXML_CFLAGS_OTHER})
find_path(LIBXML2_INCLUDE_DIR libxml/xpath.h
  HINTS ${PC_LIBXML_INCLUDEDIR} ${PC_LIBXML_INCLUDE_DIRS}
  PATH_SUFFIXES libxml2
)
# prefer to search static library
# TODO(jiaminghao) consider link static libraries or add libxml2 sources into cnnl_gtest for convenience
find_library(LIBXML2_LIBRARY NAMES xml2 libxml2
  HINTS ${PC_LIBXML_LIBDIR} ${PC_LIBXML_LIBRARY_DIRS}
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibXml2 DEFAULT_MSG
  LIBXML2_LIBRARY LIBXML2_INCLUDE_DIR
)
mark_as_advanced(LIBXML2_INCLUDE_DIR LIBXML2_LIBRARY)
set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARY})
set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR})


# prefer to use protobuf static libraries
set(ORIG_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
find_package(Protobuf)
# FindProtobuf will generate important variables (could be overrided):
# (cache) PROTOBUF_LIBRARY, (set) PROTOBUF_LIBRARIES
# (cache) PROTOBUF_INCLUDE_DIR, (set) PROTOBUF_INCLUDE_DIRS
# PROTOBUF_PROTOC_EXECUTABLE
# TODO(jiaminghao) treat these variables as hints, consider set another set of protobuf searc varaibles base on them.
set(CMAKE_FIND_LIBRARY_SUFFIXES ${ORIG_FIND_LIBRARY_SUFFIXES})


message(STATUS "libxml2 include: " ${LIBXML2_INCLUDE_DIRS})
message(STATUS "libxml2 library: " ${LIBXML2_LIBRARIES})
message(STATUS "protobuf include: " ${PROTOBUF_INCLUDE_DIRS})
message(STATUS "protobuf libraries: " ${PROTOBUF_LIBRARIES})
message(STATUS "protoc executable: " ${PROTOBUF_PROTOC_EXECUTABLE})


include_directories(${LIBXML2_INCLUDE_DIRS})
include_directories(${PROTOBUF_INCLUDE_DIR})



if(${TARGET_CPU_ARCH} MATCHES "aarch64.*linux")
  # set(DEFAULT_PROTOBUF_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aarch64/protobuf-2.6.1/include/")
  # set(DEFAULT_PROTOBUF_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aarch64/protobuf-2.6.1/lib/")
  # set(DEFAULT_LIBXML2_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aarch64/libxml2-2.7.4/include/libxml2")
  # set(DEFAULT_LIBXML2_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aarch64/libxml2-2.7.4/lib/")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -mavx2")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -mavx2")
endif()


message("[CNNL GTEST PROTOC]: ")
set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cntest_proto")
file(GLOB PROTO_FILES "${PROTO_PATH}/cnnl_test.proto")
if (NOT EXISTS ${PROTO_FILES})
  message(STATUS "cnnl_test.proto not found, trying to git submodule init")
  execute_process(
    COMMAND git submodule update --init
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  file(GLOB PROTO_FILES "${PROTO_PATH}/cnnl_test.proto")
endif()


# protoc code
set(PROTOBUF_PROTOC_EXECUTABLE protoc CACHE STRING "protoc command location")
execute_process(COMMAND ${CMAKE_COMMAND} -E remove cnnl_test.pb.cc  cnnl_test.pb.h WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
add_custom_target(cnnl_build_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS} ${PROTO_FILES})
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
add_library(cnnl_test_proto ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(cnnl_test_proto cnnl_build_proto)



# TODO(jiaminghao) generate and use FindCNNL cmake modules
set(CNNL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../" CACHE PATH "Directory for cnnl.h")
find_path(LIBCNNL_INCLUDE_DIR cnnl.h
  HINTS
  ${CNNL_DIR}
  REQUIRED
)
message(STATUS "Found cnnl.h under ${LIBCNNL_INCLUDE_DIR}")
include_directories("${LIBCNNL_INCLUDE_DIR}")


# Detect if current scope has a parent
if ( ${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME} )
  message(STATUS "Build ${PROJECT_NAME} independently")
  include(GNUInstallDirs)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCNPAPI_LIB_CNNL")
  option(CNNL_BUILD_STATIC "Build CNNL static library" ON)


  if (NOT EXISTS ${NEUWARE_HOME})
    message(FATAL_ERROR "NEUWARE_HOME cannot be found, refer README.md to prepare NEUWARE_HOME environment")
  endif()
  message ("NEUWARE_HOME: ${NEUWARE_HOME}")
  # TODO(jiaminghao) we should consider use cmake find modules for cndrv and cnrt instead of hard coded library locatoin
  include_directories("${NEUWARE_HOME}/include")
  if ( ${TARGET_CPU_ARCH} MATCHES ".*aarch64.*" AND EXISTS ${NEUWARE_HOME}/edge )
    link_directories("${NEUWARE_HOME}/edge/lib64")
  else()
    link_directories("${NEUWARE_HOME}/lib64")
  endif()


  set(CNNL_LIBDIR ${CNNL_LIBDIR})
  find_library(LIBCNNL_LIBRARY NAMES cnnl libcnnl
    HINTS
    ${CNNL_LIBDIR}
    REQUIRED
  )
  if (LIBCNNL_LIBRARY)
    message(STATUS "Found libcnnl.so: ${LIBCNNL_LIBRARY}")
    add_library(cnnl INTERFACE)
    target_link_libraries(cnnl INTERFACE -Wl,--no-as-needed cnbin -Wl,--as-needed)
    target_link_libraries(cnnl INTERFACE ${LIBCNNL_LIBRARY} cnrt cndrv cndev dl)
    # link_directories("${LIBCNNL_LIBRARY}")
  else(LIBCNNL_LIBRARY)
    message(FATAL_ERROR "libcnnl.so not found, please set CNNL_LIBDIR")
  endif(LIBCNNL_LIBRARY)


  find_library(LIBCNNL_LIBRARY_STATIC NAMES libcnnl.a
    HINTS
    ${CNNL_LIBDIR}
    QUIET
  )
  if (LIBCNNL_LIBRARY_STATIC)
    message(STATUS "Found libcnnl.a ${LIBCNNL_LIBRARY_STATIC}")
    add_library(cnnl_static INTERFACE)
    target_link_libraries(cnnl_static INTERFACE -Wl,--no-as-needed cnbin -Wl,--as-needed)
    target_link_libraries(cnnl_static INTERFACE ${LIBCNNL_LIBRARY_STATIC} cnrt cndrv cndev dl)
  endif (LIBCNNL_LIBRARY_STATIC)


  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  link_libraries(Threads::Threads)


  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${NEUWARE_HOME}/cmake"
    "${NEUWARE_HOME}/cmake/modules"
    "${CMAKE_CURRENT_LIST_DIR}/../../cmake"
  )
  set(NEUWARE_ROOT_DIR ${NEUWARE_HOME})
  include(aarch64_toolchain)
  find_package(BANG)
  if(NOT BANG_FOUND)
    message(FATAL_ERROR "BANG cannot be found.")
  elseif (NOT BANG_CNCC_EXECUTABLE)
    message(FATAL_ERROR "cncc not found, please ensure cncc is in your PATH env or set variable BANG_CNCC_EXECUTABLE from cmake. Otherwise you should check path used by find_program(BANG_CNCC_EXECUTABLE) in FindBANG.cmake")
  endif()


  # TODO remember add new arch
  # setup cncc flags
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Wall -Werror -fPIC -std=c++11 -pthread -DCNPAPI_LIB_CNNL --target=${TARGET_CPU_ARCH}")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --neuware-path=${NEUWARE_HOME}")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS}"
    "--bang-arch=compute_30"
    "--bang-mlu-arch=mtp_592"
    "--bang-wram-align64"
    "--bang-mlu-arch=mtp_322"
  )


else (${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME})
  message(STATUS "Build ${PROJECT_NAME} as sub-project of ${CMAKE_PROJECT_NAME}")
endif ( ${PROJECT_NAME} STREQUAL ${CMAKE_PROJECT_NAME} )



set(GOOGLETEST_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/googletest"
                       "${CMAKE_CURRENT_SOURCE_DIR}/googletest/include")
set(GOOGLETEST_SRC     "${CMAKE_CURRENT_SOURCE_DIR}/googletest/src")
set(CNNL_GTEST_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include"
                       "${CMAKE_CURRENT_SOURCE_DIR}/tests"
                       "${CMAKE_CURRENT_SOURCE_DIR}/src/gtest"
                       "${CMAKE_CURRENT_SOURCE_DIR}/src/zoo"
                       "${CMAKE_CURRENT_SOURCE_DIR}/src/gtest/extra_lib/jsoncppDist/json"
                       "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
set(CNNL_GTEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/gtest/*.cpp"
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/gtest/extra_lib/jsoncppDist/*.cpp"
)


include_directories("${GOOGLETEST_INCLUDE}")
include_directories("${PROTO_PATH}")
include_directories("${CNNL_GTEST_INCLUDE}")


file(GLOB_RECURSE gtest_mlu_files FOLLOW_SYMLINKS "${CMAKE_CURRENT_LIST_DIR}/src/internal_kernel/*")
bang_add_library(cnnl_gtest_kernels STATIC "${gtest_mlu_files}")


file(GLOB SRC_DIR "${GOOGLETEST_SRC}/*.cc")
#add_library(gtest_shared SHARED ${SRC_DIR})
add_library(gtest_shared STATIC ${SRC_DIR}) # for runtime convenience


file(GLOB CNNL_TEST_DIR ${CNNL_GTEST_SRC})


if (CNNL_BUILD_SPECIFIC_GTEST_OP)
  foreach(op ${CNNL_BUILD_SPECIFIC_GTEST_OP})
    file(GLOB_RECURSE OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/zoo/${op}/*.cpp")
    list(APPEND CNNL_OP_DIR ${OP_DIR})
  endforeach()
else()
  file(GLOB_RECURSE CNNL_OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/zoo/*/*.cpp")
endif()


add_library(cnnl_gtest_obj OBJECT ${CNNL_TEST_DIR} ${CNNL_OP_DIR} ${PROTO_HDRS})
add_dependencies(cnnl_gtest_obj cnnl_build_proto)
remove_definitions(-DROUND_HALF_UP)
add_executable(cnnl_gtest $<TARGET_OBJECTS:cnnl_gtest_obj>)
#target_link_libraries(cnnl_gtest cnnl_core -Wl,--whole-archive cnnl_wrapper -Wl,--no-whole-archive)
target_link_libraries(cnnl_gtest cnnl)
set(targets_cnnl_gtest cnnl_gtest)


set(CNNL_API_GTEST_INCLUDE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/cnnl_api_gtest/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/cnnl_api_gtest/src/gtest"
                "${CMAKE_CURRENT_SOURCE_DIR}/cnnl_api_gtest/src/gtest/extra_lib/jsoncppDist/json")
set(CNNL_API_GTEST_SRC
                "${CMAKE_CURRENT_SOURCE_DIR}/cnnl_api_gtest/src/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cnnl_api_gtest/src/gtest/*.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/cnnl_api_gtest/src/gtest/extra_lib/jsoncppDist/*.cpp")
include_directories("${CNNL_API_GTEST_INCLUDE}")
file(GLOB CNNL_API_TEST_DIR ${CNNL_API_GTEST_SRC})
if (CNNL_BUILD_SPECIFIC_GTEST_OP)
  foreach(op ${CNNL_BUILD_SPECIFIC_GTEST_OP})
    file(GLOB_RECURSE TEST_OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cnnl_api_gtest/src/gtest/${op}/*.cpp")
    list(APPEND CNNL_API_TEST_OP_DIR ${TEST_OP_DIR})
  endforeach()
else()
  file(GLOB_RECURSE CNNL_API_TEST_OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cnnl_api_gtest/src/gtest/*/*.cpp")
endif()
add_library(cnnl_api_gtest_obj OBJECT ${CNNL_API_TEST_DIR} ${CNNL_API_TEST_OP_DIR})
add_dependencies(cnnl_api_gtest_obj cnnl_build_proto)
add_executable(cnnl_api_gtest $<TARGET_OBJECTS:cnnl_api_gtest_obj>)
target_link_libraries(cnnl_api_gtest -Wl,--no-as-needed cnbin -Wl,--as-needed)
target_link_libraries(cnnl_api_gtest gtest_shared cnnl cnrt cndev stdc++ m dl)
set(targets_cnnl_api_gtest cnnl_api_gtest)


if (${CNNL_BUILD_STATIC})
  add_executable(cnnl_gtest_static $<TARGET_OBJECTS:cnnl_gtest_obj> ${PROTO_HDRS})
  target_link_libraries(cnnl_gtest_static -Wl,--whole-archive cnnl_static -Wl,--no-whole-archive)
  set(targets_cnnl_gtest ${targets_cnnl_gtest} cnnl_gtest_static)
endif(${CNNL_BUILD_STATIC})


# resolve dependencies of static libraries, maybe use pkgconfig for find_package with hint path is a better idea
set (EXTRA_LIBS "")
foreach(_lib ${LIBXML2_LIBRARIES} ${PROTOBUF_LIBRARIES})
  message(STATUS "resolve lib: ${_lib}")
  execute_process(
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/util_resolve_static.bash ${_lib} ${TOOLCHAIN_ROOT}/${TARGET_CPU_ARCH}/libc/usr/lib
    COMMAND grep "EXTRA_LIBS="
    COMMAND cut -d= -f2
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE EXTRA_LIB
  )
  if (EXTRA_LIB)
    list(APPEND EXTRA_LIBS ${EXTRA_LIB})
  endif()
endforeach()


message(STATUS "EXTRA_LIBS: ${EXTRA_LIBS}")


foreach(target_gtest ${targets_cnnl_gtest})
  target_link_libraries(${target_gtest} -Wl,-whole-archive cnnl_gtest_kernels -Wl,-no-whole-archive)
  target_link_libraries(${target_gtest} -Wl,--no-as-needed cnbin -Wl,--as-needed)
  target_link_libraries(${target_gtest} cnrt cndev cndrv pthread gtest_shared stdc++ m dl)
  target_link_libraries(${target_gtest} ${LIBXML2_LIBRARIES} ${PROTOBUF_LIBRARIES} ${EXTRA_LIBS})
  target_link_libraries(${target_gtest} cnnl_test_proto)
  set_target_properties(${target_gtest}
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../$LIB;../../lib${LIB_SUFFIX}"
  )
endforeach()


# pb to prototxt tool
add_executable(pb2prototxt ${CMAKE_CURRENT_SOURCE_DIR}/tools/pb2prototxt.cpp)
add_executable(prototxt2pb ${CMAKE_CURRENT_SOURCE_DIR}/tools/prototxt2pb.cpp)
target_link_libraries(pb2prototxt ${PROTOBUF_LIBRARIES} cnnl_test_proto)
target_link_libraries(prototxt2pb ${PROTOBUF_LIBRARIES} cnnl_test_proto)
set_target_properties(pb2prototxt prototxt2pb
  PROPERTIES
  INSTALL_RPATH "$ORIGIN/../../$LIB;../../lib${LIB_SUFFIX}"
)
### TODO install `patchelf` and test RUNPATH under CentOS 7
### NOTE
### ref: `man ld.so` and cmake rpath handling
### use RUNPATH instead of RPATH may be better
### ld.so has the following search order: RPATH, LD_LIBRARY_PATH, RUNPATH, ld.so.cache, system
### However, RUNPATH may not support under older system or linker
### And cnnl_gtest does not need libcndrv.so directly, so RUNPATH may not work as expected
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags")


if (NOT CMAKE_INSTALL_MESSAGE)
  set(CMAKE_INSTALL_MESSAGE NEVER) # LAZY: do not show `Up-to-date` info
endif()


install(TARGETS ${targets_cnnl_gtest}
  COMPONENT cnnl_gtest
  RUNTIME DESTINATION build/test
  ARCHIVE DESTINATION lib${LIB_SUFFIX}
  LIBRARY DESTINATION lib${LIB_SUFFIX}
)


install(TARGETS ${targets_cnnl_api_gtest}
  COMPONENT cnnl_gtest
  RUNTIME DESTINATION build/test
  ARCHIVE DESTINATION lib${LIB_SUFFIX}
  LIBRARY DESTINATION lib${LIB_SUFFIX}
)


install(TARGETS pb2prototxt prototxt2pb cnnl_test_proto gtest_shared
  COMPONENT cnnl_gtest
  RUNTIME DESTINATION build/test
  ARCHIVE DESTINATION lib${LIB_SUFFIX}
  LIBRARY DESTINATION lib${LIB_SUFFIX}
)


install(FILES "${CMAKE_CURRENT_LIST_DIR}/gtest_config/test_list"
  COMPONENT cnnl_gtest
  DESTINATION test/cnnl_gtest/gtest_config
)


install(PROGRAMS "${CMAKE_CURRENT_LIST_DIR}/tools/search_case.py"
  "${CMAKE_CURRENT_LIST_DIR}/tools/search_case_v2.py"
  "${CMAKE_CURRENT_LIST_DIR}/tools/watch_mem.bash"
  COMPONENT cnnl_gtest
  DESTINATION test/cnnl_gtest/tools
)


# TODO(jiaminghao) consider support filter by CNNL_BUILD_SPECIFIC_OP instead of copying all cases
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/src/zoo"
  COMPONENT cnnl_gtest
  DESTINATION test/cnnl_gtest/src
  FILES_MATCHING PATTERN "*.prototxt"
)
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/tests"
  COMPONENT cnnl_gtest
  DESTINATION test/cnnl_gtest/
  FILES_MATCHING PATTERN "*.prototxt"
)

