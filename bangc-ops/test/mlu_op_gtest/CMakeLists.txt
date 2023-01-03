cmake_minimum_required(VERSION 3.3)
cmake_policy(SET CMP0048 NEW) # Use project(... VERSION ...)
project(mlu_op_test)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# check
message("[MLUOP GTEST STYLE]: ")
set(MLUOP_BUILD_SPECIFIC_OP ${MLUOP_BUILD_SPECIFIC_OP})

execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/gtest_style.py ${MLUOP_BUILD_SPECIFIC_OP} RESULT_VARIABLE STYLE_CHECK)
if (STYLE_CHECK)
  message(FATAL_ERROR "-- Pleace check gtest code manually.")
endif()

message("[MLUOP GTEST REGISTER OP]: ")
file(GLOB CASE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/gtest/*case_list.cpp")
execute_process(COMMAND rm -f ${CASE_LIST})
execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/gtest/register_op.py ${MLUOP_BUILD_SPECIFIC_OP})


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
set(CMAKE_FIND_LIBRARY_SUFFIXES ${ORIG_FIND_LIBRARY_SUFFIXES})

message(STATUS "libxml2 include: " ${LIBXML2_INCLUDE_DIRS})
message(STATUS "libxml2 library: " ${LIBXML2_LIBRARIES})
message(STATUS "protobuf include: " ${PROTOBUF_INCLUDE_DIRS})
message(STATUS "protobuf libraries: " ${PROTOBUF_LIBRARIES})
message(STATUS "protoc executable: " ${PROTOBUF_PROTOC_EXECUTABLE})

include_directories(${LIBXML2_INCLUDE_DIRS})
include_directories(${PROTOBUF_INCLUDE_DIR})


if(${MLUOP_TARGET_CPU_ARCH} MATCHES "aarch64")
  # set(DEFAULT_PROTOBUF_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aarch64/protobuf-2.6.1/include/")
  # set(DEFAULT_PROTOBUF_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aarch64/protobuf-2.6.1/lib/")
  # set(DEFAULT_LIBXML2_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aarch64/libxml2-2.7.4/include/libxml2")
  # set(DEFAULT_LIBXML2_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/aarch64/libxml2-2.7.4/lib/")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

message("[MLUOP GTEST PROTOC]: ")
set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/mlu_op_test_proto")
file(GLOB PROTO_FILES "${PROTO_PATH}/mlu_op_test.proto")

# protoc code
set(PROTOBUF_PROTOC_EXECUTABLE protoc CACHE STRING "protoc command location")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
message(${PROTO_SRCS} "-----------" ${PROTO_FILES})
add_custom_target(mluop_build_proto DEPENDS ${PROTO_SRCS} ${PROTO_HDRS} ${PROTO_FILES})
include_directories("${CMAKE_CURRENT_BINARY_DIR}")
add_library(mluop_test_proto ${PROTO_SRCS} ${PROTO_HDRS})
add_dependencies(mluop_test_proto mluop_build_proto)

set(MLUOP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../" CACHE PATH "Directory for mlu_op.h")
find_path(LIBMLUOP_INCLUDE_DIR mlu_op.h
  HINTS
  ${MLUOP_DIR}
  REQUIRED
)
message(STATUS "Found mlu_op.h under ${LIBMLUOP_INCLUDE_DIR}")
include_directories("${LIBMLUOP_INCLUDE_DIR}")

set(GOOGLETEST_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/googletest"
                       "${CMAKE_CURRENT_SOURCE_DIR}/googletest/include")
set(GOOGLETEST_SRC     "${CMAKE_CURRENT_SOURCE_DIR}/googletest/src")
set(MLUOP_GTEST_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(MLUOP_PB_GTEST_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/include"
                        "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/tests"
                        "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/gtest"
                        "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/zoo"
                        "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/gtest/extra_lib/jsoncppDist/json"
                        "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src"
)
set(MLUOP_PB_GTEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/*.cpp"
                    "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/gtest/*.cpp"
                    "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/gtest/extra_lib/jsoncppDist/*.cpp"
)
set(MLUOP_API_GTEST_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/api_gtest/include"
                        "${CMAKE_CURRENT_SOURCE_DIR}/api_gtest/src/gtest"
                        "${CMAKE_CURRENT_SOURCE_DIR}/api_gtest/src/gtest/extra_lib/jsoncppDist/json"
)
set(MLUOP_API_GTEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/api_gtest/src/*.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/api_gtest/src/gtest/*.cpp"
                        "${CMAKE_CURRENT_SOURCE_DIR}/api_gtest/src/gtest/extra_lib/jsoncppDist/*.cpp"
)

include_directories("${GOOGLETEST_INCLUDE}")
include_directories("${MLUOP_GTEST_INCLUDE}")
include_directories("${PROTO_PATH}")
include_directories("${MLUOP_PB_GTEST_INCLUDE}")
include_directories("${MLUOP_API_GTEST_INCLUDE}")

file(GLOB_RECURSE gtest_mlu_files FOLLOW_SYMLINKS "${CMAKE_CURRENT_LIST_DIR}/pb_gtest/src/internal_kernel/*")
bang_add_library(mluop_gtest_kernels STATIC "${gtest_mlu_files}")

file(GLOB SRC_DIR "${GOOGLETEST_SRC}/*.cc")
#add_library(gtest_shared SHARED ${SRC_DIR})
add_library(gtest_shared STATIC ${SRC_DIR}) # for runtime convenience

file(GLOB MLUOP_PB_TEST_DIR ${MLUOP_PB_GTEST_SRC})

if (MLUOP_BUILD_SPECIFIC_OP)
  foreach(op ${MLUOP_BUILD_SPECIFIC_OP})
    file(GLOB_RECURSE OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/zoo/${op}/*.cpp")
    list(APPEND MLUOP_PB_GTEST_DIR ${OP_DIR})
  endforeach()
else()
  file(GLOB_RECURSE MLUOP_PB_GTEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pb_gtest/src/zoo/*/*.cpp")
endif()

add_library(mluop_pb_gtest_obj OBJECT ${MLUOP_PB_TEST_DIR} ${MLUOP_PB_GTEST_DIR} ${PROTO_HDRS})
add_dependencies(mluop_pb_gtest_obj mluop_build_proto)
add_executable(mluop_gtest $<TARGET_OBJECTS:mluop_pb_gtest_obj>)
target_link_libraries(mluop_gtest mluops)
set(targets_mluop_gtest mluop_gtest)

message(STATUS "EXTRA_LIBS: ${EXTRA_LIBS}")
foreach(target_gtest ${targets_mluop_gtest})
  target_link_libraries(${target_gtest} mluop_gtest_kernels)
  target_link_libraries(${target_gtest} -Wl,--no-as-needed cnbin -Wl,--as-needed)
  target_link_libraries(${target_gtest} cnrt cndev cndrv pthread gtest_shared stdc++ m dl)
  target_link_libraries(${target_gtest} ${LIBXML2_LIBRARIES} ${PROTOBUF_LIBRARIES} ${EXTRA_LIBS})
  target_link_libraries(${target_gtest} mluop_test_proto)
  set_target_properties(${target_gtest}
    PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../$LIB;../../lib${LIB_SUFFIX}"
  )
endforeach()

file(GLOB MLUOP_API_TEST_DIR ${MLUOP_API_GTEST_SRC})

if (MLUOP_BUILD_SPECIFIC_OP)
  foreach(op ${MLUOP_BUILD_SPECIFIC_OP})
    file(GLOB_RECURSE OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/api_gtest/src/gtest/${op}/*.cpp")
    list(APPEND MLUOP_API_GTEST_DIR ${OP_DIR})
  endforeach()
else()
  file(GLOB_RECURSE MLUOP_API_GTEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/api_gtest/src/gtest/*/*.cpp")
endif()

add_library(mluop_api_gtest_obj OBJECT ${MLUOP_API_TEST_DIR} ${MLUOP_API_GTEST_DIR})
add_executable(mluop_api_gtest $<TARGET_OBJECTS:mluop_api_gtest_obj>)
target_link_libraries(mluop_api_gtest cnrt cndev cndrv cndev pthread gtest_shared mluops stdc++ m dl)
target_link_libraries(mluop_api_gtest ${LIBXML2_LIBRARIES} ${EXTRA_LIBS})
set_target_properties(mluop_api_gtest
  PROPERTIES
  INSTALL_RPATH "$ORIGIN/../../$LIB;../../lib${LIB_SUFFIX}"
)
set(targets_mluop_api_gtest mluop_api_gtest)

# pb to prototxt tool
add_executable(pb2prototxt ${CMAKE_CURRENT_SOURCE_DIR}/tools/pb2prototxt.cpp)
add_executable(prototxt2pb ${CMAKE_CURRENT_SOURCE_DIR}/tools/prototxt2pb.cpp)
target_link_libraries(pb2prototxt ${PROTOBUF_LIBRARIES} mluop_test_proto)
target_link_libraries(prototxt2pb ${PROTOBUF_LIBRARIES} mluop_test_proto)
set_target_properties(pb2prototxt prototxt2pb
  PROPERTIES
  INSTALL_RPATH "$ORIGIN/../../$LIB;../../lib${LIB_SUFFIX}"
)
if (NOT CMAKE_INSTALL_MESSAGE)
  set(CMAKE_INSTALL_MESSAGE NEVER) # LAZY: do not show `Up-to-date` info
endif()
install(TARGETS ${targets_mluop_gtest}
  COMPONENT mluop_gtest
  RUNTIME DESTINATION build/test
  ARCHIVE DESTINATION lib${LIB_SUFFIX}
  LIBRARY DESTINATION lib${LIB_SUFFIX}
)

install(TARGETS ${targets_mluop_api_gtest}
  COMPONENT mluop_gtest
  RUNTIME DESTINATION build/test
  ARCHIVE DESTINATION lib${LIB_SUFFIX}
  LIBRARY DESTINATION lib${LIB_SUFFIX}
)

install(TARGETS pb2prototxt prototxt2pb mluop_test_proto gtest_shared
  COMPONENT mluop_gtest
  RUNTIME DESTINATION build/test
  ARCHIVE DESTINATION lib${LIB_SUFFIX}
  LIBRARY DESTINATION lib${LIB_SUFFIX}
)

install(FILES "${CMAKE_CURRENT_LIST_DIR}/pb_gtest/gtest_config/test_list"
  COMPONENT mluop_gtest
  DESTINATION test/mlu_op_gtest/pb_gtest/gtest_config
)
# install(PROGRAMS "${CMAKE_CURRENT_LIST_DIR}/tools/search_case.py"
#   "${CMAKE_CURRENT_LIST_DIR}/tools/search_case_v2.py"
#   "${CMAKE_CURRENT_LIST_DIR}/tools/watch_mem.bash"
#   COMPONENT mluop_gtest
#   DESTINATION test/mluop_gtest/tools
# )

install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/pb_gtest/src/zoo"
  COMPONENT mluop_gtest
  DESTINATION test/mluop_gtest/pb_gtest/src
  FILES_MATCHING PATTERN "*.prototxt"
)
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/pb_gtest/tests"
  COMPONENT mluop_gtest
  DESTINATION test/mluop_gtest/pb_gtest
  FILES_MATCHING PATTERN "*.prototxt"
)
