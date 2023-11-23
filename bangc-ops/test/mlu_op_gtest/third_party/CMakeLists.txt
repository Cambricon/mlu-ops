set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=maybe-uninitialized -Wno-error=unused-variable")
set(INSTALL_PPROF TRUE)
set(BUILD_TESTING OFF)
option(BUILD_TESTING "" OFF)
set(gperftools_build_benchmark OFF)
option(gperftools_build_benchmark "" OFF)

set(gperftools_enable_frame_pointers ON)
option(gperftools_enable_frame_pointers "" ON)

string(REPLACE "-fsanitize=address" "" CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fsanitize=address" "" CMAKE_C_FLAGS  "${CMAKE_C_FLAGS}")

string(REPLACE "-O3" "" CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-O3" "" CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE}")

string(REPLACE "-fno-omit-frame-pointer" "" CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fno-omit-frame-pointer" "" CMAKE_C_FLAGS  "${CMAKE_C_FLAGS}")

# These flags are used by ubuntu dpkg-buildpackage process, especially linker flags 'Bsymbolic-functions' could avoid potential runtime bug when use along with ASan or other third libraries
#set(CMAKE_C_FLAGS "-fstack-protector-strong -fno-strict-aliasing ${CMAKE_C_FLAGS} -O2")
#set(CMAKE_CXX_FLAGS "-fstack-protector-strong -fno-strict-aliasing ${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-as-needed")

if (MLUOP_GTEST_BUILD_GPERFTOOLS_ALL)
add_subdirectory(gperftools)
else ()
add_subdirectory(gperftools EXCLUDE_FROM_ALL)
endif ()
