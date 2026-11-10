# -----------------------------------------------------------------------------
# 新日志引擎（spdlog 后端）构建配置。
# -----------------------------------------------------------------------------

# 核心库只链接 fmt 的 header-only 目标 fmt::fmt-header-only
# spdlog 走SPDLOG_FMT_EXTERNAL_HO。
set(SPDLOG_FMT_EXTERNAL_HO ON CACHE BOOL "Use external fmt header-only" FORCE)

# 只构建静态库；关闭示例 / 测试 / 基准 / 安装，避免引入多余目标。
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS_HO OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)

if(NOT TARGET spdlog::spdlog)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/spdlog
                   ${CMAKE_BINARY_DIR}/spdlog)
endif()
