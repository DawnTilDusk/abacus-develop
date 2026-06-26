# ==============================================================================
# CMakeLists.txt — module_async_io/omp_task
#
# OpenMP 4.0 task+depend 异步 I/O (header-only)
# 当前仅包含 header, 不需要编译 .cpp
# ==============================================================================

add_library(omp_task_async INTERFACE)

target_include_directories(
    omp_task_async
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

find_package(OpenMP REQUIRED)
target_link_libraries(omp_task_async INTERFACE OpenMP::OpenMP_CXX)

# ---- 单元测试 ----
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
