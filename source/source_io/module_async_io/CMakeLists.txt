# ==============================================================================
# CMakeLists.txt — module_async_io
#
# 异步 I/O 管理器模块
# 提供双缓冲 + 独立工作线程的异步文件写入/读取能力，
# 使电荷密度 I/O 与主计算线程重叠执行。
# ==============================================================================

add_library(
    async_io
    OBJECT
    async_io_manager.cpp
)

target_include_directories(
    async_io
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(
    async_io
    PUBLIC
    Threads::Threads
)

# ---- 单元测试 ----
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
