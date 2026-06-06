# ==============================================================================
# CMakeLists.txt — module_async_io/test
#
# 单元测试: 异步 I/O 管理器 + 异步读取 (RhogReadTask)
# ==============================================================================

AddTest(
    TARGET test_async_io
    LIBS async_io
    SOURCES test_async_io.cpp
)

AddTest(
    TARGET test_async_read
    LIBS async_io
    SOURCES test_async_read.cpp
)

