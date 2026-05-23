# ==============================================================================
# CMakeLists.txt — module_async_io/test
# ==============================================================================

AddTest(
    TARGET MODULE_IO_async_io
    LIBS async_io Threads::Threads
    SOURCES test_async_io.cpp
            ../../../source_base/test/tool_quit_no_exit.cpp
            ../../../source_base/module_device/device.cpp
            ../../../source_base/module_device/output_device.cpp
)
