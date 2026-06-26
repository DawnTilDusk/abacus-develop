# ==============================================================================
# CMakeLists.txt — module_async_io/omp_task/test
#
# OpenMP 4.0 task+depend 异步 I/O 管理器单元测试
# ==============================================================================

AddTest(
    TARGET test_omp_task
    LIBS omp_task_async
    SOURCES test_omp_task.cpp
)
