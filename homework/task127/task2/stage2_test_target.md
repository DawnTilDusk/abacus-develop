# 阶段 2：新建独立 OpenMP 测试 target

## 1. 目的

题目一的测试目录 `source/source_estate/test_mpi/` 在 [test_mpi/CMakeLists.txt:7](file:///root/homework/abacus-develop/source/source_estate/test_mpi/CMakeLists.txt#L7) 显式 `remove_definitions(-D_OPENMP)`，使 OpenMP 宏在该 target 中失效。

题目二需要让 `#pragma omp` 真正生效，又不能破坏题目一现有测试。本阶段的做法：

- **保留** `test_mpi/` 的 `remove_definitions(-D_OPENMP)` 不动；
- **新建** 同级目录 `source/source_estate/test_omp/`，在那里**不**移除 `_OPENMP`；
- 在 `source/source_estate/CMakeLists.txt` 里以 `USE_OPENMP` 为条件，把新子目录挂上。

## 2. 改动清单（仅 3 处，全部为新增）

| 操作 | 路径 | 说明 |
|---|---|---|
| 修改 | `source/source_estate/CMakeLists.txt` | 在 `BUILD_TESTING / ENABLE_MPI` 内，追加 `if(USE_OPENMP) add_subdirectory(test_omp) endif()` |
| 新增 | `source/source_estate/test_omp/CMakeLists.txt` | 注册 `MODULE_ESTATE_charge_omp_test` target + 4np ctest 入口 |
| 新增 | `source/source_estate/test_omp/charge_omp_test.cpp` | 空壳测试：GTest + MPI 入口 + 一个 `_OPENMP` 宏是否生效的健全性测试 |

题目一目录下**没有任何修改**。

## 3. 关键代码片段

### 3.1 `source/source_estate/CMakeLists.txt`

```cmake
if(BUILD_TESTING)
  if(ENABLE_MPI)
    add_subdirectory(test)
    add_subdirectory(test_mpi)
    if(USE_OPENMP)
      add_subdirectory(test_omp)
    endif()
  endif()
endif()
```

`USE_OPENMP` 是 `abacus-develop` 顶层 `CMakeLists.txt` 已经定义的开关（`option(USE_OPENMP "Enable OpenMP" ON)`），同时 `cmake/Testing.cmake` 里的 `AddTest()` 函数在 `USE_OPENMP=ON` 时会自动 `target_link_libraries(... OpenMP::OpenMP_CXX)`，所以新 target 不需要任何额外的 OpenMP 链接配置。

### 3.2 `source/source_estate/test_omp/CMakeLists.txt`

与 `test_mpi/CMakeLists.txt` 大体相同，唯一关键差异是：

- **删除了** `remove_definitions(-D_OPENMP)`；
- target 名改为 `MODULE_ESTATE_charge_omp_test`，避免与题目一 target 冲突。

```cmake
remove_definitions(-D__EXX)
remove_definitions(-D__CUDA)
remove_definitions(-D__UT_USE_CUDA)
remove_definitions(-D__UT_USE_ROCM)
remove_definitions(-D__ROCM)
remove_definitions(-D__MLALGO)
# 注意：故意不要 remove_definitions(-D_OPENMP)，让 #pragma omp 生效

AddTest(
  TARGET MODULE_ESTATE_charge_omp_test
  LIBS parameter ${math_libs} psi base device planewave
  SOURCES charge_omp_test.cpp ../module_charge/charge_mpi.cpp
)

add_test(NAME MODULE_ESTATE_charge_omp_test_4np
      COMMAND mpirun -np 4 ./MODULE_ESTATE_charge_omp_test;
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
```

> 与 `test_mpi` 一样会编进 `../module_charge/charge_mpi.cpp`。也就是说，题目一文件即将在阶段 4/5 加上的 `#pragma omp parallel for`，在题目二的 target 里会被启用，在题目一的 target 里会被忽略（无 `_OPENMP` 宏，整段 `#ifdef _OPENMP` 失效）。

### 3.3 `source/source_estate/test_omp/charge_omp_test.cpp`（空壳）

阶段 2 只放一个用于验证 `_OPENMP` 宏生效的最小 GTest 用例，真正的串行 vs 并行对比留到阶段 6。

```cpp
TEST(ChargeOmpSkeleton, OpenMPMacroEnabled)
{
#ifdef _OPENMP
    SUCCEED() << "_OPENMP is defined; omp_get_max_threads = "
              << omp_get_max_threads();
#else
    FAIL() << "_OPENMP is NOT defined for MODULE_ESTATE_charge_omp_test. "
              "Check source/source_estate/test_omp/CMakeLists.txt.";
#endif
}
```

`main()` 里只有 `MPI_Init` + `RUN_ALL_TESTS()` + `MPI_Finalize()`，结构与题目一的 `charge_mpi_test.cpp` 完全对齐。

## 4. 实际编译与运行结果

### 4.1 重新 configure

```bash
cd /root/homework/abacus-develop/build
cmake ..
```

输出末尾：

```
-- Configuring done
-- Generating done
-- Build files have been written to: /root/homework/abacus-develop/build
```

新 `add_subdirectory(test_omp)` 已被识别。

### 4.2 构建新 target

```bash
cmake --build . --target MODULE_ESTATE_charge_omp_test -j 4
```

输出末尾：

```
[100%] Building CXX object source/source_estate/test_omp/CMakeFiles/MODULE_ESTATE_charge_omp_test.dir/__/module_charge/charge_mpi.cpp.o
[100%] Building CXX object source/source_estate/test_omp/CMakeFiles/MODULE_ESTATE_charge_omp_test.dir/charge_omp_test.cpp.o
[100%] Linking CXX executable MODULE_ESTATE_charge_omp_test
[100%] Built target MODULE_ESTATE_charge_omp_test
```

### 4.3 跑空壳测试

```bash
cd build/source/source_estate/test_omp
OMP_NUM_THREADS=4 ./MODULE_ESTATE_charge_omp_test
```

输出：

```
[==========] Running 1 test from 1 test suite.
[ RUN      ] ChargeOmpSkeleton.OpenMPMacroEnabled
[       OK ] ChargeOmpSkeleton.OpenMPMacroEnabled (0 ms)
[----------] 1 test from ChargeOmpSkeleton (0 ms total)
[  PASSED  ] 1 test.
```

`_OPENMP` 宏生效，否则会走 `FAIL()` 分支。

### 4.4 回归验证题目一原 target

```bash
cmake --build . --target MODULE_ESTATE_charge_mpi_test -j 4
mpirun -np 4 ./build/source/source_estate/test_mpi/MODULE_ESTATE_charge_mpi_test
```

输出：

```
[  PASSED  ] 4 tests.   (× 4 个 rank)
```

题目一两个 target 在阶段 2 之后**完全未退化**。

## 5. 阶段 2 完成判定

| 检查项 | 结果 |
|---|---|
| `test_mpi/CMakeLists.txt` 是否动过 | 否 ✅ |
| `test_mpi/charge_mpi_test.cpp` 是否动过 | 否 ✅ |
| 题目一 target 是否仍能通过 | 通过 ✅ |
| 题目二 target 是否能编译 | 通过 ✅ |
| 题目二 target 内 `_OPENMP` 是否生效 | 生效 ✅ |
| 题目二 4np ctest 入口是否注册 | 已注册 ✅ |

阶段 2 通过 ✅ —— 可以进入阶段 3。

## 6. 后续阶段的入口

阶段 4/5 加 `#pragma omp parallel for` 时，会用 `#ifdef _OPENMP` 包起来。届时：

- 编进 `MODULE_ESTATE_charge_mpi_test`（无 `_OPENMP`） → pragma 不生效，行为同基线，题目一仍过；
- 编进 `MODULE_ESTATE_charge_omp_test`（有 `_OPENMP`） → pragma 生效，进入真正的并行路径。

阶段 6 起，所有题目二相关的串并行一致性、边界条件、性能测试都会写到 `test_omp/charge_omp_test.cpp` 里，不会扩散到题目一。
