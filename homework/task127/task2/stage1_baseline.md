# 阶段 1：基线锁定

## 1. 目的

在动手做题目二（OpenMP 并行化）之前，先固定题目一现有实现的「正确性 + 性能」基线，作为后续所有阶段的对照参考。

阶段 1 的判定标准是：**不改任何代码、不改任何 CMake**，直接复用题目一已经编译好的二进制，确认测试全过并记录耗时。

## 2. 测试目标

题目一阶段四的产物位于：

- 源码：`source/source_estate/module_charge/charge_mpi.cpp`
- 头文件：`source/source_estate/module_charge/charge.h`
- 单元测试：`source/source_estate/test_mpi/charge_mpi_test.cpp`
- 测试 CMake：`source/source_estate/test_mpi/CMakeLists.txt`

题目一的测试 target：

- `MODULE_ESTATE_charge_mpi_test`（np=1）
- `MODULE_ESTATE_charge_mpi_test_4np`（np=4，由 ctest 注册）

测试覆盖到的核心函数：

- `Charge::reduce_diff_pools()`
- `Charge::rho_mpi()`
- `Charge::kin_r_mpi()`
- 间接覆盖：`reorder_pool_to_uniform`、`reorder_pool_rank_to_uniform`、`extract_uniform_to_local`、`gather_pool_data_nonblocking`

后两者正是题目二将要 OpenMP 化的目标。

## 3. 执行命令

二进制位置：

```
/root/homework/abacus-develop/build/source/source_estate/test_mpi/MODULE_ESTATE_charge_mpi_test
```

### 3.1 np = 1（直接运行）

```bash
cd /root/homework/abacus-develop/build/source/source_estate/test_mpi
./MODULE_ESTATE_charge_mpi_test
```

### 3.2 np = 4（mpirun，本机使用 MPICH，不需要 `--allow-run-as-root`）

```bash
cd /root/homework/abacus-develop/build/source/source_estate/test_mpi
mpirun -np 4 ./MODULE_ESTATE_charge_mpi_test
```

## 4. 基线结果

### 4.1 np = 1

```
[==========] Running 4 tests from 1 test suite.
[ RUN      ] ChargeMpiTest.reduce_diff_pools1   [       OK ] (0 ms)
[ RUN      ] ChargeMpiTest.reduce_diff_pools2   [       OK ] (0 ms)
[ RUN      ] ChargeMpiTest.rho_mpi              [       OK ] (0 ms)
[ RUN      ] ChargeMpiTest.kin_r_mpi            [       OK ] (0 ms)
[----------] 4 tests from ChargeMpiTest (0 ms total)
[  PASSED  ] 4 tests.
```

- 全部通过：4 / 4
- 单测耗时：~0 ms（小网格，量级太小，单进程下基本看不出可测时间）

### 4.2 np = 4

```
[ RUN      ] ChargeMpiTest.reduce_diff_pools2   [       OK ] (~2-3 ms)
[ RUN      ] ChargeMpiTest.rho_mpi              [       OK ] (~2 ms)
[ RUN      ] ChargeMpiTest.kin_r_mpi            [       OK ] (~2 ms)
[----------] 4 tests from ChargeMpiTest (~53-56 ms total)
[  PASSED  ] 4 tests.
```

- 全部通过：4 / 4（4 个 MPI rank 上各跑一遍 GTest 套件）
- 单 rank 上的 ChargeMpiTest 套件总耗时：53–56 ms（包含 MPI 启动/同步开销）

## 5. 关于「OpenMP 关闭」状态的确认

`source/source_estate/test_mpi/CMakeLists.txt:7` 中显式写了：

```cmake
remove_definitions(-D_OPENMP)
```

这一行使得题目一的测试 target 在编译时不会带上 `_OPENMP` 宏，等价于把所有 `#ifdef _OPENMP` 都视为关闭。**这就是题目二将要参照的「OpenMP 关闭」基线**，所有数据 / 行为都基于这一前提。

题目二阶段 2 会**新建独立测试目录**，在那里把 OpenMP 重新打开，而**不动**这一行，从而：

- 题目一现有测试基线不受任何影响；
- 题目二可以独立衡量 OpenMP 带来的加速效果。

## 6. 基线小结

| 维度 | 状态 |
|---|---|
| 测试正确性 | 全过（np=1、np=4） |
| 题目一阶段四逻辑 | 保留不动 |
| OpenMP 编译宏 | 关闭（`-D_OPENMP` 被 `remove_definitions` 抹掉） |
| 题目二改动范围 | 暂为 0（仅记录） |

阶段 1 通过 ✅ —— 可以进入阶段 2。

## 7. 给后续阶段的注意事项

1. 当前 4 个 GTest 用例的耗时尺度太小（小网格），不能直接用来比较 OpenMP 加速比；**阶段 6 / 阶段 7 必须自建大尺寸输入**，否则性能差异会被噪声淹没。
2. ctest 显示的 `MODULE_ESTATE_charge_mpi_test_4np` 实际是通过 `add_test(NAME ... COMMAND mpirun -np 4 ...)` 注册的，并不需要单独的二进制。题目二同样可以走这条路注册多 rank 测试。
3. 由于本机 MPI 实现是 **MPICH**（错误信息中的 `mpiexec@bohrium...`），所以**不要**用 `mpirun --allow-run-as-root`（那是 OpenMPI 的参数），直接 `mpirun -np N` 即可。
