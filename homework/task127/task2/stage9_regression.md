# 阶段 9：四种 OpenMP × MPI 组合最终回归

## 1. 目的

题目二落地完毕之后，必须验证：

- 题目一原 target（`-D_OPENMP` 被 `remove_definitions` 抹掉）行为完全未变；
- 题目二新 target（保留 `-D_OPENMP`）单进程 / 多进程都正确；
- 混合 MPI + OpenMP 运行不破坏任何东西。

阶段 9 通过 4 种正交组合的矩阵验证一次性确认这一切。

## 2. 验证矩阵

| 组合 | OpenMP | MPI | 跑的二进制 | 期望 |
|---|---|---|---|---|
| **A** | OFF（题目一编译开关） | `np=1` | `MODULE_ESTATE_charge_mpi_test` | 全过（题目一基线） |
| **B** | OFF | `mpirun -np 4` | `MODULE_ESTATE_charge_mpi_test` | 全过（题目一 4 进程） |
| **C** | ON | `np=1`，`OMP_NUM_THREADS=4` | `MODULE_ESTATE_charge_omp_test` | 全过 + perf SKIPPED |
| **D** | ON，`OMP_NUM_THREADS=2` | `mpirun -np 4` | `MODULE_ESTATE_charge_omp_test` | 全过 × 4 ranks + perf SKIPPED × 4 |

组合 A、B 直接复用题目一原 target，从根本上证明「题目二的 pragma 改动**不会**在 OpenMP 关闭的编译路径上引入任何回归」；组合 C、D 跑题目二自己的 target，验证 OpenMP 真正生效时所有 9 个单元测试都通过，并且在与 MPI 同时使用（4 ranks × 2 threads = 8 个并发执行流）的情况下也无 race / 死锁。

## 3. 执行命令

```bash
cd /root/homework/abacus-develop/build/source/source_estate

# A: OpenMP OFF + np=1
./test_mpi/MODULE_ESTATE_charge_mpi_test

# B: OpenMP OFF + np=4
mpirun -np 4 ./test_mpi/MODULE_ESTATE_charge_mpi_test

# C: OpenMP ON + np=1 + threads=4
OMP_NUM_THREADS=4 ./test_omp/MODULE_ESTATE_charge_omp_test

# D: OpenMP ON + np=4 + threads=2
OMP_NUM_THREADS=2 mpirun -np 4 ./test_omp/MODULE_ESTATE_charge_omp_test
```

## 4. 实测结果

### 4.1 组合 A：OpenMP OFF + np=1

```
[ RUN      ] ChargeMpiTest.reduce_diff_pools1   [       OK ]
[ RUN      ] ChargeMpiTest.reduce_diff_pools2   [       OK ]
[ RUN      ] ChargeMpiTest.rho_mpi              [       OK ]
[ RUN      ] ChargeMpiTest.kin_r_mpi            [       OK ]
[  PASSED  ] 4 tests.
```

→ 题目一基线未退化。

### 4.2 组合 B：OpenMP OFF + np=4

```
[==========] 4 tests from 1 test suite ran. (72 ms total)
[  PASSED  ] 4 tests.
... (× 4 ranks)
```

→ 题目一 4 进程并行未退化，MPI 通信链路（题目一阶段四的 `MPI_Waitsome` 非阻塞收集）依然正确。

### 4.3 组合 C：OpenMP ON + np=1 + threads=4

```
[==========] 10 tests from 5 test suites ran. (4 ms total)
[  PASSED  ] 9 tests.
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] ChargeOmpPerf.ReorderAndExtractSpeedup
```

→ 9 个单元测试全过（含 `ChargeOmpReorder.*`、`ChargeOmpExtract.*`、`ChargeOmpBoundary.*`、`ChargeOmpSkeleton.*`），1 个 perf benchmark 默认被 `CHARGE_OMP_BENCH=1` gating 跳过。

### 4.4 组合 D：OpenMP ON + np=4 + threads=2（MPI + OpenMP 混合）

```
      4 [  PASSED  ] 9 tests.
      4 [  SKIPPED ] 1 test, listed below:
      4 [  SKIPPED ] ChargeOmpPerf.ReorderAndExtractSpeedup
      4 [  SKIPPED ] ChargeOmpPerf.ReorderAndExtractSpeedup (0 ms)
```

→ 4 个 MPI rank 上每一份都报告「9 PASSED + 1 SKIPPED」，对应 `uniq -c` 计数全是 4。MPI + OpenMP 混合下没有死锁、没有 race、没有结果不一致。

## 5. 阶段 9 完成判定

| 检查项 | 结果 |
|---|---|
| 组合 A 全过（题目一 OpenMP OFF + np=1） | ✅ |
| 组合 B 全过（题目一 OpenMP OFF + np=4） | ✅ |
| 组合 C 全过（题目二 OpenMP ON + np=1） | ✅ |
| 组合 D 全过（MPI + OpenMP 混合） | ✅ |
| perf benchmark 默认 SKIPPED 行为一致 | ✅ |
| 题目一 4 个原测试用例数没有任何 FAILED | ✅ |
| 题目二 9 个用例 + 1 perf SKIPPED 数没有任何 FAILED | ✅ |

阶段 9 通过 ✅ —— **题目二的所有可交付内容均已锁定。**

## 6. 题目二完整交付总览

### 6.1 改动的生产代码（仅 1 文件 2 函数）

[source/source_estate/module_charge/charge_mpi.cpp](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp)：

- `Charge::reorder_pool_rank_to_uniform`（阶段 4）—— 函数体内加 `#pragma omp parallel for collapse(2) schedule(static)` 含 `#ifdef _OPENMP` 保护，提前栈上常量 `numz_ip / startz_ip / nz`
- `Charge::extract_uniform_to_local`（阶段 5）—— 同样形式

### 6.2 新增的测试基础设施

- [source/source_estate/CMakeLists.txt](file:///root/homework/abacus-develop/source/source_estate/CMakeLists.txt) +4 行：条件挂载 `test_omp` 子目录
- [source/source_estate/test_omp/CMakeLists.txt](file:///root/homework/abacus-develop/source/source_estate/test_omp/CMakeLists.txt)：新 target `MODULE_ESTATE_charge_omp_test` + ctest 入口 `MODULE_ESTATE_charge_omp_test_4np`
- [source/source_estate/test_omp/charge_omp_test.cpp](file:///root/homework/abacus-develop/source/source_estate/test_omp/charge_omp_test.cpp)：10 个 GTest 用例
  - 1 个 `_OPENMP` 健全性测试
  - 2 个 reorder 串并行一致性测试（均匀 / 不均匀切分）
  - 1 个 extract 多 rank 串并行一致性测试
  - 5 个边界测试（`ncxy=1` × 2、`numz=[0,1,...]`、单 rank pool、线程数不变性）
  - 1 个 perf benchmark（默认 `GTEST_SKIP`，`CHARGE_OMP_BENCH=1` 启用）

### 6.3 文档（题目二报告）

| 阶段 | 文档 |
|---|---|
| 1 | [stage1_baseline.md](./stage1_baseline.md) |
| 2 | [stage2_test_target.md](./stage2_test_target.md) |
| 3 | [stage3_dependency_analysis.md](./stage3_dependency_analysis.md) |
| 4 | [stage4_reorder_openmp.md](./stage4_reorder_openmp.md) |
| 5 | [stage5_extract_openmp.md](./stage5_extract_openmp.md) |
| 6 | [stage6_boundary_tests.md](./stage6_boundary_tests.md) |
| 7 | [stage7_performance.md](./stage7_performance.md) |
| 9 | [stage9_regression.md](./stage9_regression.md)（本文件） |
| 索引 | [README.md](./README.md) |

题目要求的各项对应关系：

| 题目要求 | 完成阶段 |
|---|---|
| 1. 循环依赖分析 | 阶段 3 |
| 2. OpenMP 实现 | 阶段 4 + 5 |
| 3. 性能对比 / Amdahl 分析 | 阶段 7 |
| 4. 正确性验证 / 线程安全检查 | 阶段 4 + 5 + 6 + 9 |
| 5. 单元测试（边界 / 可重复生成器） | 阶段 4 + 5 + 6 |
| 6. 代码重构（DataTransformFunc 抽象，加分项） | 未做（可选） |

### 6.4 题目一回归保障

| 维度 | 状态 |
|---|---|
| 题目一目录 `test_mpi/` | 完全未动 |
| 题目一 target `MODULE_ESTATE_charge_mpi_test` | 全过（np=1、np=4） |
| 题目一阶段四 `MPI_Waitsome` 路径 | 不受影响（函数签名 / 语义未改变） |
| 题目一编译宏 `-D_OPENMP` 被 `remove_definitions` | 仍然保留 |
| 在题目一 target 下 `#pragma omp` | 被预处理删除，行为同基线 |

## 7. 最终交付状态

- ✅ 阶段 1 ~ 7 + 9 全部通过
- ⏸ 阶段 8（加分项 `DataTransformFunc` 抽象）跳过 —— 不影响主线交付，可作为后续独立扩展
- 总测试覆盖：题目一 4 用例 + 题目二 9 用例 + 1 perf benchmark（按需启用）
- 题目二与题目一可以同时部署、互不破坏。
