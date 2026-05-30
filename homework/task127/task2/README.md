# 题目二：执行进度与产物索引

本目录存放题目二（OpenMP 多线程加速数据重排）的全部规划、阶段性报告和测试基线。原始规划文件为 `task2_plan.md`，本 README 记录已完成阶段的产物清单和验收状态。

## 已完成阶段

| 阶段 | 产物 | 状态 |
|---|---|---|
| 阶段 1 基线锁定 | [stage1_baseline.md](./stage1_baseline.md) | ✅ 完成 |
| 阶段 2 OpenMP 测试 target | [stage2_test_target.md](./stage2_test_target.md) | ✅ 完成 |
| 阶段 3 循环依赖分析 | [stage3_dependency_analysis.md](./stage3_dependency_analysis.md) | ✅ 完成 |
| 阶段 4 reorder OpenMP 化 | [stage4_reorder_openmp.md](./stage4_reorder_openmp.md) | ✅ 完成 |
| 阶段 5 extract OpenMP 化 | [stage5_extract_openmp.md](./stage5_extract_openmp.md) | ✅ 完成 |
| 阶段 6 系统化边界测试 | [stage6_boundary_tests.md](./stage6_boundary_tests.md) | ✅ 完成 |

## 阶段 1 ~ 6 一句话总结

1. **阶段 1**：在不改代码的前提下，跑通题目一现有的 `MODULE_ESTATE_charge_mpi_test`（np=1 与 np=4），4/4 测试全过，作为题目二的对照基线，并明确 OpenMP 宏当前被 `remove_definitions` 抹掉的事实。
2. **阶段 2**：新建独立目录 `source/source_estate/test_omp/`，注册 target `MODULE_ESTATE_charge_omp_test` + ctest 入口 `MODULE_ESTATE_charge_omp_test_4np`。**不动**题目一目录任何文件，由 `USE_OPENMP` 条件挂载。最小空壳测试 `ChargeOmpSkeleton.OpenMPMacroEnabled` 验证 `_OPENMP` 宏在新 target 中已启用。题目一原 target 重新编译 + 4np 跑测，仍全过。
3. **阶段 3**：对 `reorder_pool_rank_to_uniform` 与 `extract_uniform_to_local` 做严格的下标依赖分析，证明三层 `ip × ir × iz` 完全无写–写、无 RAW、无 WAR 冲突，并据此选定 OpenMP 策略为 `#pragma omp parallel for collapse(2) schedule(static)`，且加 `#ifdef _OPENMP` 保护以兼容题目一 target。
4. **阶段 4**：在 `reorder_pool_rank_to_uniform` 真正落地 `#pragma omp parallel for collapse(2) schedule(static)`（含 `#ifdef _OPENMP` 保护），并把 `numz[ip] / startz[ip] / nz` 提前到栈上常量满足 `collapse(2)` 的 rectangular 要求。新增两个对比测试 `ChargeOmpReorder.MatchesSerialReferenceUniformSplit / UnevenSplit`，在 `OMP_NUM_THREADS=1 / 4` 下均与手写串行参考逐位一致；题目一原 target 4 进程仍全过。
5. **阶段 5**：对称地把 `extract_uniform_to_local` 也做 OpenMP 化（同样 `collapse(2) schedule(static)` + `#ifdef _OPENMP`），新增 `ChargeOmpExtract.MatchesSerialReferenceForEachRank` 测试模拟 `GlobalV::RANK_IN_POOL` 与 `startz_current`，覆盖 `numz=[3,1,0,4]` 含空 slab 的不均匀切分。`threads=1/4` 全过，题目一原 target 仍全过。
6. **阶段 6**：补齐 `ChargeOmpBoundary` 套件 5 个边界测试：`ReorderNcxyOne / ExtractNcxyOne`（`ncxy=1`）、`ReorderWithZeroSlabRank`（`numz=[0,1,...]`）、`ReorderSingleRankPool`（nproc=1）以及最关键的 `ReorderThreadCountInvariance`（`omp_set_num_threads(1/2/4)` 三轮逐字节一致）。题目二总测试数升至 9/9，题目一仍 4/4。

## 改动汇总（代码层面，截至阶段 6）

| 文件 | 改动 | 备注 |
|---|---|---|
| [source/source_estate/CMakeLists.txt](file:///root/homework/abacus-develop/source/source_estate/CMakeLists.txt) | +4 行，挂载 `test_omp` 子目录 | 仅在 `USE_OPENMP` 时启用（阶段 2） |
| [source/source_estate/test_omp/CMakeLists.txt](file:///root/homework/abacus-develop/source/source_estate/test_omp/CMakeLists.txt) | 新增 | 注册 OpenMP 测试 target（阶段 2） |
| [source/source_estate/test_omp/charge_omp_test.cpp](file:///root/homework/abacus-develop/source/source_estate/test_omp/charge_omp_test.cpp) | 阶段 2 + 4 + 5 + 6 累计扩展 | `_OPENMP` 健全性 + 2 reorder + 1 extract + 5 边界（含线程数不变性） |
| [source/source_estate/module_charge/charge_mpi.cpp](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp) | `reorder_pool_rank_to_uniform`（阶段 4）和 `extract_uniform_to_local`（阶段 5）内分别加 `#pragma omp parallel for collapse(2)`，含 `#ifdef _OPENMP` 保护 | 题目一 target 无 `_OPENMP` → 行为不变 |

题目一目录 [source/source_estate/test_mpi/](file:///root/homework/abacus-develop/source/source_estate/test_mpi) 在所有阶段中**完全未动**。

## 验收命令（任意时刻可重跑）

```bash
# 重新 configure（首次必须）
cd /root/homework/abacus-develop/build && cmake ..

# 阶段 2 产物：编译并跑 OpenMP target
cmake --build . --target MODULE_ESTATE_charge_omp_test -j 4
OMP_NUM_THREADS=4 ./source/source_estate/test_omp/MODULE_ESTATE_charge_omp_test

# 回归验证：题目一 target 仍然全过
cmake --build . --target MODULE_ESTATE_charge_mpi_test -j 4
mpirun -np 4 ./source/source_estate/test_mpi/MODULE_ESTATE_charge_mpi_test
```

## 接下来要做的事

- 阶段 7：用 `OMP_NUM_THREADS=1/2/4/8` 跑性能曲线 + Amdahl 拟合（需要大网格 `nxyz ≥ 2×10^6`）
- 阶段 8（加分项）：抽象 `DataTransformFunc`
- 阶段 9：四种 OpenMP × MPI 组合的回归验证
