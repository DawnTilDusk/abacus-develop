# 题目二：执行进度与产物索引

本目录存放题目二（OpenMP 多线程加速数据重排）的全部规划、阶段性报告和测试基线。原始规划文件为 `task2_plan.md`，本 README 记录已完成阶段的产物清单和验收状态。

## 已完成阶段

| 阶段 | 产物 | 状态 |
|---|---|---|
| 阶段 1 基线锁定 | [stage1_baseline.md](./stage1_baseline.md) | ✅ 完成 |
| 阶段 2 OpenMP 测试 target | [stage2_test_target.md](./stage2_test_target.md) | ✅ 完成 |
| 阶段 3 循环依赖分析 | [stage3_dependency_analysis.md](./stage3_dependency_analysis.md) | ✅ 完成 |

## 阶段 1 ~ 3 一句话总结

1. **阶段 1**：在不改代码的前提下，跑通题目一现有的 `MODULE_ESTATE_charge_mpi_test`（np=1 与 np=4），4/4 测试全过，作为题目二的对照基线，并明确 OpenMP 宏当前被 `remove_definitions` 抹掉的事实。
2. **阶段 2**：新建独立目录 `source/source_estate/test_omp/`，注册 target `MODULE_ESTATE_charge_omp_test` + ctest 入口 `MODULE_ESTATE_charge_omp_test_4np`。**不动**题目一目录任何文件，由 `USE_OPENMP` 条件挂载。最小空壳测试 `ChargeOmpSkeleton.OpenMPMacroEnabled` 验证 `_OPENMP` 宏在新 target 中已启用。题目一原 target 重新编译 + 4np 跑测，仍全过。
3. **阶段 3**：对 `reorder_pool_rank_to_uniform` 与 `extract_uniform_to_local` 做严格的下标依赖分析，证明三层 `ip × ir × iz` 完全无写–写、无 RAW、无 WAR 冲突，并据此选定 OpenMP 策略为 `#pragma omp parallel for collapse(2) schedule(static)`，且加 `#ifdef _OPENMP` 保护以兼容题目一 target。

## 改动汇总（代码层面，截至阶段 3）

| 文件 | 改动 | 备注 |
|---|---|---|
| [source/source_estate/CMakeLists.txt](file:///root/homework/abacus-develop/source/source_estate/CMakeLists.txt) | +4 行，挂载 `test_omp` 子目录 | 仅在 `USE_OPENMP` 时启用 |
| [source/source_estate/test_omp/CMakeLists.txt](file:///root/homework/abacus-develop/source/source_estate/test_omp/CMakeLists.txt) | 新增 | 注册 OpenMP 测试 target |
| [source/source_estate/test_omp/charge_omp_test.cpp](file:///root/homework/abacus-develop/source/source_estate/test_omp/charge_omp_test.cpp) | 新增 | GTest+MPI 入口 + `_OPENMP` 健全性测试 |

题目一目录 [source/source_estate/test_mpi/](file:///root/homework/abacus-develop/source/source_estate/test_mpi) 与 [source/source_estate/module_charge/charge_mpi.cpp](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp) 在阶段 1 ~ 3 内**完全未动**。

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

- 阶段 4：在 `reorder_pool_rank_to_uniform` 加 `#pragma omp parallel for collapse(2)` + `#ifdef _OPENMP` 保护
- 阶段 5：在 `extract_uniform_to_local` 加同样的 pragma
- 阶段 6：在 `test_omp/charge_omp_test.cpp` 中补串并行一致性 + 边界 `ncxy=1` / `numz=[0,1]` 等单元测试
- 阶段 7：用 `OMP_NUM_THREADS=1/2/4/8` 跑性能曲线 + Amdahl 拟合
- 阶段 8（加分项）：抽象 `DataTransformFunc`
- 阶段 9：四种 OpenMP × MPI 组合的回归验证
