# 阶段 6：系统化边界单元测试

## 1. 目的

题目二要求项 4「正确性验证」和要求项 5「单元测试要求」要覆盖：

- `ncxy = 1`、`numz` 含 0、`numz=[0,1]` 这类极端切片
- 线程安全的、可重复的数据生成器
- 不同数据规模下都通过

阶段 4、5 已经把核心串并行一致性测试搭起来，本阶段在 [test_omp/charge_omp_test.cpp](file:///root/homework/abacus-develop/source/source_estate/test_omp/charge_omp_test.cpp) 中补齐 **5 个系统化的边界测试**，让覆盖面达到题目要求。

## 2. 改动范围

只动了一个文件：[source/source_estate/test_omp/charge_omp_test.cpp](file:///root/homework/abacus-develop/source/source_estate/test_omp/charge_omp_test.cpp)

- 新增 1 个测试套件 `ChargeOmpBoundary`，含 5 个 `TEST`
- 复用阶段 4/5 的 `make_stub_pw_basis` / `destroy_stub_pw_basis` / `serial_reorder_reference` / `serial_extract_reference`
- 没有动 `charge_mpi.cpp`、`charge.h`、`test_mpi/` 等任何生产代码或题目一相关文件

## 3. 新增测试一览

| 测试名 | 覆盖维度 | 关键参数 |
|---|---|---|
| `ChargeOmpBoundary.ReorderNcxyOne` | 二维平面退化为单点 | `nx=ny=1, nz=6, numz=[2,2,1,1]` |
| `ChargeOmpBoundary.ReorderWithZeroSlabRank` | rank 持有 0 z-plane + `numz` 取值含 0/1 混合 | `numz=[0,1,2,2]` |
| `ChargeOmpBoundary.ReorderSingleRankPool` | nproc=1，单 rank 占满 z | `nx=3,ny=4,nz=5, numz=[5]` |
| `ChargeOmpBoundary.ExtractNcxyOne` | extract 路径下的 ncxy=1 | 与 ReorderNcxyOne 同布局，按 rank 切换 `RANK_IN_POOL` 与 `startz_current` |
| `ChargeOmpBoundary.ReorderThreadCountInvariance` | 同输入在不同线程数下结果一致 | `omp_set_num_threads(1/2/4)` 三轮跑同一输入逐字节比较 |

最后一个测试用 `#ifdef _OPENMP` 包起来，因为它依赖运行时 `omp_set_num_threads`。题目一 target 无 `_OPENMP` 时这段不参与编译，互不影响。

## 4. 各测试的设计要点

### 4.1 `ReorderNcxyOne` & `ExtractNcxyOne`

- 把 `nx=1, ny=1` ⇒ `ncxy=1`，让外层循环退化成单步。这是 OpenMP `collapse(2)` 的极端情况：迭代空间从 `ncxy * numz_ip` 退化为 `numz_ip`，验证 pragma 在小迭代空间下不出错（OpenMP 可能直接把所有线程映射到同一线程上）。
- 同时也是 reorder 与 extract 都覆盖一次，确保两个函数对相同极端布局表现一致。

### 4.2 `ReorderWithZeroSlabRank`

- 直接构造 `numz = [0, 1, 2, 2]`，覆盖题目要求项 5 中明确点名的 `numz=[0,1]` 这一类边界。
- `numz[ip] = 0`：内层循环执行 0 次，函数必须不访问任何元素、不写、不崩。
- 同时让 `startz[0] = startz[1] = 0`（两个 rank 起始平面相同），验证 `numz=0` 的 rank 即便 `startz` 与其他 rank 重合也不会出错。

### 4.3 `ReorderSingleRankPool`

- nproc=1：一个 rank 持有完整 z 范围。覆盖最简单也最容易被忽略的「无 MPI 分块」情况，验证 OpenMP 并行版本与串行参考一致。

### 4.4 `ReorderThreadCountInvariance`

这是本阶段最核心的「并行正确性强证据」。逻辑：

```cpp
auto run_once = [&](std::vector<double>& out, int nthreads) {
    omp_set_num_threads(nthreads);
    std::fill(out.begin(), out.end(), 0.0);
    for (int ip = 0; ip < layout_numz.size(); ++ip)
        charge.reorder_pool_rank_to_uniform(input, out.data(), ip);
};

run_once(out_t1, 1);
run_once(out_t2, 2);
run_once(out_t4, 4);
// 逐字节断言 out_t1 == out_t2 == out_t4
```

如果存在写-写冲突或 schedule 误用，多线程下结果就会与单线程不一致，这条测试会直接抓住。结尾用 `omp_set_num_threads(saved_threads)` 还原全局状态。

### 4.5 线程安全的数据生成器

- 所有测试都用 `std::mt19937(seed)`，固定 seed 保证可重复；
- 生成发生在主线程上、单线程顺序生成；并行只发生在「读输入 → 写输出」阶段，由 OpenMP pragma 控制；
- 满足题目要求项 5 中「使用线程安全的数据生成器确保测试可重复」。

## 5. 实际编译与运行结果

### 5.1 编译

```bash
cd /root/homework/abacus-develop/build
cmake --build . --target MODULE_ESTATE_charge_omp_test -j 4
```

```
[100%] Built target MODULE_ESTATE_charge_omp_test
```

### 5.2 题目二 target × 4 线程

```bash
OMP_NUM_THREADS=4 ./MODULE_ESTATE_charge_omp_test
```

```
[==========] Running 9 tests from 4 test suites.

[ RUN      ] ChargeOmpSkeleton.OpenMPMacroEnabled                  [       OK ]
[ RUN      ] ChargeOmpReorder.MatchesSerialReferenceUniformSplit   [       OK ]
[ RUN      ] ChargeOmpReorder.MatchesSerialReferenceUnevenSplit    [       OK ]
[ RUN      ] ChargeOmpExtract.MatchesSerialReferenceForEachRank    [       OK ]
[ RUN      ] ChargeOmpBoundary.ReorderNcxyOne                      [       OK ]
[ RUN      ] ChargeOmpBoundary.ReorderWithZeroSlabRank             [       OK ]
[ RUN      ] ChargeOmpBoundary.ReorderSingleRankPool               [       OK ]
[ RUN      ] ChargeOmpBoundary.ExtractNcxyOne                      [       OK ]
[ RUN      ] ChargeOmpBoundary.ReorderThreadCountInvariance        [       OK ]

[  PASSED  ] 9 tests.
```

### 5.3 题目二 target × 1 线程

```bash
OMP_NUM_THREADS=1 ./MODULE_ESTATE_charge_omp_test
```

```
[  PASSED  ] 9 tests.
```

### 5.4 题目一 target × 4 进程（回归）

```bash
mpirun -np 4 ./MODULE_ESTATE_charge_mpi_test | grep PASSED
```

```
[  PASSED  ] 4 tests.   ×4 ranks
```

## 6. 阶段 6 完成判定

| 题目要求 | 覆盖测试 | 结果 |
|---|---|---|
| 单进程 / 多进程结果一致 | 全部 `Matches…` + `ThreadCountInvariance` | ✅ |
| `ncxy = 1` | `ReorderNcxyOne`、`ExtractNcxyOne` | ✅ |
| `numz = [0, 1]` 类边界 | `ReorderWithZeroSlabRank`、`MatchesSerialReferenceUnevenSplit` | ✅ |
| 线程安全可重复生成器 | `std::mt19937(seed)`，主线程顺序生成 | ✅ |
| 不同线程数下结果一致 | `ReorderThreadCountInvariance`（1/2/4 三档） | ✅ |
| 单 rank 路径正确 | `ReorderSingleRankPool` | ✅ |
| 题目一原 target 不退化 | `mpirun -np 4 MODULE_ESTATE_charge_mpi_test` | ✅ |

阶段 6 通过 ✅ —— 可以进入阶段 7（性能测试 + Amdahl 分析）。

## 7. 给后续阶段的提示

1. 阶段 6 的网格规模仍然非常小（最大 `nxyz=324`），耗时只在 ~1 ms 量级，**不可作为性能加速比的样本**。
2. 阶段 7 性能测试需要专门加大网格（建议 `nx=ny=nz ≥ 128`，即 `nxyz ≥ 2×10^6`）并跑多次取平均，否则 OpenMP 启动开销会把信号淹没。
3. `omp_set_num_threads` 的就地切换在 GTest 框架下是安全的；阶段 7 用同样手法切换 1/2/4/8 线程时记得在每个测试结束还原。
4. `ChargeOmpBoundary.ReorderThreadCountInvariance` 实际上也是阶段 7 「正确性兜底」的一部分，加大网格的性能测试在每次切换线程数后可以同时做一次逐字节比较，复用本阶段的模式。
