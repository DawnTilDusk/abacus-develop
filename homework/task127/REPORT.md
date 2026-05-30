# ABACUS Charge 模块 重构与优化报告

> 提交目录：`homework/task127/`
> 涉及题目：题目一「MPI 归约操作的并行化」+ 题目二「OpenMP 多线程加速数据重排」
> 受改动的核心源文件：`source/source_estate/module_charge/charge_mpi.cpp`、`source/source_estate/module_charge/charge.h`

---

## 0. 摘要

本作业对 ABACUS 中 `Charge` 类的「跨 pool 归约」与「数据重排」两条核心通信/计算路径做了一次端到端的重构与并行化优化，分两大主题：

- **题目一**：以 fallback 路径 `reduce_diff_pools()` 为主目标，把阻塞式 `MPI_Allgatherv` + `MPI_Allreduce` 改造成「非阻塞点对点收集 + `MPI_Waitsome` 按块到达即重排 + 跨 pool 归约」的非阻塞流水线，同时把临时缓冲区上提为 `Charge` 成员，拆分出三个可独立测试的辅助函数。
- **题目二**：在题目一拆分出的两个数据重排函数 `reorder_pool_rank_to_uniform`、`extract_uniform_to_local` 上，加 `#pragma omp parallel for collapse(2) schedule(static)`，并新建独立 OpenMP 测试 target，补齐串并行一致性 / 边界 / 线程数不变性 / 性能基准测试。

两个主题保持**完全可叠加**：

- 题目一改的是 MPI 通信结构；
- 题目二改的是单进程内的循环并行化；
- 二者作用于「同一个函数的内外两层」，但接口、语义与现有 MPI 流程零变化，因此互相不破坏。

最终交付状态：

| 维度 | 状态 |
|---|---|
| 题目一阶段二+三+四已落地 | ✅ |
| 题目二阶段 1~7 + 9 已落地 | ✅（阶段 8 加分项可选） |
| 题目一 4 个原单元测试 | 全过（np=1、np=4） |
| 题目二 9 个单元测试 + 1 性能基准 | 全过（perf 默认 SKIPPED） |
| 4 种 OpenMP×MPI 组合最终回归 | 全过 |
| MPI 通信链路 / 数学逻辑 | 与原版本完全一致 |

---

## 1. 改动文件总览

```
abacus-develop/
├── source/source_estate/
│   ├── module_charge/
│   │   ├── charge.h               ← 题目一：新增 4 个函数声明 + 3 个成员缓冲区
│   │   └── charge_mpi.cpp         ← 题目一：重构 reduce_diff_pools / 新增 gather_pool_data_nonblocking 等
│   │                                题目二：reorder_pool_rank_to_uniform / extract_uniform_to_local 加 OpenMP pragma
│   ├── CMakeLists.txt             ← 题目二：+4 行挂载 test_omp 子目录
│   ├── test_mpi/                  ← 题目一已有，作业不动
│   └── test_omp/                  ← 题目二新增
│       ├── CMakeLists.txt
│       └── charge_omp_test.cpp    ← 9 + 1 个 GTest 用例（含性能基准）
└── homework/task127/
    ├── REPORT.md                  ← 本文件（综合报告）
    ├── task1/                     ← 题目一过程文档
    │   ├── task1.md               ← 题目一执行规划
    │   ├── charge_mpi.md          ← 题目一现状分析（逐行讲解）
    │   └── task1_algo.md          ← 题目一阶段二/三/四算法说明
    └── task2/                     ← 题目二过程文档
        ├── README.md              ← 题目二阶段索引
        ├── task2_plan.md          ← 题目二执行规划
        ├── stage1_baseline.md
        ├── stage2_test_target.md
        ├── stage3_dependency_analysis.md
        ├── stage4_reorder_openmp.md
        ├── stage5_extract_openmp.md
        ├── stage6_boundary_tests.md
        ├── stage7_performance.md
        └── stage9_regression.md
```

---

## 2. 题目一：MPI 归约操作的并行化

### 2.1 问题背景

ABACUS 在 k-point 并行模型中，不同 pool 各算一部分 k 点对电荷密度的贡献，输出前必须把不同 pool 的贡献求和。这个工作由 `Charge::reduce_diff_pools(double* array_rho)` 完成。

`reduce_diff_pools` 有两条路径：

- **直通路径**：`KP_WORLD != MPI_COMM_NULL` 时，不同 pool 中「对应 rank」持有的局部数据布局一致，可以直接调 `MPI_Allreduce`。
- **fallback 路径**：`KP_WORLD == MPI_COMM_NULL` 时，必须先在 pool 内收集出完整数据，再把数据从「按 rank 分块的局部布局」重排成「跨 pool 一致的统一布局」，再在 `INT_BGROUP` 上做 `MPI_Allreduce`，最后切回当前 rank 的本地 slab。

### 2.2 原版本的瓶颈

- **阻塞通信**：fallback 路径用 `MPI_Allgatherv` 收集 + `MPI_Allreduce` 归约，二者都是阻塞 collective。
- **大数组反复分配释放**：每次进入 `reduce_diff_pools` 都 `new[]` 三个 `nxyz` 量级的临时数组（`array_tmp`、`array_tot`、`array_tot_aux`），多自旋多次 SCF 迭代下产生大量分配/释放开销。
- **数据重排与通信耦合**：重排逻辑写在大函数里，单独测试困难，也不利于把"通信"和"计算"重叠。
- **计算/通信无重叠**：所有收集完成后才统一重排，串行依赖明显。

### 2.3 重构方案（阶段二）

把 `reduce_diff_pools` 重构为「调用链」而非「大函数」：

```
reduce_diff_pools(array_rho)
  ├── std::memcpy / scaling     → chgmpi_tmp_
  ├── gather_pool_data_nonblocking(chgmpi_tmp_, chgmpi_tot_, chgmpi_tot_aux_)
  │     ├── 本地 rank 块就地 memcpy 到 chgmpi_tot_ 对应偏移
  │     ├── 立即 reorder_pool_rank_to_uniform(..., my_rank)（本地不等通信，直接重排）
  │     ├── 对其它 rank 发 MPI_Irecv / MPI_Isend
  │     ├── MPI_Waitsome 驱动「按块到达即重排」
  │     │     循环调用 reorder_pool_rank_to_uniform(..., ip)
  │     └── MPI_Waitall（发送端）
  ├── MPI_Allreduce(chgmpi_tot_aux_, chgmpi_tot_, …, INT_BGROUP)
  └── extract_uniform_to_local(chgmpi_tot_, array_rho)
```

具体新增 / 拆分 / 重构的对象：

- **新增成员缓冲区**（`Charge` 类内、`init_chgmpi()` 统一分配）：
  - `double* chgmpi_tmp_`
  - `double* chgmpi_tot_`
  - `double* chgmpi_tot_aux_`
- **新增 / 拆分函数**：
  - `void reorder_pool_to_uniform(const double* array_tot, double* array_tot_aux) const`
  - `void reorder_pool_rank_to_uniform(const double* array_tot, double* array_tot_aux, const int ip) const`
  - `void extract_uniform_to_local(const double* array_tot, double* array_rho) const`
  - `void gather_pool_data_nonblocking(const double* array_tmp, double* array_tot, double* array_tot_aux) const`

收益：

- 通信逻辑与下标映射逻辑解耦；
- 临时缓冲区集中由 `Charge` 持有，不再反复 `new[]/delete[]`；
- 每段都可被单独单元测试；
- 为后续替换通信模式（非阻塞、按块重排、root-only 等）提供清晰扩展点。

### 2.4 非阻塞改造（阶段三）

把 fallback 路径中的 `MPI_Allgatherv` 替换为显式的非阻塞点对点收集：

```cpp
// 本地块
memcpy(array_tot + dis[my_rank], array_tmp, sizeof(double) * rec[my_rank]);
reorder_pool_rank_to_uniform(array_tot, array_tot_aux, my_rank);

// 非本地块
for ip != my_rank: MPI_Irecv(array_tot + dis[ip], rec[ip], …, ip, POOL_WORLD)
for ip != my_rank: MPI_Isend(array_tmp,            rec[my_rank], …, ip, POOL_WORLD)
MPI_Waitall(…)
```

满足题目「使用 MPI_Irecv / MPI_Isend」的硬性要求，把原来由 `MPI_Allgatherv` 隐式做的全员互换显式化，便于进一步插入计算/通信重叠。

### 2.5 通信–计算重叠（阶段四）

在阶段三基础上引入 `MPI_Waitsome`，**接收完成一个 rank 就立即重排该 rank 的块**，把数据重排嵌进通信等待中：

```cpp
copy local block & reorder local
issue all MPI_Irecv / MPI_Isend
while there are unfinished recvs:
    MPI_Waitsome(…)
    for each completed ip:
        reorder_pool_rank_to_uniform(array_tot, array_tot_aux, ip)
MPI_Waitall(send_requests)
MPI_Allreduce(array_tot_aux, array_tot, …, INT_BGROUP)
extract_uniform_to_local(array_tot, array_rho)
```

这一步真正贴近题目要求中「通信与计算重叠」的目标。

### 2.6 正确性

- **数学逻辑不变**：`array_rho → array_tmp → array_tot → array_tot_aux → reduce → array_rho` 的语义与原版本逐字节一致。
- **接口不变**：`reduce_diff_pools`、`rho_mpi`、`kin_r_mpi` 对外签名与行为完全不变。
- **单元测试通过**：题目一原 4 个测试（`reduce_diff_pools1`、`reduce_diff_pools2`、`rho_mpi`、`kin_r_mpi`）在 np=1 与 np=4 下均全过。

### 2.7 已知局限

- 跨 pool `MPI_Allreduce` 仍为阻塞式（可在后续阶段六换 `MPI_Iallreduce`）。
- full-grid 缓冲区 `chgmpi_tot_` / `chgmpi_tot_aux_` 在每个进程中各持一份；root-only gather/reduce/scatter 方案可进一步节省内存，但会引入 root 热点。

详细数据流图、阶段二/三/四的伪代码与对比，见 [task1/task1_algo.md](./task1/task1_algo.md)。

---

## 3. 题目二：OpenMP 多线程加速数据重排

### 3.1 目标循环

题目二的「三层嵌套循环」实际由阶段四暴露出来的两个函数承载：

- `Charge::reorder_pool_rank_to_uniform`（`ir × iz`，外层 `ip` 由调用方提供 → 三层）
- `Charge::extract_uniform_to_local`（`ir × iz`，rank 固定为 `RANK_IN_POOL`）

### 3.2 循环依赖分析（阶段 3）

记一次内层迭代 `(ir, iz)` 的写入下标 `W(ir, iz)`：

- `reorder_pool_rank_to_uniform`：`W = nz * ir + startz[ip] + iz`，因 `iz ∈ [0, numz[ip]) ⊆ [0, nz)`，按 `mod nz` 单射性 ⇒ 不同 `(ir, iz)` 写入互不相等。
- `extract_uniform_to_local`：`W = numz[rank] * ir + iz`，同样有 `mod numz[rank]` 单射性。

读出端 `array_tot` 只读、写出端 `array_tot_aux` / `array_rho` 是**独立缓冲区**，没有 RAW/WAR。
不同 `ip` 之间因 `[startz[ip], startz[ip] + numz[ip])` 不相交，写入区段无重叠。

结论：三层 `ip × ir × iz` 完全 **embarrassingly parallel**，可直接 `#pragma omp parallel for collapse(2)` 内层两层，**不需要 reduction / atomic / critical**。

### 3.3 OpenMP 改造（阶段 4 / 5）

两个函数都按同一个模板改造：

```cpp
void Charge::reorder_pool_rank_to_uniform(const double* array_tot,
                                          double* array_tot_aux,
                                          const int ip) const
{
    const int ncxy = this->rhopw->nx * this->rhopw->ny;
    const int numz_ip = this->rhopw->numz[ip];
    const int startz_ip = this->rhopw->startz[ip];
    const int nz = this->rhopw->nz;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int ir = 0; ir < ncxy; ++ir)
    {
        for (int iz = 0; iz < numz_ip; ++iz)
        {
            array_tot_aux[nz * ir + startz_ip + iz]
                = array_tot[numz_ip * ir + startz_ip * ncxy + iz];
        }
    }
}
```

要点：

1. 把 `numz[ip]/startz[ip]/nz` 提前到栈上 `const int`，让 `collapse(2)` 合法（rectangular nested loop）并避免线程并发 dereference `rhopw` 成员指针。
2. `#ifdef _OPENMP` 保护：题目一原测试 target 在 `test_mpi/CMakeLists.txt` 显式 `remove_definitions(-D_OPENMP)`，因此那一份编译路径上 pragma 整段消失、行为与基线完全一致。
3. 不动调用方（包括阶段四的 `MPI_Waitsome` 路径）— 函数签名、副作用、对外语义零变化。

### 3.4 测试基础设施（阶段 2 / 4 / 5 / 6 / 7）

**新建** `source/source_estate/test_omp/`，独立挂载，不破坏题目一：

- 在 `source/source_estate/CMakeLists.txt` 添加 4 行：
  ```cmake
  if(USE_OPENMP)
      add_subdirectory(test_omp)
  endif()
  ```
- `test_omp/CMakeLists.txt`：与 `test_mpi/CMakeLists.txt` 几乎一致，**故意不**写 `remove_definitions(-D_OPENMP)`，让 `#pragma omp` 生效。
  ```cmake
  AddTest(
    TARGET MODULE_ESTATE_charge_omp_test
    LIBS parameter ${math_libs} psi base device planewave
    SOURCES charge_omp_test.cpp ../module_charge/charge_mpi.cpp
  )
  add_test(NAME MODULE_ESTATE_charge_omp_test_4np
        COMMAND mpirun -np 4 ./MODULE_ESTATE_charge_omp_test;
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  ```
- `test_omp/charge_omp_test.cpp`：10 个 GTest 用例。

### 3.5 单元测试覆盖

| 套件 | 用例 | 覆盖意图 |
|---|---|---|
| `ChargeOmpSkeleton` | `OpenMPMacroEnabled` | 编译宏 `_OPENMP` 在该 target 中已启用 |
| `ChargeOmpReorder` | `MatchesSerialReferenceUniformSplit` | 均匀 z 切分下 reorder 串并行结果逐字节一致 |
| `ChargeOmpReorder` | `MatchesSerialReferenceUnevenSplit` | 含 `numz=0` 的不均匀切分 |
| `ChargeOmpExtract` | `MatchesSerialReferenceForEachRank` | 按 rank 切换 `RANK_IN_POOL` / `startz_current` 后 extract 一致 |
| `ChargeOmpBoundary` | `ReorderNcxyOne` | `nx=ny=1` 极端二维退化 |
| `ChargeOmpBoundary` | `ExtractNcxyOne` | 同上，extract 路径 |
| `ChargeOmpBoundary` | `ReorderWithZeroSlabRank` | `numz=[0,1,2,2]`，覆盖题目要求的 `numz=[0,1]` |
| `ChargeOmpBoundary` | `ReorderSingleRankPool` | nproc=1，单 rank 占满 z |
| `ChargeOmpBoundary` | `ReorderThreadCountInvariance` | `omp_set_num_threads(1/2/4)` 三轮逐字节一致 |
| `ChargeOmpPerf` | `ReorderAndExtractSpeedup` | 128³ 网格、`CHARGE_OMP_BENCH=1` 启用、默认 SKIP 的性能基准 |

数据生成器统一用 `std::mt19937(seed)`，主线程顺序生成，满足题目「线程安全的数据生成器确保测试可重复」。

### 3.6 性能测试结果与 Amdahl 分析（阶段 7）

机器：**容器内 2 个逻辑 CPU**（Intel Xeon Platinum 8163, 2.5 GHz）。
配置：128³ 网格、`nproc=4`、每档 5 次重复取平均、3 轮独立运行取中位数。

| threads | reorder (ms) | extract (ms) | S_r 加速比 | S_e 加速比 |
|---|---|---|---|---|
| 1 | 5.116 | 13.425 | 1.00 | 1.00 |
| 2 | 4.259 | 7.356 | **1.20** | **1.83** |
| 4 | 5.161 | 9.189 | 0.99 | 1.46 |
| 8 | 5.277 | 8.394 | 0.97 | 1.60 |

代入 Amdahl 公式 `S(p) = 1 / ((1-f) + f/p)`，用 `p=2` 反推：

- **reorder**：S(2)=1.20 ⇒ `f_reorder ≈ 0.33`，`S(∞)_reorder ≤ 1.49`
- **extract**：S(2)=1.83 ⇒ `f_extract ≈ 0.91`，`S(∞)_extract ≤ 11.1`

**讨论**：

- t=2 已经是该机器的最佳点，与物理核数匹配；
- t=4/8 退化属于经典 **oversubscription**（容器只有 2 逻辑核），不是并行实现错误；
- `extract` 加速明显高于 `reorder` —— 因为 `extract` 写下标 `numz_local * ir + iz` 是**连续**的，更好地利用了内存带宽；`reorder` 写下标 `nz * ir + startz_ip + iz` 步长为 `nz`，存在跨步写，带宽利用率较低；这与 Amdahl 拟合得到的 `f` 差异一致。

### 3.7 最终回归（阶段 9）

四种 OpenMP × MPI 正交组合矩阵：

| 组合 | OpenMP | MPI | target | 结果 |
|---|---|---|---|---|
| A | OFF | np=1 | `MODULE_ESTATE_charge_mpi_test` | ✅ 4/4 PASSED |
| B | OFF | np=4 | `MODULE_ESTATE_charge_mpi_test` | ✅ 4/4 × 4 ranks PASSED |
| C | ON, threads=4 | np=1 | `MODULE_ESTATE_charge_omp_test` | ✅ 9/9 + 1 SKIPPED |
| D | ON, threads=2 | np=4 | `MODULE_ESTATE_charge_omp_test` | ✅ 9/9 × 4 ranks + 1 SKIPPED × 4 |

证明：

- 题目一原 target 不退化（A、B）；
- 题目二 OpenMP 路径正确（C）；
- MPI + OpenMP 混合下无 race、无死锁（D）。

---

## 4. 题目一与题目二的协同关系

| 维度 | 题目一 | 题目二 | 是否冲突 |
|---|---|---|---|
| 并行层级 | 进程级（MPI） | 线程级（OpenMP） | ❌ 不冲突，互补 |
| 主要目标 | 非阻塞通信 / 通信–计算重叠 | 单进程内三层循环并行 | ❌ 不冲突 |
| 共用函数 | 阶段四调用 `reorder_pool_rank_to_uniform` | 在该函数体内加 pragma | ✅ 函数内行为级优化，对外语义不变 |
| 缓冲区 `chgmpi_*_` | 题目一新增 | 题目二复用 | ❌ 不冲突 |
| 编译宏 `-D_OPENMP` | 题目一 target 关 | 题目二 target 开 | ⚠️ 用 `#ifdef _OPENMP` 隔离 |
| 单元测试 | `test_mpi/` 已有 | `test_omp/` 独立新增 | ❌ 不冲突 |

关键点：题目二的 OpenMP 改造**叠加**在题目一阶段四暴露的两个函数体内部，**不改函数签名、不改调用链、不动 MPI 流程**，所以阶段四的 `MPI_Waitsome` 路径每次调用 `reorder_pool_rank_to_uniform` 时会自然受益于多线程加速。

---

## 5. 复现命令（评阅可一键验证）

### 5.1 编译

```bash
cd /root/homework/abacus-develop/build
cmake ..
cmake --build . --target MODULE_ESTATE_charge_mpi_test MODULE_ESTATE_charge_omp_test -j 4
```

### 5.2 跑题目一测试（OpenMP 自动关闭）

```bash
cd source/source_estate/test_mpi
./MODULE_ESTATE_charge_mpi_test          # np=1
mpirun -np 4 ./MODULE_ESTATE_charge_mpi_test
```

### 5.3 跑题目二单元测试（OpenMP 启用）

```bash
cd ../test_omp
OMP_NUM_THREADS=4 ./MODULE_ESTATE_charge_omp_test
# => 9 PASSED + 1 SKIPPED (perf)
```

### 5.4 跑性能基准

```bash
CHARGE_OMP_BENCH=1 ./MODULE_ESTATE_charge_omp_test \
    --gtest_filter='ChargeOmpPerf*'
```

### 5.5 最终回归矩阵（一键全部）

```bash
cd /root/homework/abacus-develop/build/source/source_estate
echo "A:" && ./test_mpi/MODULE_ESTATE_charge_mpi_test | grep PASSED
echo "B:" && mpirun -np 4 ./test_mpi/MODULE_ESTATE_charge_mpi_test | grep PASSED | sort | uniq -c
echo "C:" && OMP_NUM_THREADS=4 ./test_omp/MODULE_ESTATE_charge_omp_test | grep -E "PASSED|SKIPPED"
echo "D:" && OMP_NUM_THREADS=2 mpirun -np 4 ./test_omp/MODULE_ESTATE_charge_omp_test \
    | grep -E "PASSED|SKIPPED" | sort | uniq -c
```

---

## 6. 题目要求逐项核对

### 6.1 题目一

| 题目要求 | 完成方式 |
|---|---|
| 现有代码分析 | [task1/charge_mpi.md](./task1/charge_mpi.md)（逐行解读） |
| MPI 通信模式识别 | [task1/task1.md](./task1/task1.md) 第 2~4 步 |
| 性能瓶颈说明 | [task1/task1_algo.md](./task1/task1_algo.md) §4 原始实现的主要问题 |
| 非阻塞并行版本设计/实现 | [task1/task1_algo.md](./task1/task1_algo.md) §6 第三阶段、§7 第四阶段 |
| 性能测试 | 题目一原 4 个 GTest 用例 + 题目二阶段 7 的 perf benchmark（间接覆盖 reorder/extract） |
| 单元测试 | `test_mpi/charge_mpi_test.cpp` 4 个用例 |

### 6.2 题目二

| 题目要求 | 完成方式 |
|---|---|
| 1. 循环依赖分析 | [task2/stage3_dependency_analysis.md](./task2/stage3_dependency_analysis.md) |
| 2. OpenMP 实现 | [task2/stage4_reorder_openmp.md](./task2/stage4_reorder_openmp.md) + [stage5_extract_openmp.md](./task2/stage5_extract_openmp.md) |
| 3. 性能对比 + Amdahl 分析 | [task2/stage7_performance.md](./task2/stage7_performance.md) |
| 4. 正确性验证 / 线程安全检查 | [task2/stage4](./task2/stage4_reorder_openmp.md) + [5](./task2/stage5_extract_openmp.md) + [6](./task2/stage6_boundary_tests.md) + [9](./task2/stage9_regression.md) |
| 5. 单元测试 + 边界 + 可重复生成器 | [task2/stage6_boundary_tests.md](./task2/stage6_boundary_tests.md) + `ChargeOmpBoundary.*` |
| 6. 代码重构（DataTransformFunc 抽象）—— 加分项 | 未做 |

---

## 7. 主要改动文件清单（精确到行）

| 文件 | 改动 | 阶段 |
|---|---|---|
| `source/source_estate/module_charge/charge.h` | 新增 4 个函数声明、3 个 `double*` 成员 | 题目一 阶段二 |
| `source/source_estate/module_charge/charge_mpi.cpp` | 重构 `init_chgmpi` / `reduce_diff_pools`，新增 `reorder_pool_to_uniform` / `reorder_pool_rank_to_uniform` / `extract_uniform_to_local` / `gather_pool_data_nonblocking` | 题目一 阶段二/三/四 |
| `source/source_estate/module_charge/charge_mpi.cpp:46-63` | `reorder_pool_rank_to_uniform` 加 `#pragma omp parallel for collapse(2) schedule(static)`（带 `#ifdef _OPENMP`） | 题目二 阶段 4 |
| `source/source_estate/module_charge/charge_mpi.cpp:65-82` | `extract_uniform_to_local` 加同样的 pragma | 题目二 阶段 5 |
| `source/source_estate/CMakeLists.txt` | +4 行：`if(USE_OPENMP) add_subdirectory(test_omp) endif()` | 题目二 阶段 2 |
| `source/source_estate/test_omp/CMakeLists.txt` | 新增 | 题目二 阶段 2 |
| `source/source_estate/test_omp/charge_omp_test.cpp` | 新增 + 阶段 4/5/6/7 累计扩展 | 题目二 阶段 2/4/5/6/7 |

**题目一目录 `test_mpi/` 在整个题目二期间完全未动。**

---

## 8. 自我评估

- **正确性**：题目一原 4 个测试 + 题目二 9 个测试 + 4 种 MPI/OpenMP 组合矩阵全过；逐字节比较多线程结果 vs 串行参考。
- **可维护性**：临时缓冲区集中由 `Charge` 管理；fallback 路径被拆成可独立测试的几个小函数；OpenMP 通过 `#ifdef _OPENMP` 隔离，原编译路径零行为变化。
- **性能**：reorder 在 2 线程下加速 1.20×、extract 加速 1.83×；硬件不允许进一步扩展，Amdahl 模型解释合理。
- **可扩展性**：阶段四已经为「`MPI_Iallreduce`」「按块归约 + 计算重叠」「DataTransformFunc 抽象（题目二加分项）」预留了清晰的扩展点。

---

## 9. 附：可选后续优化方向

- 题目一第五阶段：`reduce_diff_pools` 的 fallback 路径改成 **root-only gather / reduce / scatter**，把 `chgmpi_tot_*` 的内存占用从 `O(nxyz * nproc)` 降到 `O(nxyz)`，代价是 root 成为热点。
- 题目一第六阶段：把 `INT_BGROUP` 的 `MPI_Allreduce` 改成 `MPI_Iallreduce`，让跨 pool 归约也异步。
- 题目二加分项：把 `reorder_pool_rank_to_uniform` 的循环体抽成
  `using DataTransformFunc = std::function<double(int, int, int, const double*)>;`
  + `parallel_data_rearrange(input, output, transform)`，为后续 CPU/GPU 不同 backend 留接口。
- 用 NUMA-aware `first-touch` 初始化输出缓冲，进一步改善 reorder 的内存带宽利用率。
