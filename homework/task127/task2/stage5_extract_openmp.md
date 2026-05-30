# 阶段 5：`extract_uniform_to_local` OpenMP 化

## 1. 目的

把阶段 4 在 `reorder_pool_rank_to_uniform` 上的 OpenMP 改造，对称地落到 `Charge::extract_uniform_to_local`，并新增对应的串并行一致性测试。

`extract_uniform_to_local` 是 `reduce_diff_pools` 的最后一步：从已经归约完成的统一布局 `array_tot` 中，把**当前 rank** 的本地 z-slab 切出来写回 `array_rho`。

## 2. 代码改动

唯一修改的源文件：[source/source_estate/module_charge/charge_mpi.cpp](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp)

### 2.1 改前

```cpp
void Charge::extract_uniform_to_local(const double* array_tot, double* array_rho) const
{
    const int ncxy = this->rhopw->nx * this->rhopw->ny;
    for (int ir = 0; ir < ncxy; ir++)
    {
        for (int iz = 0; iz < this->rhopw->numz[GlobalV::RANK_IN_POOL]; iz++)
        {
            array_rho[this->rhopw->numz[GlobalV::RANK_IN_POOL] * ir + iz]
                = array_tot[this->rhopw->nz * ir + this->rhopw->startz_current + iz];
        }
    }
}
```

### 2.2 改后

```cpp
void Charge::extract_uniform_to_local(const double* array_tot, double* array_rho) const
{
    const int ncxy = this->rhopw->nx * this->rhopw->ny;
    const int numz_local = this->rhopw->numz[GlobalV::RANK_IN_POOL];
    const int startz_local = this->rhopw->startz_current;
    const int nz = this->rhopw->nz;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int ir = 0; ir < ncxy; ir++)
    {
        for (int iz = 0; iz < numz_local; iz++)
        {
            array_rho[numz_local * ir + iz]
                = array_tot[nz * ir + startz_local + iz];
        }
    }
}
```

### 2.3 改动要点（与阶段 4 对齐）

1. **栈上常量提升**：`numz[RANK_IN_POOL] / startz_current / nz` 提到 `const int`，满足 `collapse(2)` 对 rectangular nested loop 的要求，同时避免线程并发反复 dereference `rhopw` 成员指针。
2. **`#pragma omp parallel for collapse(2) schedule(static)`**：与阶段 4 完全一致的并行策略。
3. **`#ifdef _OPENMP` 保护**：题目一 target（无 `_OPENMP` 宏）下行为完全不变。
4. **不动调用方**：函数签名与对外语义不变。`reduce_diff_pools` 的尾部仍然原样调用 `extract_uniform_to_local(chgmpi_tot_, array_rho)`，只是函数内部走了多线程。

### 2.4 与 `reorder_pool_rank_to_uniform` 的细微差别

| 维度 | `reorder_pool_rank_to_uniform` | `extract_uniform_to_local` |
|---|---|---|
| 外部参数提供的 `ip` | 有（任意 rank） | 无（固定为 `RANK_IN_POOL`） |
| z 区间偏移 | `startz[ip]`（参数化 ip） | `startz_current`（成员变量，调用前由 PW 设好） |
| z 层数 | `numz[ip]`（参数化 ip） | `numz[RANK_IN_POOL]` |

所以测试需要额外**模拟** `GlobalV::RANK_IN_POOL` 与 `rhopw->startz_current` 才能让被测函数定位到正确的 slab。

## 3. 新增单元测试

测试文件：[source/source_estate/test_omp/charge_omp_test.cpp](file:///root/homework/abacus-develop/source/source_estate/test_omp/charge_omp_test.cpp)

### 3.1 新增组件

- `serial_extract_reference`：手写串行参考，逐字节复现改造前的循环体。
- 复用阶段 4 的 `make_stub_pw_basis` 来构造任意 z 切分。

### 3.2 测试用例

`ChargeOmpExtract.MatchesSerialReferenceForEachRank`：

- 网格：`nx=4, ny=5, nz=8`
- rank 切分：`numz=[3, 1, 0, 4]`，`startz=[0, 3, 4, 4]`
  - 同时覆盖：满 slab、最小 slab、空 slab、与上一个 rank `startz` 相同的 slab
- 循环模拟每个 rank：
  - 设置 `GlobalV::RANK_IN_POOL = ip`
  - 设置 `pw->startz_current = layout_startz[ip]`
  - 跑被测的 `extract_uniform_to_local` 与串行参考，逐位 `ASSERT_EQ`
- 测试结束后**还原** `GlobalV::RANK_IN_POOL`（避免污染其他测试）

### 3.3 测试可重复性

输入数据仍由 `std::mt19937(seed=7)` 生成，串行入栈写入；所有线程的写入位置由 `collapse(2)` 分片后互不重叠，任何 race 都会直接导致 `ASSERT_EQ` 失败。

## 4. 实际编译与运行结果

### 4.1 同时编译两个 target

```bash
cd /root/homework/abacus-develop/build
cmake --build . --target MODULE_ESTATE_charge_omp_test MODULE_ESTATE_charge_mpi_test -j 4
```

```
[100%] Built target MODULE_ESTATE_charge_omp_test
...
[100%] Built target MODULE_ESTATE_charge_mpi_test
```

### 4.2 题目二 target × 4 线程

```bash
OMP_NUM_THREADS=4 ./MODULE_ESTATE_charge_omp_test
```

```
[==========] Running 4 tests from 3 test suites.
[ RUN      ] ChargeOmpSkeleton.OpenMPMacroEnabled                  [       OK ] (0 ms)
[ RUN      ] ChargeOmpReorder.MatchesSerialReferenceUniformSplit   [       OK ] (0 ms)
[ RUN      ] ChargeOmpReorder.MatchesSerialReferenceUnevenSplit    [       OK ] (0 ms)
[ RUN      ] ChargeOmpExtract.MatchesSerialReferenceForEachRank    [       OK ] (0 ms)
[  PASSED  ] 4 tests.
```

### 4.3 题目二 target × 1 线程

```bash
OMP_NUM_THREADS=1 ./MODULE_ESTATE_charge_omp_test
```

```
[  PASSED  ] 4 tests.
```

### 4.4 题目一 target × 4 进程（回归）

```bash
mpirun -np 4 ./MODULE_ESTATE_charge_mpi_test | grep PASSED
```

```
[  PASSED  ] 4 tests.   ×4 ranks
```

## 5. 阶段 5 完成判定

| 检查项 | 结果 |
|---|---|
| `#pragma omp parallel for collapse(2)` 已加入 `extract_uniform_to_local` | ✅ |
| `#ifdef _OPENMP` 保护到位，题目一 target 行为不变 | ✅ |
| OpenMP 版本在 `threads=4 / 1` 下与串行参考逐位一致 | ✅ |
| 不均匀切分（含空 slab）下仍正确 | ✅ |
| 题目一原 target 4 进程仍全过 | ✅ |
| 测试结束后 `GlobalV::RANK_IN_POOL` 被还原，无污染 | ✅ |

阶段 5 通过 ✅ —— 可以进入阶段 6（系统补齐边界单测：`ncxy=1`、`numz=[0,1]`、单 rank 等）。

## 6. 当前两个 OpenMP 化函数的对比小结

| 函数 | 输入读位置 | 输出写位置 | 内层 z 上界 | OpenMP pragma |
|---|---|---|---|---|
| `reorder_pool_rank_to_uniform(*, ip)` | `numz[ip]*ir + startz[ip]*ncxy + iz` | `nz*ir + startz[ip] + iz` | `numz[ip]`（常量内联） | `parallel for collapse(2) schedule(static)` |
| `extract_uniform_to_local` | `nz*ir + startz_current + iz` | `numz[RANK]*ir + iz` | `numz[RANK]`（常量内联） | `parallel for collapse(2) schedule(static)` |

两个函数的读写下标在阶段 3 都已证明在固定 `ip` / `RANK` 时 `(ir, iz)` 完全独立可并行，**`#pragma omp parallel for collapse(2)` 是最自然且无需 reduction/atomic 的写法**。
