# 阶段 4：`reorder_pool_rank_to_uniform` OpenMP 化

## 1. 目的

在阶段 3 完成依赖分析后，正式把 `#pragma omp parallel for collapse(2)` 落到 `Charge::reorder_pool_rank_to_uniform`，并通过新增的两个单元测试验证：

- 单线程（`OMP_NUM_THREADS=1`）下结果与未改造前的串行参考一致
- 多线程（`OMP_NUM_THREADS=4`）下结果与串行参考完全一致（逐位 `EXPECT_EQ`）
- 题目一原 target（`_OPENMP` 未定义）行为完全未变，`MODULE_ESTATE_charge_mpi_test` 4 进程仍全过

## 2. 代码改动

唯一修改的源文件：[source/source_estate/module_charge/charge_mpi.cpp](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp)

### 2.1 改前

```cpp
void Charge::reorder_pool_rank_to_uniform(const double* array_tot,
                                          double* array_tot_aux,
                                          const int ip) const
{
    const int ncxy = this->rhopw->nx * this->rhopw->ny;
    for (int ir = 0; ir < ncxy; ++ir)
    {
        for (int iz = 0; iz < this->rhopw->numz[ip]; ++iz)
        {
            array_tot_aux[this->rhopw->nz * ir + this->rhopw->startz[ip] + iz]
                = array_tot[this->rhopw->numz[ip] * ir + this->rhopw->startz[ip] * ncxy + iz];
        }
    }
}
```

### 2.2 改后

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

### 2.3 改动要点

1. **提前把 `numz[ip]` / `startz[ip]` / `nz` 取成栈上 `const int`**
   - 让 OpenMP 内层循环上界 `numz_ip` 成为编译期可见的常量，使 `collapse(2)` 合法（OpenMP 要求 perfectly nested rectangular loop）。
   - 也避免线程并发访问 `rhopw->numz`、`rhopw->startz` 等成员指针时被反复 dereference。
2. **`#pragma omp parallel for collapse(2) schedule(static)`**
   - `collapse(2)`：把 `ir × iz` 视为单一迭代空间，最大化并行粒度，特别在小 `ncxy`、大 `numz_ip` 或反之时仍能均衡负载。
   - `schedule(static)`：每次迭代负载固定（一次赋值），静态切分零调度开销。
   - 没有 `reduction` / `atomic` / `critical`：阶段 3 已证明无写写、无 RAW 冲突。
3. **`#ifdef _OPENMP` 保护**
   - 题目一 target（`test_mpi/CMakeLists.txt` 中 `remove_definitions(-D_OPENMP)`）下 pragma 段消失，行为与基线完全一致。
   - 题目二 target（`test_omp/`）下 pragma 段生效。
4. **不动任何调用方**
   - 函数签名、参数语义、返回值、外部可见行为不变。
   - 题目一阶段四 `gather_pool_data_nonblocking` 的 `MPI_Waitsome` 路径每次只调一个 `ip` → 自然受益于函数内部加速，且接口无任何改变。

## 3. 新增单元测试

测试文件：[source/source_estate/test_omp/charge_omp_test.cpp](file:///root/homework/abacus-develop/source/source_estate/test_omp/charge_omp_test.cpp)

### 3.1 测试策略

- 自己手写一份串行参考 `serial_reorder_reference`，逐字节复现改造前的循环体。
- 自己手写一个 `PW_Basis` mock 工厂 `make_stub_pw_basis`，**绕过** `initgrids/setuptransform`，直接把 `nx/ny/nz/nxy/nxyz/numz[]/startz[]` 写死。这样可以构造任意 z 切片分布（包括 `numz[ip]=0` 这种极端情况）来测边界。
- 使用 `std::mt19937` + 固定 seed 生成确定性输入数据；生成在主线程上单线程完成，不会被多线程竞争。

### 3.2 测试用例

| 测试名 | 网格 | rank 切分 | 说明 |
|---|---|---|---|
| `ChargeOmpReorder.MatchesSerialReferenceUniformSplit` | nx=4, ny=5, nz=8 | numz=[2,2,2,2], startz=[0,2,4,6] | 均匀切分 |
| `ChargeOmpReorder.MatchesSerialReferenceUnevenSplit` | nx=3, ny=3, nz=7 | numz=[3,1,0,3], startz=[0,3,4,4] | 包含 `numz=0`、相同 `startz` 等不均匀边界 |

测试断言用 `ASSERT_EQ`（整数式重排，浮点数逐位精确相等），任何一处下标错位或 race 都会立刻被发现。

> 边界用例 `ncxy=1`、`numz=[0,1]` 等更复杂场景会放到阶段 6 系统补齐。这里只覆盖「OpenMP 并行版本能跑」+「不均匀切分仍正确」最关键的两条。

### 3.3 单线程退化测试

`OMP_NUM_THREADS=1` 时同样会进入 `#pragma omp parallel for collapse(2)`，但 OpenMP 运行时只创建 1 个线程。这等价于验证 pragma **本身**不会改变结果（即使没有真正并行）。

## 4. 实际编译与运行结果

### 4.1 同时编译两个 target

```bash
cd /root/homework/abacus-develop/build
cmake --build . --target MODULE_ESTATE_charge_omp_test MODULE_ESTATE_charge_mpi_test -j 4
```

输出末尾（截选）：

```
[100%] Building CXX object source/source_estate/test_omp/CMakeFiles/.../charge_mpi.cpp.o
[100%] Linking CXX executable MODULE_ESTATE_charge_omp_test
[100%] Built target MODULE_ESTATE_charge_omp_test
...
[100%] Building CXX object source/source_estate/test_mpi/CMakeFiles/.../charge_mpi.cpp.o
[100%] Linking CXX executable MODULE_ESTATE_charge_mpi_test
[100%] Built target MODULE_ESTATE_charge_mpi_test
```

同一份 `charge_mpi.cpp` 被分别编进两个 target，编译均无报错。

### 4.2 题目二 target × 4 线程

```bash
OMP_NUM_THREADS=4 ./MODULE_ESTATE_charge_omp_test
```

```
[==========] Running 3 tests from 2 test suites.
[ RUN      ] ChargeOmpSkeleton.OpenMPMacroEnabled                [       OK ] (0 ms)
[ RUN      ] ChargeOmpReorder.MatchesSerialReferenceUniformSplit [       OK ] (0 ms)
[ RUN      ] ChargeOmpReorder.MatchesSerialReferenceUnevenSplit  [       OK ] (0 ms)
[  PASSED  ] 3 tests.
```

### 4.3 题目二 target × 1 线程

```bash
OMP_NUM_THREADS=1 ./MODULE_ESTATE_charge_omp_test
```

```
[ RUN      ] ChargeOmpReorder.MatchesSerialReferenceUniformSplit [       OK ] (0 ms)
[ RUN      ] ChargeOmpReorder.MatchesSerialReferenceUnevenSplit  [       OK ] (0 ms)
[  PASSED  ] 3 tests.
```

### 4.4 题目一 target × 4 进程（回归）

```bash
mpirun -np 4 ./MODULE_ESTATE_charge_mpi_test | grep PASSED
```

```
[  PASSED  ] 4 tests.   (每个 MPI rank 一行，共 4 行)
```

## 5. 阶段 4 完成判定

| 检查项 | 结果 |
|---|---|
| `#pragma omp parallel for collapse(2)` 已加入 `reorder_pool_rank_to_uniform` | ✅ |
| `#ifdef _OPENMP` 保护到位，题目一 target 不受影响 | ✅ |
| OpenMP 版本在 `threads=4` 下与串行参考逐字节一致 | ✅ |
| OpenMP 版本在 `threads=1` 下与串行参考逐字节一致 | ✅ |
| `numz=0` / 相同 `startz` 等不均匀切分仍正确 | ✅ |
| 题目一原 target 4 进程全过 | ✅ |

阶段 4 通过 ✅ —— 可以进入阶段 5（对 `extract_uniform_to_local` 做对称的 OpenMP 化）。

## 6. 给后续阶段的提醒

1. **小网格无法体现加速比**：阶段 4 的测试网格（`nxyz ≤ 160`）只测正确性，不能用于性能比较。阶段 7 会用更大的网格（建议 `nxyz ≥ 10^7`）专门跑加速曲线。
2. **`schedule(static)` 当前最稳**：因为每次迭代代价均匀。如果阶段 7 测出尾部线程不平衡，再考虑换 `dynamic` 或显式 `chunk`。
3. **阶段 5 同样适用本阶段的模板**：把 `numz[RANK_IN_POOL]` 提前到栈上常量，再 `collapse(2)`。
4. **阶段 8 的 `DataTransformFunc` 抽象不要替换本阶段实现**：让题目一阶段四的 Waitsome 路径继续走当前 `reorder_pool_rank_to_uniform`，新抽象作为「另一种实现」并行存在即可。
