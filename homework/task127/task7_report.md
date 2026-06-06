# 题目 7：三线性插值的 OpenMP 并行化实验报告

## 1. 题目概述

本题要求优化 `source/source_io/module_output/read_cube.cpp` 中的 `trilinear_interpolate` 函数，实现其 OpenMP 并行版本，并完成正确性验证、性能测试、SIMD 可行性评估以及后续可扩展性分析。

需要完成的核心内容包括：

1. 分析三层嵌套循环的数据依赖，确定可并行层级。
2. 使用 `#pragma omp parallel for` 实现并行化。
3. 测试不同线程数下的性能并分析 cache 影响。
4. 可选使用 `#pragma omp simd` 进一步优化。
5. 补充单元测试，验证数值精度与边界条件。
6. 给出后续更高阶插值的可扩展重构方向。

## 2. 代码位置

- 目标实现文件：`source/source_io/module_output/read_cube.cpp`
- 目标函数：`ModuleIO::trilinear_interpolate`
- 调用位置：`ModuleIO::read_vdata_palgrid`
- 正确性测试文件：`source/source_io/test_serial/rho_io_test.cpp`
- 最小性能基准：`source/source_io/test_serial/trilinear_bench.cpp`

## 3. 原始实现分析

原始版本的 `trilinear_interpolate` 主要由两部分组成：

1. 先把输入数据从原始 `data_in[(ix * ny_read + iy) * nz_read + iz]` 布局重排到 `read_rho[iz][ix * ny_read + iy]`。
2. 再通过三层循环 `ix / iy / iz` 对目标网格上的每一个点执行八点三线性插值。

原始代码中，输出写入方式为：

```cpp
data_out[(ix * ny + iy) * nz + iz] = result;
```

这说明每个输出点 `(ix, iy, iz)` 只会被写一次，不同迭代之间不存在写冲突。插值时读取的数据全部来自只读输入数组，因此主循环天然满足并行条件。

## 4. 数据依赖与并行层级分析

### 4.1 数据依赖

三层循环中每次迭代只做两类操作：

- 读取输入网格中 8 个相邻顶点的值；
- 计算当前目标点的插值结果并写入唯一输出位置。

因此：

- `data_in` 是共享只读数据；
- `data_out[(ix, iy, iz)]` 是独占写入；
- `fracx/fracy/fracz`、`low/high`、权重等变量都是迭代私有变量；
- 不存在归约变量，也没有跨迭代前后依赖。

### 4.2 并行层级选择

本题中最适合并行的是外层二维平面 `(ix, iy)`，原因如下：

1. 每个线程负责一个输出平面的若干行，任务粒度较大。
2. 最内层 `iz` 保持连续访问，有利于顺序写出 `data_out`。
3. 若只并行 `iz`，粒度过细，线程调度开销较大。

因此本实现采用：

```cpp
#pragma omp parallel for collapse(2) schedule(static)
for (int ix = 0; ix < nx; ix++)
{
    for (int iy = 0; iy < ny; iy++)
    {
        ...
    }
}
```

其中 `collapse(2)` 将 `ix` 和 `iy` 两层循环合并，能够在规则网格任务中更均匀地分配工作量；`schedule(static)` 适用于每次迭代计算量近似一致的情形。

## 5. 并行实现方案

### 5.1 预计算单轴插值映射

为了减少在三重循环中重复执行 `fmod`、类型转换和权重计算，本次实现引入了单轴映射结构：

```cpp
struct AxisInterpolationMap
{
    int low = 0;
    int high = 0;
    double w_low = 1.0;
    double w_high = 0.0;
};
```

并通过 `build_axis_interpolation_map(src_size, dst_size)` 预先为 `x`、`y`、`z` 三个方向分别构造：

- 下界索引 `low`
- 上界索引 `high`
- 下界权重 `w_low`
- 上界权重 `w_high`

这样在主插值循环中只需要直接查表，而不需要重复计算坐标映射关系。

### 5.2 周期边界处理

原实现使用 `fmod` 做周期回绕。本次实现中使用：

```cpp
frac -= std::floor(frac / period) * period;
```

这样能够在负值情况下稳定回绕到合法范围，避免边界附近出现下标截断错误。

### 5.3 OpenMP 并行化

最终实现中，先分别构造：

```cpp
const std::vector<AxisInterpolationMap> x_map = build_axis_interpolation_map(nx_read, nx);
const std::vector<AxisInterpolationMap> y_map = build_axis_interpolation_map(ny_read, ny);
const std::vector<AxisInterpolationMap> z_map = build_axis_interpolation_map(nz_read, nz);
```

然后在主循环上进行 OpenMP 并行：

```cpp
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
for (int ix = 0; ix < nx; ix++)
{
    for (int iy = 0; iy < ny; iy++)
    {
        ...
    }
}
```

在每个 `(ix, iy)` 上，先预计算四个角点基址与四个二维权重：

- `idx_x0y0`
- `idx_x1y0`
- `idx_x0y1`
- `idx_x1y1`

以及：

- `w00`
- `w10`
- `w01`
- `w11`

然后在 `iz` 上完成 z 方向插值并写回输出数组。

## 6. SIMD 向量化评估

在最内层 `iz` 循环中加入了：

```cpp
#ifdef _OPENMP
#pragma omp simd
#endif
for (int iz = 0; iz < nz; iz++)
{
    ...
}
```

这样做的原因是：

1. `iz` 维没有跨迭代依赖；
2. `out_row[iz]` 连续写出；
3. `z_map[iz]` 连续访问；
4. 编译器可以据此尝试自动向量化。

实验表明，`omp simd` 在该循环上是可行的，数值结果和参考实现一致，但收益有限，说明该内核的主要瓶颈仍然偏向访存而不是纯计算。

## 7. 内存访问模式分析

### 7.1 输出访问

输出数组按：

```cpp
data_out[(ix * ny + iy) * nz + iz]
```

写入，因此最内层 `iz` 对应连续内存写出，cache 友好。

### 7.2 输入访问

每个输出点需要读取 8 个邻近顶点值。本实现先固定 `(ix, iy)`，只在 `iz` 上移动，因此对输入数组的访问表现为 4 条沿 z 方向的连续读流，整体访问规律较好。

### 7.3 cache 影响

本次最小性能基准中设计了两类工作集：

- `l3_fit`：`64^3 -> 128^3`，输入+输出约 `18.00 MiB`
- `l3_exceed`：`96^3 -> 192^3`，输入+输出约 `60.75 MiB`

由于测试机器的 L3 cache 为 `35.8 MiB`，因此：

- `l3_fit` 基本可以落在 L3 范围内；
- `l3_exceed` 超过 L3 容量，更容易受到主存带宽影响。

测试结果显示，当工作集超过 L3 后，2 线程相对 1 线程的收益略有提升，但总体仍不高，说明当前环境下主要瓶颈受单核 SMT、带宽和调度开销限制。

## 8. 正确性验证

### 8.1 参考实现对比

在 `rho_io_test.cpp` 中新增了独立参考实现，用于逐点比较 OpenMP 版本与参考串行版本的输出误差。核心要求为：

```cpp
EXPECT_LT(max_err, 1e-6);
```

实际测试中，误差满足要求。

### 8.2 覆盖的测试场景

补充的测试包括：

1. `TrilinearInterpolate`
   - 使用真实 `chg.cube` 数据；
   - 比较并行实现与参考实现的最大误差。

2. `TrilinearInterpolateConstantFieldBoundary`
   - 使用常量场；
   - 验证退化维度和边界情况下结果保持不变。

3. `TrilinearInterpolateDifferentGridSizes`
   - 覆盖多组升采样和降采样尺寸组合；
   - 验证不同规模输入下的正确性。

4. `TrilinearInterpolateBoundarySamples`
   - 对网格边缘、角点等样本进行显式检查；
   - 验证周期回绕和边界插值行为。

### 8.3 基准程序中的正确性复核

在最小 benchmark 中，也对两个大规模 case 做了参考实现比较，得到最大误差：

- `l3_fit`：`2.13e-14`
- `l3_exceed`：`4.26e-14`

远小于题目要求的 `1e-6`。

## 9. 性能测试

### 9.1 测试环境

- CPU：`Intel(R) Xeon(R) Platinum 8269CY @ 2.50GHz`
- 逻辑 CPU：`2`
- 物理核心：`1`
- SMT：`2` 路
- L1d：`32 KiB`
- L2：`1 MiB`
- L3：`35.8 MiB`

需要说明的是，当前环境只有 2 个逻辑 CPU，因此 4/8 线程测试主要用于观察 oversubscription，不代表真实多核扩展能力。

### 9.2 运行方式

最小 benchmark 复现命令如下：

```bash
g++ -O3 -march=native -std=c++17 source/source_io/test_serial/trilinear_bench.cpp \
    -o source/source_io/test_serial/trilinear_bench_serial
g++ -O3 -march=native -std=c++17 -fopenmp \
    source/source_io/test_serial/trilinear_bench.cpp \
    -o source/source_io/test_serial/trilinear_bench_omp_simd
g++ -O3 -march=native -std=c++17 -fopenmp -DTRILINEAR_DISABLE_SIMD \
    source/source_io/test_serial/trilinear_bench.cpp \
    -o source/source_io/test_serial/trilinear_bench_omp_nosimd
OMP_PROC_BIND=true OMP_PLACES=threads ./source/source_io/test_serial/trilinear_bench_omp_simd --threads 1,2,4,8
```

### 9.3 测试结果

| Case | 工作集 | 配置 | 1 线程 | 2 线程 | 4 线程 | 8 线程 |
|---|---:|---|---:|---:|---:|---:|
| `l3_fit` (`64^3 -> 128^3`) | `18.00 MiB` | OpenMP + SIMD | `8.582 ms` | `8.360 ms` (`1.027x`) | `8.592 ms` | `8.560 ms` |
| `l3_fit` (`64^3 -> 128^3`) | `18.00 MiB` | OpenMP + noSIMD | `8.767 ms` | `8.514 ms` (`1.030x`) | `8.470 ms` | `8.770 ms` |
| `l3_fit` (`64^3 -> 128^3`) | `18.00 MiB` | 纯串行 | `8.735 ms` | - | - | - |
| `l3_exceed` (`96^3 -> 192^3`) | `60.75 MiB` | OpenMP + SIMD | `32.383 ms` | `28.271 ms` (`1.145x`) | `29.482 ms` | `31.551 ms` |
| `l3_exceed` (`96^3 -> 192^3`) | `60.75 MiB` | OpenMP + noSIMD | `34.103 ms` | `29.344 ms` (`1.162x`) | `31.265 ms` | `29.837 ms` |
| `l3_exceed` (`96^3 -> 192^3`) | `60.75 MiB` | 纯串行 | `29.407 ms` | - | - | - |

### 9.4 性能结果分析

1. 在 `l3_fit` 场景下，2 线程相对 1 线程仅有很小提升，说明工作集虽然落在 L3 内，但当前平台仍受单核资源和线程调度限制。
2. 在 `l3_exceed` 场景下，2 线程收益提升到约 `1.145x`，说明当工作集超出 L3 后，额外线程在一定程度上能隐藏访存等待。
3. 超过 2 个逻辑线程后，4/8 线程没有稳定收益，反而因为 oversubscription 带来上下文切换和调度开销。
4. `OpenMP + SIMD` 对比 `OpenMP + noSIMD` 有小幅收益，但不是决定性优化，表明本内核更偏 memory-bound。

## 10. 可扩展性与重构

本次实现中已经把单轴插值映射抽取为：

- `AxisInterpolationMap`
- `build_axis_interpolation_map(src_size, dst_size)`

这为后续扩展提供了一个最小参数化接口，具体体现在：

1. `x/y/z` 三个方向共用同一套映射逻辑；
2. 当前 `trilinear_interpolate` 的外部接口无需改变；
3. 未来若要实现更高阶插值，可以在保留并行框架不变的前提下，替换单轴邻域生成和权重计算策略；
4. 精度评估也可以继续沿用“参考实现对比 + 最大误差统计”的框架扩展。

## 11. 遇到的问题

1. 原始实现中存在重复计算坐标映射和权重的问题，影响性能。
2. 直接并行最内层循环并不合适，任务粒度过细，开销偏大。
3. 当前环境缺少 `ZLIB` 开发依赖，因此仓库级完整 `cmake + ctest` 回归未全部跑通。
4. 当前机器只有 1 个物理核心，线程扩展结果不能代表真实多核服务器上的加速上限。

## 12. 总结

本次实验完成了 `trilinear_interpolate` 的 OpenMP 并行化改造，并在不改变函数接口和输出布局的前提下，实现了：

- 基于 `collapse(2)` 的二维外层并行；
- 基于 `omp simd` 的最内层向量化提示；
- 基于单轴映射预计算的重复计算消除；
- 基于参考实现的数值正确性验证；
- 基于最小 benchmark 的线程扩展、SIMD 和 cache/访存分析。

实验结果表明，该实现能够保证与参考实现数值一致，最大误差远小于 `1e-6`。在当前单核双线程环境下，并行加速效果有限，但已验证该并行方案正确、可运行，并为更高核数平台上的进一步优化和更高阶插值扩展打下了基础。
