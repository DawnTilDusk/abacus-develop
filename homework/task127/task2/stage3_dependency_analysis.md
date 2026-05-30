# 阶段 3：循环依赖分析

## 1. 目的

题目二要求项 1 是「循环依赖分析」。本阶段不写代码，只通过对下标表达式的代数分析，证明题目二要并行化的三层嵌套循环可以无锁、无 race 地并行执行，并在此基础上**选定** OpenMP 的具体并行策略（哪一层并行、是否 `collapse`、是否需要 `reduction`）。

## 2. 分析目标

题目二涉及的「三层嵌套循环」实际上分布在 [charge_mpi.cpp](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp) 的两个函数中：

- 函数 A：`Charge::reorder_pool_rank_to_uniform`
  - 位置：[charge_mpi.cpp:46-57](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L46-L57)
  - 自身是 `ir × iz` 的二层循环
  - 外层由 `Charge::reorder_pool_to_uniform`（[charge_mpi.cpp:38-44](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L38-L44)）或题目一阶段四的 `gather_pool_data_nonblocking` 提供 `ip` 维 → 一起构成「三层嵌套」
- 函数 B：`Charge::extract_uniform_to_local`
  - 位置：[charge_mpi.cpp:59-70](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L59-L70)
  - 二层循环 `ir × iz`（rank 是隐含的，固定为 `GlobalV::RANK_IN_POOL`）

## 3. 函数 A：`reorder_pool_rank_to_uniform`

### 3.1 原始代码

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

### 3.2 符号说明

- `ncxy = nx * ny`：一个 z 平面上的格点数；常量（在循环内不变）。
- `nz`：全局 z 方向格点总数；常量。
- `numz[ip]`：rank `ip` 在 z 方向上分到的层数；对**当前**调用 `ip` 是常量。
- `startz[ip]`：rank `ip` 在全局 z 方向上的起始平面；对当前 `ip` 是常量。
- 输入 `array_tot`：只读。
- 输出 `array_tot_aux`：仅写，不读，不累加。

### 3.3 下标表达式

记一次内层迭代 `(ir, iz)` 的写入下标和读出下标为：

- **写入下标**：`W(ir, iz) = nz * ir + startz[ip] + iz`
- **读出下标**：`R(ir, iz) = numz[ip] * ir + startz[ip] * ncxy + iz`

注意 `0 ≤ iz < numz[ip]`，`0 ≤ ir < ncxy`。

### 3.4 写–写依赖分析

设有两次不同迭代 `(ir1, iz1) ≠ (ir2, iz2)`，问：

$$
W(ir_1, iz_1) = W(ir_2, iz_2) \quad?
$$

即：

$$
nz \cdot ir_1 + iz_1 = nz \cdot ir_2 + iz_2
$$

因为 `0 ≤ iz < numz[ip] ≤ nz`（每个 rank 拿到的 z 切片不可能超过总 z 长度），所以 `iz1, iz2 ∈ [0, nz)`。在「以 `nz` 为模」的离散视角下：

- `iz1 ≡ iz2 (mod nz)` ⇒ `iz1 = iz2`
- 进而 `ir1 = ir2`

矛盾。所以**不存在不同迭代写到同一个内存位置 ⇒ 无写–写依赖**。

### 3.5 读–写依赖（RAW / WAR）分析

- `array_tot` 只读，`array_tot_aux` 只写；二者是**两个不同的缓冲区**（在调用栈上分别对应 `chgmpi_tot_` 与 `chgmpi_tot_aux_`，由 `Charge::init_chgmpi()` 分别 `new[]`）。
- 因此根本不存在跨迭代的 R-after-W 或 W-after-R 关系。

### 3.6 外层 `ip` 维的安全性（与「三层并行」相关）

题目示例代码（[10_charge.md:585-592](file:///root/homework/abacus-develop/10_charge.md#L585-L592)）希望在 `ip × ir × iz` 三层上做 `collapse(2)`，所以同样要确认**不同 `ip` 之间**写入区段是否重叠。

考察不同 `ip = p` 与 `ip = q (p ≠ q)`：

- `ip = p` 写入区段：`{nz * ir + startz[p] + iz | 0 ≤ ir < ncxy, 0 ≤ iz < numz[p]}`
- `ip = q` 写入区段：`{nz * ir + startz[q] + iz | 0 ≤ ir < ncxy, 0 ≤ iz < numz[q]}`

ABACUS 在 z 方向上做 rank 切片的硬约束是：

```
startz[p] + numz[p] = startz[p+1],  且各段不重叠
```

也就是说对**同一个 `ir`**，不同 `ip` 写入的 z 偏移区间 `[startz[ip], startz[ip] + numz[ip])` 两两不相交。再叠加上不同 `ir` 间步长为 `nz` 的跳跃也不会跨入相邻 rank 的区间。结论：**不同 `ip` 之间也无写–写冲突**。

### 3.7 函数 A 结论

| 依赖类型 | 是否存在 | 备注 |
|---|---|---|
| 同一 `ip` 内 `ir × iz` 间写–写 | ❌ 无 | 由 `W(ir,iz)` 单射性证明 |
| 同一 `ip` 内 RAW / WAR | ❌ 无 | 两个独立缓冲区 |
| 不同 `ip` 间写–写 | ❌ 无 | `startz[ip]` 分段无重叠 |
| reduce 累加（需要 reduction） | ❌ 无 | 纯赋值，非累加 |

⇒ 三层 `ip × ir × iz` 完全 **embarrassingly parallel**。

## 4. 函数 B：`extract_uniform_to_local`

### 4.1 原始代码

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

### 4.2 下标

记 `M = numz[RANK_IN_POOL]`（当前 rank 在 z 上的层数，循环内常量），`s = startz_current`（当前 rank 起始 z，循环内常量）：

- **写入下标**：`W(ir, iz) = M * ir + iz`，`0 ≤ ir < ncxy`，`0 ≤ iz < M`
- **读出下标**：`R(ir, iz) = nz * ir + s + iz`

### 4.3 写–写依赖

`W(ir1, iz1) = W(ir2, iz2)` 即 `M * ir1 + iz1 = M * ir2 + iz2`，由于 `iz ∈ [0, M)`，按 `mod M` 唯一性得 `iz1 = iz2`、`ir1 = ir2`。⇒ **无写–写依赖**。

### 4.4 RAW / WAR

`array_tot` 只读、`array_rho` 只写，二者属于不同缓冲区（前者是题目一新增的成员 `chgmpi_tot_`，后者是 `Charge::rho[is]` 之类的本地 slab）。⇒ **无 RAW / WAR**。

### 4.5 函数 B 结论

`ir × iz` 两层完全独立可并行；不存在外层 `ip`，所以 `collapse(2)` 是最自然选择。

## 5. OpenMP 并行策略选型

基于第 3、4 节的结论，并结合**题目一阶段四的调用方式**，确定如下策略：

### 5.1 函数 A：`reorder_pool_rank_to_uniform`

候选方案：

| 方案 | 并行维度 | 是否破坏题目一阶段四调用方式 |
|---|---|---|
| A1 | 在函数内对 `ir` 加 `#pragma omp parallel for` | 不破坏 |
| A2 | 在函数内对 `ir + iz` 加 `#pragma omp parallel for collapse(2)` | 不破坏 |
| A3 | 在调用方（`reorder_pool_to_uniform`）上对 `ip + ir` 加 `collapse(2)` | **会破坏** —— 题目一阶段四的 Waitsome 路径**每次只调一个 `ip`**，根本没有外层 `ip` 循环 |

→ 选 **A2（首选）** 或 **A1（保守）**。

原因：

- 与题目一阶段四的「按块到达即重排」语义对齐：阶段四会针对单个 rank 调用一次 `reorder_pool_rank_to_uniform`，函数内部的并行收益最大化。
- 不依赖外层 `ip`，对调用方零侵入。
- `collapse(2)` 合法：内层上界 `numz[ip]` 在函数体内对**当前调用**是常量，满足 OpenMP 「rectangular nested loop」要求。

### 5.2 函数 B：`extract_uniform_to_local`

仅有 `ir × iz` 两层，且内层上界 `numz[RANK_IN_POOL]` 在函数体内是常量，**直接 `collapse(2)`**。

### 5.3 调度子句

- `schedule(static)`：每次迭代工作量近似（都是一次赋值），静态切分负载均衡足够好，且没有调度开销。
- 不需要 `reduction`、`atomic`、`critical`：因为没有累加，没有冲突写。
- 不需要 `private` 子句：内层下标变量是循环计数器，OpenMP 自动私有；其他都是 `const`。

### 5.4 编译宏保护

所有 `#pragma omp` 必须用 `#ifdef _OPENMP` 包起来。这样：

- 在题目一的 target（`-D_OPENMP` 已 `remove_definitions`）下，pragma 段消失 → 行为与基线完全一致；
- 在题目二的 target（保留 `-D_OPENMP`）下，pragma 段生效 → 进入并行路径。

## 6. 计划写入的代码形态（阶段 4/5 的模板）

> 仅作为阶段 3 选型结论的展示，代码本体留到阶段 4/5 真正落地。

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
- 提前把 `numz[ip]`、`startz[ip]`、`nz` 取到栈上局部常量，便于 OpenMP 内层上界推断。
- `collapse(2)`，`schedule(static)`。
- `array_tot_aux`、`array_tot` 自动是 `shared`（C++ 默认共享），无需显式声明。

`extract_uniform_to_local` 同理，只是把 `numz[ip]` 换成 `numz[RANK_IN_POOL]`，`startz[ip]` 换成 `startz_current`。

## 7. 与题目一兼容性的最终断言

题目二的 OpenMP 改动**完全嵌在函数体内**，对调用方接口（参数、返回值、外部可见行为）不做任何更改。因此：

- 题目一阶段四的 `gather_pool_data_nonblocking` → `MPI_Waitsome` → 按 rank 调用 `reorder_pool_rank_to_uniform` 的链路**逐字节不变**，只是单次调用内部走得更快。
- 题目一阶段四的 `extract_uniform_to_local` 调用同理。
- 在题目一 target 下（`_OPENMP` 被 `remove_definitions`）pragma 全部消失，串行行为完全一致。

阶段 3 通过 ✅ —— 可以进入阶段 4（落地 `reorder_pool_rank_to_uniform` 的 OpenMP 化）。

## 8. 阶段 3 决策清单（速查表）

| 决策点 | 结论 |
|---|---|
| 函数 A 并行维度 | 函数内 `ir × iz` `collapse(2)` |
| 函数 B 并行维度 | `ir × iz` `collapse(2)` |
| 调度 | `schedule(static)` |
| reduction | 不需要 |
| 临界区 / atomic | 不需要 |
| 编译宏保护 | `#ifdef _OPENMP` 包整段 pragma |
| 是否触碰调用方 | 否 |
| 是否触碰题目一现有 target | 否 |
