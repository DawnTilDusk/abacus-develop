# 题目一算法文档：`reduce_diff_pools` 的流程分析与优化

## 1. 文档目标

本文档对应大作业题目一“MPI 归约操作的并行化”，目标是：

- 梳理 ABACUS 中电荷密度跨 pool 归约的原始流程
- 解释原始实现中的数据布局和通信逻辑
- 说明当前已完成的第二阶段、第三阶段优化
- 为后续测试、性能分析和报告撰写提供统一算法说明

本次分析的核心函数位于：

- `source/source_estate/module_charge/charge_mpi.cpp`
- 关键函数：`init_chgmpi()`、`reduce_diff_pools()`、`rho_mpi()`

***

## 2. 问题背景

在 ABACUS 的 k-point 并行模型中，不同 pool 分别负责不同 k 点的计算。\
每个 pool 计算得到的电荷密度只是总电荷密度的一部分贡献，因此在输出或进入后续流程前，需要把不同 pool 的结果进行归约求和。

对于 `reduce_diff_pools(double* array_rho)` 而言：

- 输入 `array_rho` 是当前进程持有的本地电荷密度片段
- 每个进程只拥有自己在 z 方向上的一个局部 slab
- 最终目标是把不同 pool 对相同物理网格点的贡献求和

当 `KP_WORLD != MPI_COMM_NULL` 时，不同 pool 中“对应 rank”持有的局部数据布局一致，因此可以直接调用 `MPI_Allreduce`。

当 `KP_WORLD == MPI_COMM_NULL` 时，不同 pool 的局部分布格式不能直接一一对应，于是必须走 fallback 路径：

1. 先在同一个 pool 内收集出完整数据
2. 再把数据重排为跨 pool 一致的统一布局
3. 再在不同 pool 之间做求和归约
4. 最后切回当前 rank 对应的本地布局

***

## 3. 原始算法流程

### 3.1 原始流程图

```text
当前进程本地的电荷密度
array_rho
  含义：
  - 只是一小段本地 z-slab
  - 长度 = nrxx = numz[RANK_IN_POOL] * nxy

    |
    | 1. 复制并缩放
    |    for ir: array_tmp[ir] = array_rho[ir] / NPROC_IN_POOL
    v

array_tmp
  含义：
  - 当前进程准备发给 pool 内其他进程的数据
  - 仍然只是“本进程局部块”
  - 只前 nrxx 个元素有效

    |
    | 2. 在 POOL_WORLD 上做 MPI_Allgatherv
    |    把同一个 pool 内所有 rank 的局部块拼起来
    v

array_tot   (第一次出现)
  含义：
  - 当前 pool 的完整三维数据
  - 但排列方式还是“按 pool 内 rank 分块”的局部格式
  - 这时不同 pool 的 array_tot 不一定能直接逐元素相加

    |
    | 3. 三层循环重排
    |    array_tot_aux[...] = array_tot[...]
    |    把“局部格式”改成“全局统一格式”
    v

array_tot_aux
  含义：
  - 当前 pool 的完整三维数据
  - 但现在布局已经统一
  - 不再依赖当前 pool 内部怎么切 z
  - 所有 pool 在同一物理点上拥有相同的线性下标

    |
    | 4. 在 INT_BGROUP 上做 MPI_Allreduce(..., MPI_SUM, ...)
    |    把不同 pool 对同一个网格点的贡献相加
    v

array_tot   (第二次作为输出缓冲区)
  含义：
  - 所有相关 pool 的贡献求和之后的“完整全局密度”
  - 仍然是统一格式
  - 这时每个元素都已经是正确的总和

    |
    | 5. 按当前 rank 的 z-slab 切回本地格式
    |    array_rho[...] = array_tot[...]
    v

array_rho
  含义：
  - 回到“当前进程本地只保留自己那一段”的形式
  - 但数值已经是跨 pool 归约后的正确结果
```

### 3.2 原始流程的直观理解

- `array_rho`
  - 当前 rank 手里只有自己负责的那一小块
- `array_tmp`
  - 先把这一小块拷到临时缓冲区，作为通信输入
- `array_tot`
  - 在一个 pool 内把所有 rank 的局部块拼成完整图
- `array_tot_aux`
  - 将 pool 内局部格式重排成跨 pool 一致的统一格式
- `array_tot`
  - 对不同 pool 的统一格式数据做逐元素求和
- `array_rho`
  - 再从完整结果中切回当前 rank 需要的局部部分

***

## 4. 原始实现的主要问题

原始 fallback 路径存在以下几个主要问题：

### 4.1 阻塞式通信

原始实现直接使用：

- `MPI_Allgatherv(...)`
- `MPI_Allreduce(...)`

这两步都是阻塞 collective，意味着当前通信器中的所有进程都需要同步等待。

### 4.2 临时大数组频繁申请和释放

在原始版本中，`reduce_diff_pools()` 每次调用时都会新建：

- `array_tmp`
- `array_tot`
- `array_tot_aux`

这三个数组都具有 `nxyz` 级别的大小。\
当 `rho_mpi()` 对多个自旋通道、多次 SCF 迭代反复调用该函数时，内存分配和释放本身就会带来额外开销。

### 4.3 数据重排和通信逻辑耦合

原始代码把：

- pool 内数据收集
- 局部格式到统一格式的重排
- 统一格式到本地格式的提取

全部写在一个大函数中，导致：

- 难以单独测试某一段布局转换是否正确
- 不利于后续将通信与计算重叠
- 不利于继续替换为其他通信模式

### 4.4 full-grid 缓冲区在每个进程中重复持有

无论是 `MPI_Allgatherv` 还是我们当前第三阶段的非阻塞收集实现，本质语义都还是“每个进程都拿到完整 `array_tot`”。\
因此：

- 每个进程都要为 `chgmpi_tot_` 和 `chgmpi_tot_aux_` 预留 `nxyz` 大小的内存
- 这会造成较高的总内存占用

这部分问题在当前第三阶段尚未彻底解决，属于后续可继续优化的方向。

***

## 5. 第二阶段优化：结构重构与缓冲区成员化

第二阶段的目标不是改变数学逻辑，而是把 fallback 路径拆清楚，使其更适合进一步优化。

### 5.1 主要修改

#### 5.1.1 将三个临时数组改为 `Charge` 成员缓冲区

新增成员：

- `chgmpi_tmp_`
- `chgmpi_tot_`
- `chgmpi_tot_aux_`

作用：

- 将原本函数内部的临时数组提升为对象成员
- 统一由 `Charge` 负责分配和释放
- 为后续真正的缓冲区复用打基础

#### 5.1.2 提取数据布局转换函数

从原来的大函数中拆出两个辅助函数：

- `reorder_pool_to_uniform(const double* array_tot, double* array_tot_aux) const`
- `extract_uniform_to_local(const double* array_tot, double* array_rho) const`

拆分后，fallback 路径可以更清晰地表达为：

```text
复制本地块 -> pool 内收集 -> 重排为统一布局 -> 跨 pool 归约 -> 提取本地块
```

### 5.2 第二阶段的意义

- 保持算法逻辑不变，方便和原始实现逐元素对比
- 将“通信逻辑”和“下标映射逻辑”解耦
- 降低后续引入非阻塞通信时的改动风险

***

## 6. 第三阶段优化：使用 `MPI_Irecv/MPI_Isend` 替代 `MPI_Allgatherv`

第三阶段的核心目标是：\
把 fallback 路径中 pool 内的阻塞式 `MPI_Allgatherv` 替换为显式的非阻塞点对点通信。

### 6.1 修改前后的对比

#### 原始实现

```cpp
MPI_Allgatherv(array_tmp, this->rhopw->nrxx, MPI_DOUBLE,
               array_tot, rec, dis, MPI_DOUBLE, POOL_WORLD);
```

#### 第三阶段实现

我们引入新函数：

- `gather_pool_data_nonblocking(const double* array_tmp, double* array_tot) const`

在 `reduce_diff_pools()` 中替代原来的 `MPI_Allgatherv`。

### 6.2 新算法的核心思想

虽然最终语义仍然是“每个进程都得到完整的 `array_tot`”，但通信实现方式变成了：

1. 每个 rank 先把自己的局部块本地拷贝到 `array_tot` 对应位置
2. 每个 rank 对 pool 内其他所有 rank 挂好 `MPI_Irecv`
3. 每个 rank 再向其他所有 rank 发起 `MPI_Isend`
4. 调用 `MPI_Waitall` 等待所有发送和接收完成

这样就把原本由 `MPI_Allgatherv` 隐式完成的“全员互相交换局部块”过程，显式地展开成了点对点通信。

### 6.3 第三阶段通信流程图

```text
每个进程：

array_tmp
  |
  | 1. 先把自己的局部块 memcpy 到 array_tot 对应偏移 dis[my_rank]
  v

array_tot (只填好了自己那一段)
  |
  | 2. 对其他 rank 逐个 MPI_Irecv
  |    接收目标地址：array_tot + dis[ip]
  v

未完成接收的 array_tot
  |
  | 3. 对其他 rank 逐个 MPI_Isend
  |    发送内容：array_tmp
  v

所有进程之间互相交换局部块
  |
  | 4. MPI_Waitall
  v

array_tot (完整)
```

### 6.4 伪代码

```cpp
copy local block to array_tot[dis[my_rank]]

for each rank ip != my_rank:
    MPI_Irecv(array_tot + dis[ip], rec[ip], ..., ip, POOL_WORLD)

for each rank ip != my_rank:
    MPI_Isend(array_tmp, rec[my_rank], ..., ip, POOL_WORLD)

MPI_Waitall(...)
```

### 6.5 第三阶段的意义

- 满足题目中“使用 `MPI_Irecv` / `MPI_Isend`”的要求
- 将 pool 内数据收集从黑盒 collective 变成可控的显式通信过程
- 为后续进一步做 `MPI_Waitsome`、实现“边接收边重排”提供接口基础

### 6.6 第三阶段的局限性

第三阶段虽然完成了非阻塞点对点通信改造，但仍有两个限制：

#### 6.6.1 还没有真正实现通信与计算重叠

当前版本最后仍然调用的是：

- `MPI_Waitall`

因此流程仍然是：

1. 发起所有 `Irecv/Isend`
2. 等待全部完成
3. 再执行后续重排

也就是说，第三阶段只是“非阻塞通信接口改造”，还不是“重叠执行”的最终版本。

#### 6.6.2 还没有减少 full-grid 缓冲区的总内存占用

当前实现依然属于 all-gather 语义：

- 每个进程最终都会得到完整的 `array_tot`
- 每个进程仍然持有 `chgmpi_tot_` 和 `chgmpi_tot_aux_`

因此 full-grid 缓冲区的内存冗余问题仍然存在。

***

## 7. 第四阶段优化：使用 `MPI_Waitsome` 实现按块接收与按块重排

第四阶段的核心目标是：\
在第三阶段“非阻塞收集”的基础上，不再等所有消息到齐后再统一重排，而是做到：

- 本地块准备好后立即重排
- 某个远端 rank 的数据块一到达，就立即重排这一块
- 使“接收通信”和“数据重排”发生部分重叠

### 7.1 第四阶段的核心思想

第三阶段的流程是：

1. 发起全部 `MPI_Irecv`
2. 发起全部 `MPI_Isend`
3. 调用 `MPI_Waitall`
4. 所有接收完成后，统一调用 `reorder_pool_to_uniform(...)`

第四阶段把它改成：

1. 先把本地 rank 的数据 `memcpy` 到 `array_tot`
2. 立即把本地 rank 这一块重排到 `array_tot_aux`
3. 对其他 rank 发起 `MPI_Irecv`
4. 对其他 rank 发起 `MPI_Isend`
5. 使用 `MPI_Waitsome` 等待一批接收完成
6. 某个 rank 的数据一旦到达，就立刻把该 rank 的局部块重排到 `array_tot_aux`
7. 全部接收完成后，再等待发送完成
8. 然后再进入 `INT_BGROUP` 上的 `MPI_Allreduce`

### 7.2 第四阶段引入的辅助函数

为了支持“按 rank 增量重排”，我们把原先整块重排函数进一步拆细，新增了：

- `reorder_pool_rank_to_uniform(const double* array_tot, double* array_tot_aux, int ip) const`

它只负责把某一个 `ip` 对应的局部块从 pool 内布局拷贝到统一布局。

于是：

- `reorder_pool_to_uniform(...)`
  - 表示“对所有 rank 逐个执行重排”
- `reorder_pool_rank_to_uniform(...)`
  - 表示“只重排某一个 rank 的局部块”

### 7.3 第四阶段流程图

```text
array_rho
  |
  | 1. 复制到 array_tmp，并做归约前缩放
  v

array_tmp
  |
  | 2. 本地 rank 数据 memcpy 到 array_tot 对应位置
  | 3. 本地 rank 数据立即重排到 array_tot_aux
  v

已包含本地块的 array_tot / array_tot_aux
  |
  | 4. 对其他 rank 挂 MPI_Irecv
  | 5. 对其他 rank 发 MPI_Isend
  v

通信进行中
  |
  | 6. MPI_Waitsome
  | 7. 哪个 rank 的接收完成，就立刻执行
  |    reorder_pool_rank_to_uniform(array_tot, array_tot_aux, ip)
  v

完整的统一布局 array_tot_aux
  |
  | 8. 在 INT_BGROUP 上做 MPI_Allreduce
  v

归约后的完整统一布局结果
  |
  | 9. 切回当前 rank 的本地 slab
  v

array_rho
```

### 7.4 第四阶段伪代码

```cpp
copy local array_rho to array_tmp

memcpy local block to array_tot + dis[my_rank]
reorder_pool_rank_to_uniform(array_tot, array_tot_aux, my_rank)

for each ip != my_rank:
    MPI_Irecv(array_tot + dis[ip], rec[ip], ..., ip, POOL_WORLD)

for each ip != my_rank:
    MPI_Isend(array_tmp, rec[my_rank], ..., ip, POOL_WORLD)

while there are unfinished receives:
    MPI_Waitsome(...)
    for each completed ip:
        reorder_pool_rank_to_uniform(array_tot, array_tot_aux, ip)

MPI_Waitall(all send requests)

MPI_Allreduce(array_tot_aux, array_tot, ..., INT_BGROUP)
extract_uniform_to_local(array_tot, array_rho)
```

### 7.5 第四阶段相对第三阶段的改进点

- 第三阶段是“非阻塞收集 + 统一等待 + 统一重排”
- 第四阶段是“非阻塞收集 + 分批完成检测 + 到达即重排”

因此第四阶段相对第三阶段的主要改进是：

- 减少“所有接收都结束之后才开始重排”的等待时间
- 将重排工作分散到通信过程中执行
- 更符合题目中“实现计算与通信重叠”的要求

### 7.6 第四阶段的局限性

虽然第四阶段已经实现了接收与重排的部分重叠，但仍有两个限制：

#### 7.6.1 `INT_BGROUP` 上的归约仍然是阻塞式

当前版本仍然保留：

- `MPI_Allreduce(array_tot_aux, array_tot, ..., INT_BGROUP)`

因此“跨 pool 的全局求和”仍然没有进一步异步化。

#### 7.6.2 full-grid 缓冲区仍然在每个进程重复持有

当前实现依然是 all-gather 语义：

- 每个进程都会收集完整的 `array_tot`
- 每个进程都会持有 `array_tot_aux`

因此内存冗余问题仍然存在，尚未进入 root-only gather/reduce/scatter 方案。

***

## 8. 当前算法总结

结合第二阶段和第三阶段，当前 fallback 路径可以总结为：

```text
输入：当前 rank 的局部电荷密度 array_rho

1. 将 array_rho 拷到 array_tmp，并除以 NPROC_IN_POOL
2. 本地 rank 数据先写入 array_tot，并立即重排到 array_tot_aux
3. 在 POOL_WORLD 内通过 `MPI_Irecv/MPI_Isend` 发起非阻塞点对点收集
4. 使用 `MPI_Waitsome` 等待一批接收完成；每当某个 rank 的数据到达，就立刻把该 rank 的局部块重排到 array_tot_aux
5. 在 INT_BGROUP 上对完整的统一布局 `array_tot_aux` 做 MPI_Allreduce，结果写回 array_tot
6. 从统一布局的完整结果中，提取当前 rank 对应的本地部分，写回 array_rho
7. 如果 all_ks_run 且 bndpar > 1，再在 BP_WORLD 上进行一次额外归约
```

***

## 9. 正确性说明

当前优化版本保持了原始算法的数学逻辑不变，仅替换了 pool 内完整数据收集的实现方式。

### 9.1 不变部分

- `array_rho -> array_tmp` 的输入输出关系不变
- `reorder_pool_to_uniform()` 的重排映射不变
- `INT_BGROUP` 上的 `MPI_Allreduce` 不变
- `extract_uniform_to_local()` 的结果提取逻辑不变

### 9.2 改变部分

- 原先由 `MPI_Allgatherv` 自动完成的“全员收集完整数组”
- 先由 `MPI_Irecv/MPI_Isend` 显式实现
- 再由 `MPI_Waitsome` 驱动按块重排

由于收集结果 `array_tot` 的内容和原始版本一致，因此整体数值结果保持不变。

***

## 10. 复杂度与性能分析

### 10.1 计算复杂度

- 数据复制：`O(nrxx)`
- 数据重排：`O(nxyz)`
- 本地结果提取：`O(nrxx)`

### 10.2 通信复杂度

#### 原始版本

- pool 内：一次 `MPI_Allgatherv`
- pool 间：一次 `MPI_Allreduce`

#### 当前第四阶段版本

- pool 内：每个进程发起 `nproc - 1` 个 `MPI_Irecv` 和 `nproc - 1` 个 `MPI_Isend`
- pool 内：使用 `MPI_Waitsome` 分批确认接收完成
- pool 间：一次 `MPI_Allreduce`

### 10.3 当前预期收益

当前版本的收益主要体现在：

- 将阻塞式收集改造成了显式非阻塞点对点通信
- 已经实现了接收与重排的部分重叠
- 通过成员缓冲区管理降低了代码耦合和重复分配释放开销

需要说明的是：\
第四阶段版本不一定在所有环境下都比 `MPI_Allgatherv` 更快，因为 MPI 内部可能已经对 collective 做了很强优化。\
因此最终是否获得更好的性能，需要通过实际测试给出结论。

***

## 11. 测试与验证方案

当前实现已经通过现有单元测试：

- `MODULE_ESTATE_charge_mpi_test`
- `MODULE_ESTATE_charge_mpi_test_4np`

建议提交报告时包含以下测试内容：

### 11.1 功能正确性测试

- 单进程与多进程结果一致
- 不同进程数下结果一致
- `reduce_diff_pools()` 输出与原始版本逐元素对比

### 11.2 性能测试

- 进程数：`1, 2, 4, 8, 16`
- 记录 `reduce_diff_pools` 的耗时
- 对比原始版本、第三阶段版本和第四阶段版本

### 11.3 边界情况测试

- 极小网格
- `nspin = 1` 和 `nspin = 2`
- `numz` 分布不均匀时的情况

***

## 12. 后续可继续优化的方向

### 12.1 第五阶段：改造为 root-only gather/reduce/scatter

当前版本的 full-grid 缓冲区在每个进程中都重复存在。\
未来可以进一步考虑：

- pool 内只让 root 收集完整数据
- root 之间做归约
- 再由 root 把本地块发回给各 rank

这样可以显著降低 total memory footprint，但也会带来新的权衡：

- root 会成为 gather、重排、reduce、scatter 的集中热点
- 通信与计算负载会从“分散到各 rank”变成“集中到少数 root”
- 在 pool 内进程较多时，时间开销不一定优于当前版本

因此，本作业当前阶段优先完成“非阻塞通信 + `MPI_Waitsome` 重叠”这条优化主线，暂不继续推进 root-only 方案；后续如需进一步降低内存占用，可将其作为独立优化方向，再结合实际测试结果评估时间与空间的权衡。

***

### 12.2 第六阶段：尝试异步化 `INT_BGROUP` 上的归约

当前版本中，pool 内收集已经进行了非阻塞化与重排重叠，但：

- `INT_BGROUP` 上的 `MPI_Allreduce`
- `BP_WORLD` 上的附加归约

仍然是阻塞式的。\
未来可进一步研究是否能够引入：

- `MPI_Iallreduce`

从而继续缩短跨 pool 求和阶段的等待时间。

***

## 13. 结论

本次题目一的算法优化目前已经完成三个阶段：

- 第二阶段：完成结构重构与成员缓冲区管理
- 第三阶段：完成基于 `MPI_Irecv/MPI_Isend` 的 pool 内非阻塞收集实现
- 第四阶段：完成基于 `MPI_Waitsome` 的按块接收与按块重排

当前版本已经做到：

- 保持数值结果与原始实现一致
- 用显式点对点非阻塞通信替代原始 `MPI_Allgatherv`
- 在 pool 内实现了接收与重排的部分重叠
- 通过单元测试验证功能正确性

虽然当前版本尚未彻底解决 full-grid 缓冲区重复占用问题，也尚未实现真正的通信与计算重叠，但已经形成了一个清晰、可扩展、可继续优化的中间版本，可作为题目一的阶段性成果提交。
