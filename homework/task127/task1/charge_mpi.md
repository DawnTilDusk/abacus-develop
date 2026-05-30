**文件定位**

- 目标文件：[charge\_mpi.cpp](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L1-L162)
- 配套头文件：[charge.h](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge.h#L126-L176)
- 这份文件只做一件核心事情：**把不同 pool / band 上分散的电荷密度** **`rho`** **或动能密度** **`kin_r`** **做 MPI 归约**。

**4 个核心概念**

- `rho[is]`：某个自旋通道在**当前进程本地**持有的实空间电荷密度数组。
- `nrxx`：当前进程本地拥有的网格点数。
- `nxyz`：整个全局三维网格点总数。
- `rhopw->numz[ip] / startz[ip]`：第 `ip` 个 pool 内进程在 z 方向上分到多少层、从哪一层开始。

**整体结构**

- `init_chgmpi()`：准备 `MPI_Allgatherv` 需要的接收长度和偏移。
- `reduce_diff_pools()`：真正执行“跨 pool 归约”的核心函数。
- `rho_mpi()`：对所有 `rho[is]` 调用归约。
- `kin_r_mpi()`：对所有 `kin_r[is]` 调用归约。

***

**头文件部分**

- [charge\_mpi.cpp:L1-L8](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L1-L8)
- `#include "charge.h"`
  - 引入 `Charge` 类定义。
  - 你后面会看到的 `rho`、`kin_r`、`nrxx`、`rhopw`、`rec`、`dis` 都在这里声明。
- `#include "source_base/global_function.h"`
  - 提供一些全局辅助函数或宏，项目里常见基础设施。
- `#include "source_base/global_variable.h"`
  - 引入 `GlobalV::KPAR`、`GlobalV::NPROC_IN_POOL`、`GlobalV::RANK_IN_POOL` 这些全局并行变量。
- `#include "source_base/parallel_comm.h"`
  - 引入 `KP_WORLD`、`POOL_WORLD`、`BP_WORLD`、`INT_BGROUP` 这些 MPI 通信器。
- `#include "source_base/parallel_reduce.h"`
  - 虽然本文件里主要直接用了 MPI API，但它属于归约相关基础设施。
- `#include "source_base/timer.h"`
  - 用来给函数计时，便于性能分析。
- `#include "source_hamilt/module_xc/xc_functional.h"`
  - 用来判断是否启用了 `KED`，也就是是否要处理 `kin_r`。
- `#include "source_io/module_parameter/parameter.h"`
  - 用来访问全局输入参数，比如 `nspin`、`bndpar`、`out_elf`。
- `#ifdef __MPI`
  - 整个文件只有在 MPI 编译打开时才生效。
  - 也就是说，这个文件是“并行版 Charge 通信逻辑”。

***

**函数一：`init_chgmpi()`**

- 代码位置：[charge\_mpi.cpp:L10-L26](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L10-L26)

**函数作用**

- 这个函数不是做归约本身。
- 它的作用是：**给后面** **`MPI_Allgatherv`** **准备两个数组**
  - `rec[ip]`：每个进程要接收多少个元素
  - `dis[ip]`：这些元素在接收缓冲区中的起始偏移

**逐行讲解**

- `void Charge::init_chgmpi()`
  - `Charge` 的成员函数。
- `if (KP_WORLD == MPI_COMM_NULL)`
  - 只有当 `KP_WORLD` 不存在时，才走这里。
  - 这说明：如果没法直接通过 `KP_WORLD` 做跨 pool 归约，就要走“先 gather 再 reduce”的备用路径。
- `delete[] rec;`
- `rec = new int[GlobalV::NPROC_IN_POOL];`
  - 先释放旧数组，再分配新数组。
  - 长度等于“每个 pool 内有多少进程”。
- `delete[] dis;`
- `dis = new int[GlobalV::NPROC_IN_POOL];`
  - 同理，分配偏移数组。
- `const int ncxy = this->rhopw->nx * this->rhopw->ny;`
  - 这里把一个 z 平面上的点数记成 `ncxy = nx * ny`。
  - 后面所有 z 分块长度都要乘它。
- `for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ip++)`
  - 遍历当前 pool 内的每个进程。
- `rec[ip] = this->rhopw->numz[ip] * ncxy;`
  - 第 `ip` 个进程负责 `numz[ip]` 个 z 平面。
  - 每个 z 平面有 `ncxy` 个点。
  - 所以它总共有 `numz[ip] * ncxy` 个数据点。
  - 这就是 `Allgatherv` 中这个进程发来的数据量。
- `dis[ip] = this->rhopw->startz[ip] * ncxy;`
  - 第 `ip` 个进程负责的数据在“按 z 连续铺开”的全局数组里，从第 `startz[ip]` 个 z 平面开始。
  - 因此偏移是 `startz[ip] * ncxy`。
- 这一段结束后，`rec/dis` 就描述清楚了“pool 内各进程的数据如何拼成一个完整三维块”。

**这一段和习题一的关系**

- 这是备用路径的前置准备。
- 习题一里如果你要优化 fallback 分支，这两个数组是关键。

***

**函数二：`reduce_diff_pools(double* array_rho)`**

- 代码位置：[charge\_mpi.cpp:L28-L118](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L28-L118)
- 这是全文件最重要的函数。

**函数作用**

- 把不同 pool 上的同一物理量做求和归约。
- 输入 `array_rho` 是**当前进程本地的一段数据**。
- 输出还是写回 `array_rho`，即**原地更新成本进程应该持有的归约结果**。

***

**函数开头**

- [charge\_mpi.cpp:L30-L31](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L30-L31)
- `ModuleBase::TITLE("Charge", "reduce_diff_pools");`
  - 打印标题或登记当前进入的模块函数。
- `ModuleBase::timer::start("Charge", "reduce_diff_pools");`
  - 开始计时。
  - 这个计时结果就是你做性能分析时很值得观察的地方。

***

**路径 A：`KP_WORLD`** **存在，直接 Allreduce**

- [charge\_mpi.cpp:L32-L35](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L32-L35)

**逐行讲解**

- `if (KP_WORLD != MPI_COMM_NULL)`
  - 如果存在跨 pool 的统一通信器，就走快路径。
- `MPI_Allreduce(MPI_IN_PLACE, array_rho, this->nrxx, MPI_DOUBLE, MPI_SUM, KP_WORLD);`
  - 对 `array_rho` 做原地全归约。
  - `MPI_IN_PLACE` 表示输入和输出都在 `array_rho` 本身。
  - `this->nrxx` 表示当前进程手里这一段本地数据长度。
  - `MPI_SUM` 表示把不同 pool 的对应元素加起来。
  - `KP_WORLD` 表示通信范围是“跨 pool 对应 rank 的进程们”。

**这里的物理含义**

- 如果所有 pool 的数据分布方式一致，那么“每个 pool 中 rank 相同的进程”持有的是相同位置那一块网格。
- 那就可以直接把这些块对齐做 `Allreduce`。
- 这是最简单、最干净的情况。

**这一段的问题**

- 这里是**阻塞式** `MPI_Allreduce`。
- 所有进程要等这次归约彻底完成。
- 这正是习题一的一个核心优化点。

***

**路径 B：`KP_WORLD`** **不存在，走回退逻辑**

- [charge\_mpi.cpp:L36-L112](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L36-L112)

这段最难，我给你按“小块”拆开。

***

**先申请三个大数组**

- [charge\_mpi.cpp:L38-L40](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L38-L40)
- `double* array_tmp = new double[this->rhopw->nxyz];`
- `double* array_tot = new double[this->rhopw->nxyz];`
- `double* array_tot_aux = new double[this->rhopw->nxyz];`

**它们分别想干什么**

- `array_tmp`
  - 当前进程临时发送缓冲区。
- `array_tot`
  - gather 之后，或者 reduce 之后的全局数组。
- `array_tot_aux`
  - 为了跨 pool 做统一格式重排而准备的辅助数组。

**注意**

- 这里一次性分配了 3 个 `nxyz` 大小的数组。
- 对大体系来说很重。
- 这是性能瓶颈之一，也是内存占用瓶颈之一。

***

**先把本地数据缩放一遍**

- [charge\_mpi.cpp:L44-L47](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L44-L47)
- `for (int ir = 0; ir < this->rhopw->nrxx; ++ir)`
- `array_tmp[ir] = array_rho[ir] / GlobalV::NPROC_IN_POOL;`

**这是什么意思**

- 把当前进程本地的 `array_rho` 拷到 `array_tmp`。
- 顺便除以 `NPROC_IN_POOL`。
- 它的设计目的通常是为了补偿后面 pool 内 gather 和跨 pool reduce 的组合方式，避免重复计数。

**你现在只要先记住**

- `array_tmp` 是从 `array_rho` 复制出来的本地版本。
- 它不是全局数组，只有前 `nrxx` 个位置有效。

***

**在 pool 内 gather，拼出本 pool 的完整数据**

- [charge\_mpi.cpp:L48-L49](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L48-L49)
- `MPI_Allgatherv(array_tmp, this->rhopw->nrxx, MPI_DOUBLE, array_tot, rec, dis, MPI_DOUBLE, POOL_WORLD);`

**这句非常关键**

- `array_tmp`：当前进程发出的本地块
- `this->rhopw->nrxx`：当前块大小
- `array_tot`：所有块拼起来之后的缓冲区
- `rec/dis`：告诉 MPI 每个进程发多少、放在哪里
- `POOL_WORLD`：只在当前 pool 内 gather

**执行结果**

- gather 完后，`array_tot` 里就有了“本 pool 的完整电荷密度数据”，但是仍然是**本地 pool 的布局顺序**。

**问题点**

- 不同 pool 可能布局顺序不一致。
- 所以后面还不能直接跨 pool reduce。

***

**计算** **`ncxy`**

- [charge\_mpi.cpp:L50](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L50)
- `const int ncxy = this->rhopw->nx * this->rhopw->ny;`
- 还是一个 z 平面的点数。

***

**三层循环：把本 pool 的布局重排成统一布局**

- [charge\_mpi.cpp:L51-L91](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L51-L91)

这是整个文件最值得你画图的地方。

**三层循环的含义**

- 第一层 `ip`
  - 遍历 pool 内所有进程
- 第二层 `ir`
  - 遍历每个 z 平面上的平面内索引
- 第三层 `iz`
  - 遍历该进程负责的 z 层数

**核心赋值语句**

- [charge\_mpi.cpp:L87-L88](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L87-L88)

```cpp
array_tot_aux[this->rhopw->nz * ir + this->rhopw->startz[ip] + iz]
    = array_tot[this->rhopw->numz[ip] * ir + this->rhopw->startz[ip] * ncxy + iz];
```

这句你必须吃透。

**先看左边**

- `array_tot_aux[this->rhopw->nz * ir + this->rhopw->startz[ip] + iz]`
- 这是在构造一种“统一格式”的数组。
- 它的意思是：
  - 先固定平面内位置 `ir`
  - 再沿 z 方向连续放
- 所以这个布局相当于：
  - 对每个 `ir`，它在 z 方向上有 `nz` 个值，连续排开

**再看右边**

- `array_tot[this->rhopw->numz[ip] * ir + this->rhopw->startz[ip] * ncxy + iz]`
- 这里表示的是当前 pool 内原本的局部布局。
- 它按照“每个进程自己那段 z 数据块”的方式存放。

**这一步本质上在干嘛**

- 把“每个 pool 自己内部的分布格式”
- 变成“所有 pool 都一致的标准格式”

**为什么必须这么做**

- 因为只有把各个 pool 的数据重排成同样的顺序
- 才能在后面直接跨 pool 做逐元素求和

**注释那一大段在说什么**

- [charge\_mpi.cpp:L57-L86](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L57-L86)
- 其实核心就是一句：
  - 不同 pool 内部的数据排列方式可能不同，所以要先重排成一个统一的、不依赖 z 划分方式的标准布局，再做归约。

**这一段的问题**

- 三层循环，访存不连续。
- 非常容易 cache 不友好。
- 这也是题目 2 想让你优化的地方。
- 但对习题一来说，这里也是“通信前的重排开销”。

***

**跨 pool 做真正的归约**

- [charge\_mpi.cpp:L93-L97](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L93-L97)
- `MPI_Allreduce(array_tot_aux, array_tot, this->rhopw->nxyz, MPI_DOUBLE, MPI_SUM, INT_BGROUP);`

**这句的含义**

- 现在 `array_tot_aux` 已经是所有 pool 都能对齐的统一布局了。
- 就可以在 `INT_BGROUP` 这个通信器上，对整块 `nxyz` 长度的全局数组做逐元素求和。
- 求和结果放到 `array_tot`。

**你可以把它理解成**

- 前面的三层循环是在“对齐坐标系”
- 这一句才是“真正跨 pool 求和”

**问题**

- 又是阻塞式 `MPI_Allreduce`
- 而且一次归约的是整块 `nxyz`
- 数据量很大

***

**把统一布局的结果再拷回本地布局**

- [charge\_mpi.cpp:L98-L108](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L98-L108)
- 这一步是“从统一格式还原成当前进程的局部格式”
- 外层 `ir` 遍历平面内点
- 内层 `iz` 遍历当前 rank 在 pool 内负责的 z 层

**核心赋值**

- [charge\_mpi.cpp:L105-L106](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L105-L106)

```cpp
array_rho[this->rhopw->numz[GlobalV::RANK_IN_POOL] * ir + iz]
    = array_tot[this->rhopw->nz * ir + this->rhopw->startz_current + iz];
```

**左边**

- 当前进程本地数组 `array_rho` 的局部布局

**右边**

- 统一格式下，当前进程对应那段 z 区间的数据

**这一段执行完之后**

- 当前进程本地的 `array_rho` 就被更新成“已经跨 pool 求和后的正确结果”

***

**释放临时数组**

- [charge\_mpi.cpp:L109-L111](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L109-L111)
- `delete[] array_tot_aux;`
- `delete[] array_tot;`
- `delete[] array_tmp;`

**说明**

- 这三个大数组只在 fallback 路径里临时使用。
- 每次调用都申请、释放一次，开销不小。

***

**如果 band 并行也开了，还要再做一次归约**

- [charge\_mpi.cpp:L113-L116](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L113-L116)
- `if(PARAM.globalv.all_ks_run && PARAM.inp.bndpar > 1)`
  - 如果是 `all_ks_run` 模式，并且 band 并行数大于 1
- `MPI_Allreduce(MPI_IN_PLACE, array_rho, this->nrxx, MPI_DOUBLE, MPI_SUM, BP_WORLD);`
  - 再在 `BP_WORLD` 上对当前本地块做一次归约

**这是什么意思**

- 说明除了 k-point pool 维度以外，band 维度上也可能有重复贡献需要合并。
- 因此会多做一次通信。

***

**函数结束计时**

- [charge\_mpi.cpp:L117](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L117)
- `ModuleBase::timer::end("Charge", "reduce_diff_pools");`

***

**函数三：`rho_mpi()`**

- 代码位置：[charge\_mpi.cpp:L120-L140](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L120-L140)

**函数作用**

- 对所有自旋通道的 `rho` 做归约。
- 如果启用了 `kin_r` 相关功能，也顺便把 `kin_r` 一起归约。

**逐行讲解**

- `ModuleBase::TITLE("Charge", "rho_mpi");`
  - 记录当前函数名。
- `if (GlobalV::KPAR * PARAM.inp.bndpar <= 1)`
  - 如果 `KPAR * bndpar <= 1`
  - 说明压根没有跨 pool/band 的并行需求
  - 那就没必要 MPI 归约，直接返回
- `ModuleBase::timer::start("Charge", "rho_mpi");`
  - 开始计时。
- `for (int is = 0; is < PARAM.inp.nspin; ++is)`
  - 遍历每个自旋通道
- `reduce_diff_pools(this->rho[is]);`
  - 对当前自旋的电荷密度做归约
- `if (XC_Functional::get_ked_flag() || PARAM.inp.out_elf[0] > 0)`
  - 如果启用了 KED，或者要求输出 ELF
- `reduce_diff_pools(this->kin_r[is]);`
  - 那么对对应的动能密度也做归约
- `ModuleBase::timer::end("Charge", "rho_mpi");`
  - 结束计时

**怎么理解这个函数**

- `rho_mpi()` 不是实现细节函数
- 它是一个调度函数
- 真正干活的是 `reduce_diff_pools()`

***

**函数四：`kin_r_mpi()`**

- 代码位置：[charge\_mpi.cpp:L142-L160](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L142-L160)

**函数作用**

- 单独对 `kin_r` 做归约。
- 逻辑和 `rho_mpi()` 很像，但这里只处理 `kin_r`。

**逐行讲解**

- `ModuleBase::TITLE("Charge", "kin_r_mpi");`
  - 打印函数名
- `if (GlobalV::KPAR * PARAM.inp.bndpar <= 1)`
  - 没并行就直接返回
- `ModuleBase::timer::start("Charge", "kin_r_mpi");`
  - 开始计时
- `if (XC_Functional::get_ked_flag() || PARAM.inp.out_elf[0] > 0)`
  - 只有真的需要 `kin_r` 时才执行
- `for (int is = 0; is < PARAM.inp.nspin; ++is)`
  - 遍历自旋通道
- `reduce_diff_pools(this->kin_r[is]);`
  - 调归约
- `ModuleBase::timer::end("Charge", "kin_r_mpi");`
  - 结束计时

**为什么有了** **`rho_mpi()`** **还要有** **`kin_r_mpi()`**

- 因为有的场景可能只需要 `kin_r` 的归约，不需要整套 `rho_mpi()` 流程。
- 属于接口层面的分离。

***

- [charge\_mpi.cpp](file:///root/homework/abacus-develop/source/source_estate/module_charge/charge_mpi.cpp#L1-L162) 的核心思路是：
  - 如果 pool 之间布局一致，就直接 `MPI_Allreduce`
  - 如果不一致，就：
    - pool 内 `Allgatherv`
    - 做一次统一布局重排
    - 跨 pool `Allreduce`
    - 再拷回本地布局
  - 如果 band 并行也参与，还要再额外归约一次

***

- `reduce_diff_pools()` 里多次使用**阻塞式** `MPI_Allreduce`
- fallback 路径每次都会分配 3 个 `nxyz` 大数组
- fallback 路径有一次 `Allgatherv` + 一次 `Allreduce`
- 中间有三层循环做数据重排，访存模式不够友好
- 最后还可能再来一次 `BP_WORLD` 归约

***

流程图

```
当前进程本地的电荷密度
array_rho
  含义：
  - 只是一小段本地 z-slab
  - 长度 = nrxx = numz[RANK_IN_POOL] * nxy

    |
    | 1. 复制并缩放
    |    for ir: array_tmp[ir] = array_rho[ir] / 
    NPROC_IN_POOL
    v

array_tmp
  含义：
  - 当前进程准备发给 pool 内其他进程的数据
  - 仍然只是“本进程局部块”
  - 只前 nrxx 个元素有效

    |
    | 2. 在 POOL_WORLD 上做 MPI_Allgatherv
    |    把同一个 pool 内所有 rank 的局部块拼起来
    v

array_tot   (第一次出现)
  含义：
  - 当前 pool 的完整三维数据
  - 但排列方式还是“按 pool 内 rank 分块”的局部格式
  - 这时不同 pool 的 array_tot 不一定能直接逐元素相加

    |
    | 3. 三层循环重排
    |    array_tot_aux[...] = array_tot[...]
    |    把“局部格式”改成“全局统一格式”
    v

array_tot_aux
  含义：
  - 当前 pool 的完整三维数据
  - 但现在布局已经统一
  - 不再依赖当前 pool 内部怎么切 z
  - 所有 pool 在同一物理点上拥有相同的线性下标

    |
    | 4. 在 INT_BGROUP 上做 MPI_Allreduce(..., 
    MPI_SUM, ...)
    |    把不同 pool 对同一个网格点的贡献相加
    v

array_tot   (第二次作为输出缓冲区)
  含义：
  - 所有相关 pool 的贡献求和之后的“完整全局密度”
  - 仍然是统一格式
  - 这时每个元素都已经是正确的总和

    |
    | 5. 按当前 rank 的 z-slab 切回本地格式
    |    array_rho[...] = array_tot[...]
    v

array_rho
  含义：
  - 回到“当前进程本地只保留自己那一段”的形式
  - 但数值已经是跨 pool 归约后的正确结果
```

<br />

- array\_rho
  - 我手里只有自己负责的那一小块
- array\_tmp
  - 我先把自己的这一小块打包好
- array\_tot
  - 在一个 pool 里把大家的小块拼成完整图
- array\_tot\_aux
  - 但这个完整图的摆放方式还不统一，所以重排成统一格式
- array\_tot
  - 然后跨 pool 把同一个物理点的贡献相加
- array\_rho
  - 最后再把完整图里属于我的那一小块切回来

