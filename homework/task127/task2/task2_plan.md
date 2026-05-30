## 题目二九阶段执行计划
设计原则： 每个阶段只动尽可能少的东西，结束时立刻有一个可跑的测试 。一旦失败回滚成本极低。

### 阶段 1：基线锁定（不写代码，只跑测试）
目的 ：把"题目一已经实现的现状"作为本次题目二的对照基线。

动作 ：

- 在不改动任何代码、不改 CMake 的情况下，编译并跑：
  - MODULE_ESTATE_charge_mpi_test
  - MODULE_ESTATE_charge_mpi_test_4np
- 把 reduce_diff_pools 内部加一对 ModuleBase::timer 包围（如果题目一里没加），记录 np=1/2/4 下的耗时作为"OpenMP 关闭"基线。
测试方式 ：测试全过 + 基线时间记录在报告里。

回滚成本 ：无代码改动。

### 阶段 2：编译开关 & 测试 target 隔离
目的 ：解决 test_mpi/CMakeLists.txt:7 显式关闭 OpenMP 的问题， 但不要直接删那一行 ，否则题目一现有测试可能被并发问题搞乱。

动作 （推荐方案 B）：

- 方案 A（粗暴）：删掉 remove_definitions(-D_OPENMP) 。简单但题目一也跟着开 OpenMP。
- ✅ 方案 B（推荐）：保留这一行， 新增 一个独立的 OpenMP 测试 target，例如：
  - 让题目二的循环正确性 / 性能测试都跑在这个 target 上。
  - 题目一原 target 不动，保证回归。
测试方式 ：cmake reconfigure 通过 + 空壳 charge_omp_test.cpp （只有 main + MPI_Init ）能编译运行。

回滚成本 ：只加 CMake 段落和一个空文件，删除即可。

### 阶段 3：循环依赖分析（不写代码，只写文档）
目的 ：题目要求项 1（循环依赖分析），同时作为后续 #pragma omp 选择策略的依据。

动作 ：在题目二的 markdown 报告里写出：

- reorder_pool_rank_to_uniform （ charge_mpi.cpp:46-57 ）
  - 写下标： nz*ir + startz[ip] + iz ，对固定 ip 互不重叠 → 无写写依赖
  - 读下标：纯读 → 无 RAW 依赖
  - 结论： ir 、 iz 两层完全独立可并行；外层 ip 之间也无重叠（rank 切 z 分块）
- extract_uniform_to_local （ charge_mpi.cpp:59-70 ）
  - 写下标： numz[rank]*ir + iz ，互不重叠 → 同样安全
- 决定策略：对 ir 用 #pragma omp parallel for ，可选 collapse(2) 包 ir+iz （因 numz[ip] 在该函数里是常量）。
测试方式 ：无代码改动，只校验文档内容。

### 阶段 4：OpenMP 加到 reorder_pool_rank_to_uniform
目的 ：题目要求项 2（OpenMP 实现）。 只动一个函数 。

动作 ：在 charge_mpi.cpp:49 上面加一行：

- 注意用 #ifdef _OPENMP 包，没开 OpenMP 时一切如旧。
- 不动任何调用方（题目一阶段四的 Waitsome 路径会自然受益）。
测试方式 ：

1. 关 OpenMP： MODULE_ESTATE_charge_mpi_test + _4np 全过（确认题目一未坏）。
2. 开 OpenMP： MODULE_ESTATE_charge_omp_test 跑一个最小用例验证 reorder_pool_rank_to_uniform 串行 vs 并行结果完全一致。
回滚成本 ：删一段 pragma。

### 阶段 5：OpenMP 加到 extract_uniform_to_local
目的 ：与阶段 4 对称的另一处循环。 只动一个函数 。

动作 ：在 charge_mpi.cpp:62 加 #pragma omp parallel for collapse(2) ，同样 #ifdef _OPENMP 包裹。

测试方式 ：与阶段 4 相同，再加一个 extract_uniform_to_local 串并行对比测试。

### 阶段 6：补单元测试（题目要求项 4、5）
目的 ：满足题目要求"线程安全的数据生成器、不同规模边界、串并行一致"。

动作 ：在新建的 charge_omp_test.cpp 中写：

1. 生成器 ：用 std::mt19937 + 固定 seed 确保可重复（线程不参与生成）。
2. 测试 A：串并行一致
   - 构造 rhopw （参考 charge_mpi_test.cpp:75-79 ）。
   - 准备两份相同输入 array_tot 。
   - 一份用 OMP_NUM_THREADS=1 、另一份用 OMP_NUM_THREADS=4 各跑一遍 reorder_pool_rank_to_uniform 。
   - EXPECT_EQ 逐元素相等（整数式重排，浮点零误差，可以用 EXPECT_EQ 而非 _NEAR ）。
3. 测试 B：边界 ncxy=1
   - 用极小网格（ initgrids 用够小的截断），断言不崩 + 结果正确。
4. 测试 C：边界 numz=[0,1] 之类不均匀分布
   - 通过 mock rhopw->numz / startz 数组直接构造极端切片，断言并行版本仍然正确。
5. 测试 D：extract_uniform_to_local 同样的三组测试 。
测试方式 ： ctest -R MODULE_ESTATE_charge_omp_test 全过。

### 阶段 7：性能测试 + Amdahl 分析（题目要求项 3）
目的 ：交报告里要的"1/2/4/8 线程加速比 + Amdahl 拟合"。

动作 ：

- 用阶段 6 的固定大网格输入。
- 用 ModuleBase::timer 包住 reorder_pool_rank_to_uniform 与 extract_uniform_to_local ，分别记录 N 次平均耗时。
- 在外部脚本里循环：
- 用 Amdahl 公式 S(p) = 1 / ((1-f) + f/p) 反推可并行部分占比 f ，写进报告。
测试方式 ：跑分输出表格 + 写入报告。

### 阶段 8（加分项）：抽象 DataTransformFunc
目的 ：题目要求项 6。 重点：以"叠加"形式新增，不替换原有调用链 ，避免破坏题目一阶段四的 Waitsome 路径。

动作 ：

- 在 charge.h （第 156 行附近的注释块后）声明：
- 在 charge_mpi.cpp 文件末尾新增 parallel_data_rearrange(...) 模板函数，让它 内部调用 reorder_pool_rank_to_uniform 或者直接用 transform lambda。
- 关键：不要删除/替换 reorder_pool_rank_to_uniform ，让题目一阶段四的 Waitsome 流程继续使用旧函数。新抽象只是"另一种实现的演示"。
- 加一个测试：用 parallel_data_rearrange + serial_transform lambda 实现结果与 reorder_pool_rank_to_uniform 结果完全一致。
测试方式 ：新加 TEST_F(*, DataTransformAbstraction) 通过。

回滚成本 ：删除新增声明 + 函数即可，原代码不变。

### 阶段 9：回归验证（最终交付前）
目的 ：保证题目一题目二同时通过，且 OpenMP 开关都可控。

动作矩阵 ：

配置 跑的测试 期望 OpenMP 关 + np=1 全部 mpi 测试 全过 OpenMP 关 + np=4 *_4np 全过（题目一不退化） OpenMP 开 + np=1 + threads=4 MODULE_ESTATE_charge_omp_test 全过 OpenMP 开 + np=4 + threads=2 同上 + mpi 测试 全过（MPI+OpenMP 混合）

测试方式 ：跑完上面 4 组组合后即可提交。

## 给你的执行节奏建议
- 必做最小集 ：阶段 1-6 + 9。能拿题目 80% 分。
- 建议加上 ：阶段 7（性能曲线，分数大头之一）。
- 行有余力 ：阶段 8（加分项）。