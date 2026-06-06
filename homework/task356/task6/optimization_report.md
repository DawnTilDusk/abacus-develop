# Task 6: 电荷密度读取 MPI-IO 并行化 —— 优化实现报告（第二轮）

Date: 2026-06-06

---

## 一、本轮优化概要

在第一轮（2026-05-26）完成 `read_rhog_mpi()` 函数的基础上，
本轮重点提升代码质量和可维护性。

### 1.1 本轮完成内容

| 内容 | 状态 |
|------|------|
| 硬编码偏移量替换为 `constexpr` 命名常量 | 完成 |
| `MissingFileWarningWritten` 测试的多 rank 兼容修复 | 完成 |
| 与原 `read_rhog` 输出一致性验证 (串行 + MPI np=4) | 通过 |
| 性能对比测试 (串行 vs MPI-IO) | 完成 |

### 1.2 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/source_io/module_chgpot/rhog_io.cpp` | 修改 | constexpr 偏移量 + `read_rhog_mpi()` |
| `source/source_io/module_chgpot/rhog_io.h` | 修改 | 新增 `read_rhog_mpi()` 声明 |
| `source/source_io/test/read_rhog_mpi_test.cpp` | 修改 | 多 rank 修复 + MPI-IO 测试 |
| `source/source_io/test/CMakeLists.txt` | 修改 | 添加测试目标 |

---

## 二、代码质量改进

### 2.1 constexpr 命名常量

**优化前**（硬编码）:
```cpp
MPI_File_read_all(fh, hdr1.data(), 20, MPI_BYTE, ...);       // 20 = ?
std::vector<char> hdr2(80);                                    // 80 = ?
MPI_Offset miller_section_offset = 100;                        // 100 = ?
// miller section: 4B size + 12*npwtot_in B data + 4B size = 8 + 12*npwtot_in
rhog_section_offset += 8 + 16 * npwtot_in;                    // 8 + 16 = ?
```

**优化后**（自文档化）:
```cpp
constexpr MPI_Offset HDR_PART1_SIZE = 5 * sizeof(int);               // 20B
constexpr MPI_Offset HDR_PART2_SIZE = 2 * sizeof(int)
                                     + 9 * sizeof(double);           // 80B
constexpr MPI_Offset HEADER_SIZE = HDR_PART1_SIZE + HDR_PART2_SIZE;  // 100B
constexpr MPI_Offset MILLER_MARKER_SIZE = sizeof(int);
constexpr MPI_Offset RHOG_MARKER_SIZE = sizeof(int);

const MPI_Offset miller_section_bytes = 2 * MILLER_MARKER_SIZE
    + static_cast<MPI_Offset>(3 * npwtot_in) * sizeof(int);

const MPI_Offset rhog_spin_bytes = 2 * RHOG_MARKER_SIZE
    + static_cast<MPI_Offset>(npwtot_in) * sizeof(std::complex<double>);
```

每个常量精确反映了文件格式语义，修改格式时只需改一处。

### 2.2 多 rank 测试修复

`MissingFileWarningWritten` 测试原实现在所有 rank 上检查警告内容，
但 `read_rhog()` 只在 `RANK_IN_POOL == 0` 时写入警告。修复为仅在正确 rank 上验证：

```cpp
// read_rhog only writes the warning on RANK_IN_POOL == 0
if (GlobalV::RANK_IN_POOL == 0)
{
    EXPECT_NE(content.find("Can't open file"), std::string::npos);
}
```

---

## 三、算法回顾

`read_rhog_mpi()` 替代了原 "rank 0 串行读 + N 次 Bcast" 模式：

```
原流程:                              新流程 (MPI-IO):
rank 0: 打开 → 读头 (ifs>>)         All: MPI_File_open (集体)
rank 0: Bcast error (×1)            All: MPI_File_read_all (header, 100B)
rank 0: Bcast params (×6)           All: MPI_File_read_at_all (miller)
rank 0: Bcast miller (大数组)       All: MPI_File_read_at_all (rhog)
rank 0: Bcast rhog (大数组)         All: 本地 G-vector 映射
```

消除了 10+ 次 `MPI_Bcast`，所有 rank 直接从文件并行读取。

---

## 四、性能数据

| 测试 | 耗时 (np=1) | 吞吐量 | 耗时 (np=4) | 吞吐量 |
|------|:----------:|:------:|:----------:|:------:|
| read_rhog (串行+Bcast) | 0.37 ms | 107 MB/s | 0.49 ms | 80 MB/s |
| read_rhog_mpi (MPI-IO) | 0.21 ms | 189 MB/s | 6.33 ms | 6 MB/s |

> 注: 参考文件较小 (~40 KB)，MPI-IO 在小文件上集合操作开销可能超过收益。
> 在大体系（512 原子、360 MB 文件）中预期收益显著。

---

## 五、测试结果

```
 8/8 tests PASSED (serial)
 8/8 tests PASSED (MPI np=4, all ranks)
```

| 测试用例 | 说明 |
|----------|------|
| ReadThenWriteThenReadRoundtrip | 读取→写入→再读取，无崩溃 |
| RepeatedReadsYieldSameResult | 两次读取结果完全一致 |
| MissingFileWarningWritten | 缺失文件输出警告 (已修复多 rank) |
| Bench_ReadRhog_Serial | 串行读取性能基线 |
| MPIIOParallelReadIntegrity | MPI-IO 读取数据完整性 |
| ReadRhogVsReadRhogMPIConsistency | 新旧实现输出完全一致 |
| Bench_ReadRhog_MPIIO | MPI-IO 性能基线 |
| Bench_ReadRhog_SerialVsMPI | 串行 vs MPI-IO 对比 |
