# Task 6 第二轮优化：代码质量 + 多 rank 测试修复

Date: 2026-06-06

## 做了什么

1. **硬编码替换为 constexpr**：文件偏移量全部改为命名常量，自文档化
2. **多 rank 测试修复**：`MissingFileWarningWritten` 在 MPI 多 rank 下只 RANK_IN_POOL=0 写警告，断言改为仅在正确 rank 上检查

## 代码对比

```cpp
// 第一轮 (2026-05-26)
MPI_File_read_all(fh, hdr1.data(), 20, MPI_BYTE, ...);    // 20 = ?
MPI_Offset miller_section_offset = 100;                      // 100 = ?
rhog_section_offset += 8 + 16 * npwtot_in;                   // 8 + 16 = ?

// 第二轮 (2026-06-06)
constexpr MPI_Offset HDR_PART1_SIZE = 5 * sizeof(int);              // 20B
constexpr MPI_Offset HEADER_SIZE = HDR_PART1_SIZE + HDR_PART2_SIZE;  // 100B
const MPI_Offset rhog_spin_bytes = 2 * RHOG_MARKER_SIZE
    + static_cast<MPI_Offset>(npwtot_in) * sizeof(std::complex<double>);
```

## 测试结果

- 8/8 PASSED (串行)，8/8 PASSED (MPI np=4, 全 rank)
- 修复前 MPI 下 MissingFileWarningWritten 在 rank 1-3 失败，修复后全通过
