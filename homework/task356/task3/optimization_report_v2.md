# Task 3 第二轮优化：压缩集成 + 格式自动检测

Date: 2026-06-06

## 做了什么

1. **压缩集成到 write_cube**：新增 `compress` / `compress_nthreads` 参数（默认 false，向后兼容）
2. **read_cube 格式自动检测**：文本 / ZCMP(v0压缩) / ZCM2(v1压缩) / CIPM(MPI二进制) 四种格式自动识别
3. **提取独立压缩模块** `charge_compress.h/.cpp`，供写入/读取两侧复用

## 性能对比

| 256³ 写入 | 第一轮 (文本) | 第二轮 (MPI-IO 二进制) | 提升 |
|-----------|:----------:|:-------------------:|:----:|
| 耗时 | 24,476 ms | **394 ms** | **62x** |
| 吞吐量 | 5.3 MB/s | **327 MB/s** | **62x** |

> 第一轮数据来自 2026-05-22 基线测试；第二轮增加了 MPI-IO 并行写入。

## 测试结果

- 10/10 PASSED (串行)，10/10 PASSED (MPI np=4)
- 新增：MPIWriteBinaryCubeAndReadBack、TextVsBinaryConsistency、Bench_WriteCube_MPIIO_Binary_256
