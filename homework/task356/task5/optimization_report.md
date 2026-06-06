# Task 5: 电荷密度数据压缩 —— 优化实现报告（第二轮）

Date: 2026-06-06

---

## 一、本轮优化概要

在第一轮（2026-05-26）完成串行 zlib + OpenMP 并行压缩（v0 格式）的基础上，
本轮引入 v1 线格式，解决格式检测歧义，补齐 OpenMP 并行解压，添加压缩自动回退。

### 1.1 本轮完成内容

| 内容 | 状态 |
|------|------|
| v1 线格式: "ZCM2" magic + format type byte + flags byte | 完成 |
| 自动回退: 压缩膨胀时自动存储原始数据 | 完成 |
| OpenMP 并行解压 (`decompress_charge_data_omp` + v1 OMP 解压) | 完成 |
| 通用解压 `decompress_charge_data_any()` 自动检测 v0/v1/serial/OMP | 完成 |
| v0 向后兼容 (ZCMP magic 保持不变) | 完成 |
| 完整测试 (20 个，全部通过) | 通过 |

### 1.2 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/source_io/module_output/charge_compress.h` | 修改 | v1 常量和函数声明 |
| `source/source_io/module_output/charge_compress.cpp` | 修改 | v1 实现 + 并行解压 + 通用解压 |
| `source/source_io/module_output/write_cube.cpp` | 修改 | 升级为 v1 压缩调用 |
| `source/source_io/module_output/read_cube.cpp` | 修改 | 升级为 `decompress_charge_data_any()` |
| `source/source_io/test/charge_compression_test.cpp` | 修改 | v1 + OpenMP 解压 + 集成测试 |
| `source/source_io/test/CMakeLists.txt` | 修改 | 链接 charge_compress.cpp |

---

## 二、v1 线格式设计

### 2.1 格式对比

| 格式 | Header 大小 | 结构 |
|------|:-----------:|------|
| v0 serial | 12B | `[ZCMP 4B][count 8B][zlib payload]` |
| v0 OMP | 16B | `[ZCMP 4B][nthreads 4B][total_n 8B][chunk_size+data...]` |
| **v1 serial** | **14B** | `[ZCM2 4B][fmt=1 1B][flags 1B][total_n 8B][payload]` |
| **v1 OMP** | **16B** | `[ZCM2 4B][fmt=2 1B][flags 1B][nthreads 2B][total_n 8B][chunk_size+data...]` |

### 2.2 v1 改进点

1. **精确格式识别**: `fmt_type` 字节（V1_FMT_SERIAL=1, V1_FMT_OMP=2），消除 v0 中
   通过 nthreads 字段区分 serial/OMP 的歧义（当 total_n 的前 2 字节恰好 > 1 时会误判）

2. **自动回退**: flags bit 0 (`COMPRESS_FLAG_COMPRESSED`):
   - 设为 1 → payload 是 zlib 压缩数据
   - 设为 0 → payload 是原始数据（当压缩并不减少体积时自动使用）

3. **通用解压**: `decompress_charge_data_any()` 单一入口，自动检测:
   ```
   magic == ZCM2? → fmt_type == V1_FMT_OMP? → decompress_v1_omp()
                    fmt_type == V1_FMT_SERIAL? → decompress_v1()
   magic == ZCMP? → nthreads > 1? → decompress_charge_data_omp()
                    else → decompress_charge_data()
   ```

### 2.3 OpenMP 并行解压

```
1. 串行预扫描: 解析所有 chunk 的 (offset, compressed_size)
2. #pragma omp parallel for
   → 每个线程独立解压自己的 chunk 到 dst 的不重叠区域
3. raw 模式同样并行 memcpy
```

---

## 三、性能数据

### 3.1 压缩性能基线

| 网格 | 原始大小 | 压缩后 | 耗时 | 吞吐量 |
|------|----------|--------|------|--------|
| 64³ | 2.0 MB | 0.9 MB | 64 ms | 31.4 MB/s |
| 128³ | 16.0 MB | 10.3 MB | 739 ms | 21.7 MB/s |
| 256³ | 128.0 MB | 97.2 MB | 5,913 ms | 21.7 MB/s |

### 3.2 压缩率

| 数据模式 | 原始 | 压缩后 | 压缩率 |
|----------|------|--------|:------:|
| 全零 | 2.0 MB | 0.00 MB | 0.1% |
| 常数 | 2.0 MB | 0.00 MB | 0.1% |
| 平滑高斯 | 2.0 MB | 0.89 MB | 44.7% |
| 随机 | 2.0 MB | 1.91 MB | 95.4% |

### 3.3 OpenMP 扩展性 (128³)

| 线程数 | 耗时 | 吞吐量 | 加速比 |
|:------:|------|--------|:------:|
| 1 | 651 ms | 24.6 MB/s | 1.00x |
| 2 | 345 ms | 46.4 MB/s | 1.89x |
| 4 | 267 ms | 59.8 MB/s | 2.43x |
| 8 | 254 ms | 63.1 MB/s | 2.57x |

### 3.4 v1 自动回退验证

对随机数据（不可压缩），v1 自动存储原始数据而非膨胀后的压缩结果，确保输出始终 ≤ 原始大小。

---

## 四、测试结果

```
20/20 tests PASSED (serial)
20/20 tests PASSED (MPI np=4)
```

| 类别 | 测试用例 | 说明 |
|------|----------|------|
| 正确性 | CompressDecompressRoundtrip_Small | 10000 随机元素 roundtrip |
| 正确性 | AllZerosCompressesWell | 100000 零值压缩率 < 10% |
| 正确性 | ConstantDataCompressesWell | 100000 常数压缩率 < 10% |
| 正确性 | RandomDataRoundtrip | 50000 随机 roundtrip |
| 正确性 | SmoothGaussianDataRoundtrip | 20³ 模拟电荷密度 |
| 格式验证 | WireFormatValidatesMagic | magic + count 字段验证 |
| 格式验证 | DecompressRejectsBadMagic | 错误 magic 正确拒绝 |
| 格式验证 | DecompressRejectsWrongCount | count 不匹配正确拒绝 |
| 格式验证 | RejectsTooSmallBuffer | 缓冲区过小正确拒绝 |
| OpenMP | OmpCompressDecompressRoundtrip | OMP 压缩解压 roundtrip |
| OpenMP | OmpCompressSameAsSerial | OMP vs 串行结果一致 |
| OpenMP | OmpAllZerosCompressesWell | OMP 全零压缩 |
| 集成 | WriteCubeWithCompressionRoundtrip | write → read 完整链路 |
| 集成 | AutoDetectCompressedFile | 压缩/未压缩文件自动检测 |
| 性能 | Bench_Compress_{Small,Medium,Large} | 64³/128³/256³ 基线 |
| 性能 | Bench_CompressRatioByPattern | 4 种模式压缩率 |
| 性能 | Bench_Compress_OMP_nthreads4_256 | OMP 4 线程 256³ |
| 性能 | Bench_Compress_OMP_Scaling_128 | OMP 扩展性 1/2/4/8 线程 |
