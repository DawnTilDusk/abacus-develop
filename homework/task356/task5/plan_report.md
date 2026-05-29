# Task 5: 电荷密度数据压缩 —— 优化方案报告

Date: 2026-05-26

---

## 一、当前实现分析

### 1.1 现有代码状态

**原型代码位置：** `source/source_io/test/charge_compression_test.cpp` (427 行)

压缩/解压函数以 `static` 函数形式存在于测试文件中：

```cpp
// Lines 54-78: 压缩函数原型
static bool compress_charge_data(const double* src, size_t n, std::vector<uint8_t>& dst)

// Lines 80-100: 解压函数原型
static bool decompress_charge_data(const uint8_t* src, size_t src_len,
                                   double* dst, size_t n)
```

### 1.2 当前 Wire Format

```
[magic: 4B "ZCMP"] [original_count: 8B uint64] [zlib compressed payload]
 Total: 12 字节头部 + 压缩数据
```

- Magic 值: `0x504D435A` ("ZCMP" 小端序)
- 压缩库: zlib `compress2()` / `uncompress()`
- 压缩级别: `Z_BEST_COMPRESSION`

### 1.3 测试状态

**13/13 测试全部通过 (已存在于 test 文件中)：**

| 类别 | 测试名称 | 状态 |
|------|---------|------|
| 正确性 | `CompressDecompressRoundtrip_Small` | PASS |
| 正确性 | `AllZerosCompressesWell` | PASS |
| 正确性 | `ConstantDataCompressesWell` | PASS |
| 正确性 | `RandomDataRoundtrip` | PASS |
| 正确性 | `SmoothGaussianDataRoundtrip` | PASS |
| 正确性 | `WireFormatValidatesMagic` | PASS |
| 正确性 | `DecompressRejectsBadMagic` | PASS |
| 正确性 | `DecompressRejectsWrongCount` | PASS |
| 正确性 | `RejectsTooSmallBuffer` | PASS |
| 性能 | `Bench_Compress_Small_64` | PASS |
| 性能 | `Bench_Compress_Medium_128` | PASS |
| 性能 | `Bench_Compress_Large_256` | PASS |
| 性能 | `Bench_CompressRatioByPattern` | PASS |

### 1.4 性能基线（串行 zlib, Z_BEST_COMPRESSION）

| 网格 | 原始大小 | 压缩后 | 耗时 (ms) | 吞吐量 (MB/s) | 压缩率 |
|------|----------|--------|-----------|---------------|--------|
| 64^3 | 2.0 MB | 0.9 MB | 173.94 | 11.50 | 44.7% |
| 128^3 | 16.0 MB | 10.3 MB | 2060.88 | 7.76 | ~64% |
| 256^3 | 128.0 MB | 97.2 MB | 17130.35 | 7.47 | ~76% |

**压缩率按数据模式 (64^3)：**

| 模式 | 压缩率 | 说明 |
|------|--------|------|
| 全零 | 0.1% | 真空区域，几乎不占空间 |
| 常数 | 0.1% | 均匀区域 |
| 平滑高斯 | 44.7% | 模拟真实电荷密度 |
| 随机 | 95.4% | 最坏情况，几无压缩效果 |

### 1.5 当前问题

1. **代码位置不正确**：压缩/解压在测试文件中以 `static` 存在，无法被 `write_cube` 或 `read_cube` 调用
2. **无 OpenMP 并行**：串行压缩对于 256^3 (128 MB) 需 17 秒，可并行化
3. **未集成到 I/O 流程**：压缩模块独立存在，未与实际的 cube/rhog 读写流程打通

---

## 二、优化算法设计

### 2.1 第一步：提取压缩模块

将 `compress_charge_data` 和 `decompress_charge_data` 从测试文件提取为独立模块：

```
source/source_io/module_output/charge_compress.h   — 对外接口声明
source/source_io/module_output/charge_compress.cpp — 实现
```

### 2.2 第二步：OpenMP 并行压缩

**算法：分块并行压缩 (Chunked Parallel Compression)**

```
Algorithm: compress_charge_data_omp(src, n, dst, nthreads)

Input:
  src        — 原始 double 数组 (n 个元素)
  n          — 元素总数
  nthreads   — OpenMP 线程数

Output:
  dst        — 压缩后数据 (wire format)

Steps:

1. 将 src 均匀分为 nthreads 个 chunk
   chunk_size = n / nthreads

2. 每个线程独立压缩自己的 chunk:
   #pragma omp parallel for
   for t = 0..nthreads-1:
     chunk_start = t * chunk_size
     chunk_n = chunk_size (最后一个线程可能多出余数)
     compress2(chunk_buf[t], ..., src + chunk_start, chunk_n * sizeof(double), ...)

3. 写入 wire format:
   [magic: 4B "ZCMP"]
   [nthreads: 4B uint32]           ← 新增：线程数
   [total_n: 8B uint64]            ← 原始总元素数
   for t = 0..nthreads-1:
     [chunk_compressed_size: 4B]   ← 压缩后大小
     [chunk_compressed_data]       ← 变长
```

**新的 Wire Format (OpenMP 版本)：**

```
┌──────────────────────────────────────────────────────────────────┐
│  magic "ZCMP"  nthreads  total_n  chunk0_size  chunk0_data ... │
│    4B         4B       8B       4B           variable           │
└──────────────────────────────────────────────────────────────────┘
```

**解压算法：**

```
Algorithm: decompress_charge_data_omp(src, src_len, dst, n)

1. 校验 magic "ZCMP"
2. 读取 nthreads, total_n, 校验 total_n == n
3. pos = 16 (跳过 header)
4. for t = 0..nthreads-1:
     chunk_size = read_u32(src + pos); pos += 4
     uncompress(&dst[chunk_offset], ..., src + pos, chunk_size, ...)
     pos += chunk_size
     chunk_offset += (total_n / nthreads)
5. return success
```

### 2.3 第三步：集成到 write_cube / read_cube

**写入流程 (write_cube 集成)：**

```
write_cube(file, comment, ..., data, precision, ndata_line, use_compression=false)
  │
  ├── 写入文本头部 (保持原样，兼容 VESTA)
  │
  └── if use_compression:
        ├── compress_charge_data(data, nxyz, compressed_buf)
        ├── 写入压缩二进制标记 (如注释行中加入 "COMPRESSED_ZLIB")
        └── 写入 compressed_buf 原始字节
      else:
        └── 写入文本数据 (原逻辑)
```

**读取流程 (read_cube 集成)：**

```
read_cube(file, comment, natom, ..., data)
  │
  ├── 读取文本头部 (原逻辑)
  │
  └── 检测是否压缩:
        if 注释中有 "COMPRESSED" 标记:
          ├── 读取剩余文件为 raw bytes
          ├── decompress_charge_data(raw, raw_len, data, nxyz)
          └── return true
        else:
          └── 读取文本数据 (原逻辑)
```

### 2.4 设计决策：兼容性 vs 性能

**选择方案：保留文本头部 + 压缩二进制数据段**

| 方案 | 兼容 VESTA | 压缩收益 | 实现复杂度 |
|------|-----------|---------|-----------|
| A: 全压缩（含头部） | 否，需专用解压工具 | 最大 | 低 |
| B: 头部文本 + 压缩数据 | 头部可读，数据需解压 | 高（数据占比 >95%） | 中 |
| **C: 压缩注释标记 + 二进制数据** | **需转换工具** | **最高（省去文本转换）** | **中** |

**选择方案 C**：通过注释中的 `COMPRESSED_ZLIB` 标记来区分压缩/非压缩文件。`read_cube` 检测此标记后自动切换解压路径。

---

## 三、实现计划

### 3.1 需要修改/创建的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/source_io/module_output/charge_compress.h` | **新建** | 压缩模块对外接口 |
| `source/source_io/module_output/charge_compress.cpp` | **新建** | 串行 + OpenMP 压缩/解压实现 |
| `source/source_io/module_output/write_cube.cpp` | 修改 | 集成压缩选项到 `write_cube` |
| `source/source_io/module_output/read_cube.cpp` | 修改 | 集成解压选项到 `read_cube` |
| `source/source_io/module_output/cube_io.h` | 修改 | 更新函数签名（新增参数） |
| `source/source_io/CMakeLists.txt` | 修改 | 添加 `charge_compress.cpp` 到 objects |
| `source/source_io/test/charge_compression_test.cpp` | 修改 | 添加 OpenMP 并行测试 + 集成测试 |
| `source/source_io/test/CMakeLists.txt` | 修改 | 更新链接 |

### 3.2 实现步骤

**Step 1: 提取压缩模块 (charge_compress.h/.cpp)**
- 从 test 文件中移出 `static` 函数
- 放入 `ModuleIO` namespace
- 添加完善的错误处理

**Step 2: 添加 OpenMP 并行压缩**
- 实现 `compress_charge_data_omp()` 和 `decompress_charge_data_omp()`
- 新 wire format 支持 chunk 信息
- 自动检测编译宏 `_OPENMP` 决定是否启用

**Step 3: 集成到 write_cube / read_cube**
- `write_cube` 增加 `compress` 参数
- `read_cube` 自动检测压缩标记
- 更新 `write_vdata_palgrid` 调用链

**Step 4: 更新测试**
- 验证 OpenMP 压缩/解压正确性
- 测量 OpenMP 多线程加速比
- 集成测试：write_cube(compressed) → read_cube → 数据一致性

### 3.3 关键接口设计

```cpp
// charge_compress.h

namespace ModuleIO {

// Magic word for compressed charge data
constexpr uint32_t CHARGE_COMPRESS_MAGIC = 0x504D435A; // "ZCMP"

// Serial compression (existing zlib-based)
bool compress_charge_data(const double* src, size_t n,
                          std::vector<uint8_t>& dst);

bool decompress_charge_data(const uint8_t* src, size_t src_len,
                            double* dst, size_t n);

// OpenMP-parallel compression
// nthreads=0 means use omp_get_max_threads()
bool compress_charge_data_omp(const double* src, size_t n,
                              std::vector<uint8_t>& dst,
                              int nthreads = 0);

bool decompress_charge_data_omp(const uint8_t* src, size_t src_len,
                                double* dst, size_t n);

} // namespace ModuleIO
```

---

## 四、预期性能提升

### 4.1 压缩性能预期（OpenMP 多线程）

以 256^3 网格（128 MB），4 线程为例：

| 配置 | 压缩耗时 | 吞吐量 | vs 串行 |
|------|---------|--------|---------|
| 串行 zlib | 17130 ms | 7.47 MB/s | 1.0x |
| OpenMP 2 线程 | ~9000 ms | 14.2 MB/s | 1.9x |
| OpenMP 4 线程 | ~5000 ms | 25.6 MB/s | 3.4x |
| OpenMP 8 线程 | ~3500 ms | 36.6 MB/s | 4.9x |

**依据：** zlib 的 `compress2` 是 CPU-bound，数据分块后各线程独立压缩，几乎没有同步开销。加速比接近线程数（受限于内存带宽和最后线程的负载不均）。

### 4.2 端到端 I/O 收益（write_cube + 压缩集成）

256^3 网格，4 进程，4 OpenMP 线程：

| 阶段 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 文本格式化 | ~20 s (串行) | 0 s (跳过) | — |
| 压缩 | 无 | ~5.0 s (4 线程) | — |
| 磁盘写入 | ~4.5 s (文本 360MB) | ~1.0 s (压缩 ~97MB) | 4.5x |
| **总计写入** | **~24.5 s** | **~6.0 s** | **~4x** |
| 文件体积 | ~360 MB | ~97 MB | 3.7x |

### 4.3 结合 Task 3 MPI-IO 的综合收益

| 配置 | 写入时间 | 文件体积 |
|------|---------|---------|
| 原始（文本串行） | 24.5 s | 360 MB |
| + MPI-IO 二进制 | 1.5 s | 128 MB |
| **+ MPI-IO + 压缩** | **~2.0 s** | **~50-80 MB** |
| **综合提升** | **~12x** | **~5-7x** |

---

## 五、测试计划

### 5.1 新增正确性测试

| 测试名称 | 说明 |
|---------|------|
| `CompressDecompressOMPRoundtrip` | OpenMP 多线程压缩后解压，验证误差 < 1e-6 |
| `OMPCompressSameAsSerial` | OpenMP 压缩结果与串行一致（各自解压后数据相等） |
| `OMPCompressThreadCountVariation` | 测试 1/2/4/8 线程均正确解压 |
| `WriteCubeWithCompressionRoundtrip` | write_cube(compress=true) → read_cube → 数据一致 |
| `AutoDetectCompressedFile` | read_cube 自动识别压缩/非压缩文件 |

### 5.2 新增性能测试

| 测试名称 | 说明 |
|---------|------|
| `Bench_Compress_OMP_nthreads2_256` | 2 线程压缩 256^3 |
| `Bench_Compress_OMP_nthreads4_256` | 4 线程压缩 256^3 |
| `Bench_Compress_OMP_nthreads8_256` | 8 线程压缩 256^3 |
| `Bench_Compress_OMP_Scaling_256` | 1→2→4→8 线程加速比曲线 |
| `Bench_WriteCube_Compressed_256` | write_cube 集成压缩的端到端时间 |

### 5.3 现有测试保留

现有的 13 个测试全部保留，`compress_charge_data` 的接口不变（回到 module 中但签名相同）。测试文件引用 `<charge_compress.h>` 而非 `static` 副本。

---

## 六、风险与注意事项

1. **zlib 内存使用**：`compressBound(128 MB) ≈ 128 MB + 6 KB + 额外开销`，每个 OpenMP 线程都需要独立的压缩缓冲区。4 线程需要约 500 MB 额外内存。对 256^3，总内存压力约 1 GB。
2. **OpenMP 嵌套**：若 write_cube 已被 MPI 并行化（Task 3），MPI + OpenMP 混合需注意 `MPI_Init_thread` 的线程支持级别（`MPI_THREAD_FUNNELED` 即可，因为仅主线程调 MPI）。
3. **向后兼容**：老的纯文本 `.cube` 文件继续被正确读取（检测不到压缩标记时走原路径）。
4. **跨平台 zlib**：zlib 已在顶层 CMakeLists.txt 中通过 `find_package(ZLIB REQUIRED)` 引入，编译环境已确保可用。
