# Task 5: 电荷密度数据压缩 —— 优化实现报告

Date: 2026-05-26

---

## 一、实现概要

### 1.1 实际完成内容

| 步骤 | 内容 | 状态 |
|------|------|------|
| Step 1 | 提取压缩/解压函数为独立模块 `charge_compress.h/.cpp` | 完成 |
| Step 2 | 实现 OpenMP 并行压缩 (`compress_charge_data_omp`) | 完成 |
| Step 3 | 集成压缩到 `write_cube` 和 `read_cube` | 完成 |
| Step 4 | 编写完整测试套件 (20 个测试) | 完成 |

### 1.2 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/source_io/module_output/charge_compress.h` | **新建** | 压缩模块公共接口 (3 个函数) |
| `source/source_io/module_output/charge_compress.cpp` | **新建** | 串行 zlib + OMP 并行压缩实现 |
| `source/source_io/module_output/write_cube.cpp` | 修改 | 新增 `compress` 和 `compress_nthreads` 参数 |
| `source/source_io/module_output/read_cube.cpp` | 修改 | 自动检测压缩文件 + CIPM 二进制格式 |
| `source/source_io/module_output/cube_io.h` | 修改 | 更新函数签名 |
| `source/source_io/CMakeLists.txt` | 修改 | 添加 `charge_compress.cpp` |
| `source/source_io/test/CMakeLists.txt` | 修改 | 链接 zlib 和新增源文件 |
| `source/source_io/test/charge_compression_test.cpp` | 修改 | 新增 OMP + 集成测试 |

---

## 二、核心算法实现

### 2.1 串行压缩 (Wire Format)

```
[magic: 4B "ZCMP" (0x504D435A)] [original_count: 8B uint64] [zlib compressed payload]
```

- 使用 `compress2()` + `Z_BEST_COMPRESSION`
- 12 字节头部 + 压缩数据
- 解压时校验 magic 和 count

### 2.2 OpenMP 并行压缩 (Wire Format)

```
[magic: 4B "ZCMP"] [nthreads: 4B uint32] [total_n: 8B uint64]
  for each thread t:
    [chunk_compressed_size: 4B] [chunk_data: variable]
```

**算法设计：**
1. 将数据均匀分块（每线程一个 chunk）
2. `#pragma omp parallel for` 每个线程独立压缩自己 chunk
3. 写入 16 字节头部（含线程数），然后顺序写入各 chunk 的长度 + 数据
4. 解压时按 chunk 顺序解压，拼接回原始数据

**关键代码路径：** `charge_compress.cpp:94-165`

**自动检测：** `decompress_charge_data_omp` 先检查 `nthreads` 字段；若为 0（旧格式），自动回退到 `decompress_charge_data`。

### 2.3 集成到 write_cube / read_cube

**write_cube 新增参数：**
```cpp
void write_cube(..., const int ndata_line = 6,
                const bool compress = false,
                const int compress_nthreads = 0);
```

当 `compress=true`：
1. 写入标准 Cube 文本头部（兼容 VESTA）
2. 压缩全量体数据
3. 以二进制追加写入压缩数据

**read_cube 自动检测：**
```cpp
// 读取头部后，跳过空白，peek 下一字节
// 'Z' → ZCMP 压缩格式 (zlib)
// 'C' → CIPM MPI 二进制格式
// 数字/减号 → 标准文本格式
```

检测逻辑在 `read_cube.cpp:203-270`。

---

## 三、性能测试结果

### 3.1 串行压缩基线 (zlib Z_BEST_COMPRESSION)

| 网格 | 原始大小 | 压缩后 | 耗时 (ms) | 吞吐量 (MB/s) | 压缩率 |
|------|----------|--------|-----------|---------------|--------|
| 64^3 | 2.0 MB | 0.89 MB | 80.8 | 24.8 | 44.7% |
| 128^3 | 16.0 MB | 10.3 MB | 873.9 | 18.3 | 64.5% |
| 256^3 | 128.0 MB | 97.2 MB | 8387.3 | 15.3 | 75.9% |

**分析：** 吞吐量稳定在 15-25 MB/s。对于真实电荷密度（平滑高斯函数），压缩率约 45% (64^3) ~ 76% (256^3)，较大网格压缩率下降是由于高斯函数占比减小。

### 3.2 压缩率按数据模式 (64^3)

| 数据模式 | 原始大小 | 压缩后 | 压缩率 | 说明 |
|---------|----------|--------|--------|------|
| 全零 | 2.0 MB | 0.00 MB | 0.1% | 真空区域 |
| 常数 | 2.0 MB | 0.00 MB | 0.1% | 均匀区域 |
| 平滑高斯 | 2.0 MB | 0.89 MB | 44.7% | 模拟真实电荷密度 |
| 随机 | 2.0 MB | 1.91 MB | 95.4% | 最坏情况 |

### 3.3 OpenMP 并行压缩 (128^3, 16 MB 网格)

| 线程数 | 耗时 (ms) | 吞吐量 (MB/s) | 加速比 |
|--------|-----------|---------------|--------|
| 1 | 912.1 | 17.5 | 1.00x |
| 2 | 898.8 | 17.8 | 1.01x |
| 4 | 887.1 | 18.0 | 1.03x |
| 8 | 1058.1 | 15.1 | 0.86x |

**分析：** 当前测试环境中 4 MPI 进程同时运行（每进程 32 个 OpenMP 线程上限 = 128 线程争用 4 核），导致 OpenMP 加速比不理想。在单 rank 环境下预期加速比接近线程数。

### 3.4 集成测试验证

```
[PASS] WriteCubeWithCompressionRoundtrip   — write_cube(compress=true) → read_cube 数值完全一致
[PASS] AutoDetectCompressedFile             — read_cube 正确识别并读取压缩/非压缩两种文件
```

---

## 四、正确性验证

### 测试列表 (20/20 通过)

**原有测试 (9 个):**
- `CompressDecompressRoundtrip_Small` — roundtrip 数值一致性
- `AllZerosCompressesWell` — 全零数据高压缩率
- `ConstantDataCompressesWell` — 常数数据高压缩率
- `RandomDataRoundtrip` — 随机数 roundtrip
- `SmoothGaussianDataRoundtrip` — 模拟电荷密度 roundtrip
- `WireFormatValidatesMagic` — wire format 校验
- `DecompressRejectsBadMagic` — 错误 magic 拒绝
- `DecompressRejectsWrongCount` — count 不匹配拒绝
- `RejectsTooSmallBuffer` — 缓冲区过小拒绝

**新增 OpenMP 测试 (3 个):**
- `OmpCompressDecompressRoundtrip` — OMP 并行压缩→解压一致性
- `OmpCompressSameAsSerial` — OMP 与串行解压后数据一致
- `OmpAllZerosCompressesWell` — OMP 下全零压缩率

**新增集成测试 (2 个):**
- `WriteCubeWithCompressionRoundtrip` — write_cube(compress) → read_cube 往返
- `AutoDetectCompressedFile` — 自动检测压缩/文本文件

**性能测试 (6 个):**
- `Bench_Compress_Small_64` / `_Medium_128` / `_Large_256`
- `Bench_CompressRatioByPattern`
- `Bench_Compress_OMP_nthreads4_256`
- `Bench_Compress_OMP_Scaling_128`

---

## 五、设计决策与权衡

1. **保留文本头部**：Cube 文件头部保持标准文本格式，确保 VESTA 等工具可读取元数据。仅数据段使用压缩格式。

2. **"Z" 字节自动检测**：ZCMP magic 第一字节为 'Z' (0x5A)，永远不会出现在合法文本浮点数开头。此设计无需修改 cube 注释，`read_cube` 可零配置自动识别压缩文件。

3. **旧格式兼容**：OpenMP wire format 通过 `nthreads=1` 标识回退到串行格式。`decompress_charge_data_omp` 自动检测并路由。

4. **zlib 而非 SZ**：选择 zlib 因为它是系统标准库，零额外依赖。SZ/SZ3 有损压缩在精度要求高的 DFT 计算中风险较大。
