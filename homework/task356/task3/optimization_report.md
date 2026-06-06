# Task 3: Cube 文件 MPI-IO 并行写入 —— 优化实现报告（第二轮）

Date: 2026-06-06

---

## 一、本轮优化概要

在第一轮（2026-05-26）完成 `write_cube_mpi()` + CIPM marker 的基础上，
本轮进一步优化了压缩集成和代码质量。

### 1.1 本轮完成内容

| 内容 | 状态 |
|------|------|
| `write_cube()` 集成 v1 压缩格式（ZCM2 magic + flags byte + 自动回退） | 完成 |
| `read_cube()` 升级为通用解压（`decompress_charge_data_any()`） | 完成 |
| `read_cube()` 新增 ZCM2(v1) 格式自动检测 | 完成 |
| `cube_io.h` 新增 `write_cube_mpi()` 和压缩参数声明 | 完成 |
| 正确性测试 (10 个，全部通过) | 通过 |

### 1.2 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/source_io/module_output/write_cube.cpp` | 修改 | 压缩路径升级为 v1 格式 |
| `source/source_io/module_output/read_cube.cpp` | 修改 | 通用解压自动检测 |
| `source/source_io/module_output/cube_io.h` | 修改 | 新增 MPI-IO 声明 |
| `source/source_io/module_output/charge_compress.h` | 新建 | 压缩模块接口 |
| `source/source_io/module_output/charge_compress.cpp` | 新建 | 压缩模块实现 |
| `source/source_io/test/write_cube_test.cpp` | 修改 | MPI-IO 正确性 + 性能测试 |
| `source/source_io/test/CMakeLists.txt` | 修改 | 链接 zlib |

---

## 二、核心改进

### 2.1 压缩集成

`write_cube()` 新增 `compress` 和 `compress_nthreads` 参数（默认 false/0，向后兼容）：

```
write_cube()
  ├── 文本头部 (comment, atom info, grid vectors) — 与原来完全一致
  ├── 如果 compress=true:
  │     ├── compress_charge_data_v1() 或 compress_charge_data_omp_v1()
  │     │     → v1 格式: [ZCM2 4B][fmt_type 1B][flags 1B][total_n 8B][payload]
  │     │     → 自动回退: 若压缩后 ≥ 原始大小，存原始数据 (flags & 0x01 = 0)
  │     └── 二进制追加写入
  └── 否则: 原有文本格式写入 (完全兼容)
```

### 2.2 格式自动检测

`read_cube()` 支持 4 种格式的自动检测：

| 首字节 | Magic | 格式 | 处理 |
|--------|-------|------|------|
| 数字/`-`/`.` | — | 文本 | 原有流式读取 |
| `Z` | `0x504D435A` (ZCMP) | 压缩 v0 | `decompress_charge_data_any()` |
| `Z` | `0x324D435A` (ZCM2) | 压缩 v1 | `decompress_charge_data_any()` |
| `C` | `0x4D504943` (CIPM) | MPI 二进制 | 直接 memcpy |

---

## 三、性能数据

| 操作 | 网格 | 耗时 | 吞吐量 |
|------|------|------|--------|
| 文本写入 (基线) | 64³ | 141 ms | 14.5 MB/s |
| 文本写入 (基线) | 128³ | 1,220 ms | 13.3 MB/s |
| 文本写入 (基线) | 256³ | 7,315 ms | 17.6 MB/s |
| **MPI-IO 二进制写** | **256³** | **394 ms** | **326.7 MB/s** |

> MPI-IO 相比文本写入加速 **18.6x**。

---

## 四、测试结果

```
10/10 tests PASSED (serial)
10/10 tests PASSED (MPI np=4)
```

| 测试用例 | 说明 |
|----------|------|
| WriteTextCubeAndReadBack | 文本写入 → 读取 roundtrip |
| ReadbackHeaderCorrect | 文件头所有字段正确恢复 |
| DataLayoutZFastest | z-fastest 索引顺序验证 |
| ReadCubeFileNotFound | 不存在文件返回 false |
| Bench_WriteCube_SerialText_{64,128,256} | 串行文本性能基线 |
| MPIWriteBinaryCubeAndReadBack | MPI-IO 写入 → 读取 roundtrip |
| TextVsBinaryConsistency | 文本 vs MPI 二进制一致性 |
| Bench_WriteCube_MPIIO_Binary_256 | MPI-IO 性能基线 |
