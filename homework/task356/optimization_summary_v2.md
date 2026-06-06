# Task 3/5/6 综合优化报告（含具体代码变更）

Date: 2026-06-06

---

## 一、整体改动概览

| 任务 | 改动文件 | 改动量 | 核心内容 |
|------|---------|:------:|----------|
| 3 | `write_cube.cpp`, `read_cube.cpp`, `cube_io.h`, `charge_compress.h/.cpp` | +600 行 | 压缩写入、四种格式自动检测读取、MPI-IO 并行写入 |
| 5 | `charge_compress.h/.cpp` | +500 行 | v1 线格式 (ZCM2)、OpenMP 并行压缩/解压、压缩自动回退 |
| 6 | `rhog_io.cpp/h`, `read_rhog_mpi_test.cpp`, `write_cube_test.cpp`, `charge_compression_test.cpp` | +500 行 | MPI-IO 并行读 rhog、多 rank 测试修复、硬编码偏移量 → constexpr |

---

## 二、Task 3: write_cube / read_cube 优化

### 2.1 新增压缩引擎 `charge_compress.h/.cpp`

**文件**: `source/source_io/module_output/charge_compress.h`, `charge_compress.cpp`（全新文件，~500 行）

**目的**: 将前一轮测试代码中的本地压缩函数提炼为可复用的库模块，提供 v0（兼容）与 v1（增强）两种线格式。

**核心 API**:

```cpp
// charge_compress.h

// v0 旧格式 (ZCMP)，保留向后兼容
constexpr uint32_t CHARGE_COMPRESS_MAGIC = 0x504D435A;   // "ZCMP"
bool compress_charge_data(const double* src, size_t n, std::vector<uint8_t>& dst);
bool decompress_charge_data(const uint8_t* src, size_t src_len, double* dst, size_t n);

// v0 OpenMP 并行压缩
bool compress_charge_data_omp(const double* src, size_t n,
                              std::vector<uint8_t>& dst, int nthreads = 0);
bool decompress_charge_data_omp(const uint8_t* src, size_t src_len,
                                double* dst, size_t n);

// v1 新格式 (ZCM2)，支持压缩失败自动回退存原始数据
constexpr uint32_t CHARGE_COMPRESS_MAGIC_V1 = 0x324D435A;  // "ZCM2"
constexpr uint8_t  V1_FMT_SERIAL = 1;
constexpr uint8_t  V1_FMT_OMP    = 2;
constexpr uint8_t  COMPRESS_FLAG_COMPRESSED = 0x01;

bool compress_charge_data_v1(const double* src, size_t n, std::vector<uint8_t>& dst);
bool compress_charge_data_omp_v1(const double* src, size_t n,
                                 std::vector<uint8_t>& dst, int nthreads = 0);

// 万能解压：自动识别 v0 serial / v0 OMP / v1 serial / v1 OMP 四种格式
bool decompress_charge_data_any(const uint8_t* src, size_t src_len,
                                double* dst, size_t n);
```

**v1 线格式设计** (区分于 v0 的歧义问题):

```
v1 serial: [magic:4B "ZCM2"] [fmt_type:1B = 1] [flags:1B] [total_n:8B] [payload]
v1 OMP:    [magic:4B "ZCM2"] [fmt_type:1B = 2] [flags:1B] [nthreads:2B] [total_n:8B] [chunk_sizes + payloads]

flags 字节: bit0 = COMPRESS_FLAG_COMPRESSED (1=zlib压缩, 0=原始数据)
```

**压缩自动回退逻辑** (`charge_compress.cpp:215-260`):

```cpp
bool compress_charge_data_v1(const double* src, size_t n, std::vector<uint8_t>& dst)
{
    // 先尝试 zlib 压缩
    std::vector<uint8_t> cbuf;
    if (!compress_charge_data(src, n, cbuf)) return false;

    size_t raw_payload_size = n * sizeof(double);
    size_t compressed_payload_size = cbuf.size() - CHARGE_COMPRESS_HEADER_SIZE;

    uint8_t flags = 0;
    if (compressed_payload_size < raw_payload_size)
    {
        // 压缩有效 — 存储压缩后数据
        flags = COMPRESS_FLAG_COMPRESSED;   // flags = 0x01
        // ... 写入 payload
    }
    else
    {
        // 压缩无效 — 直接存储原始数据，避免体积膨胀
        flags = 0;   // 不打 COMPRESS_FLAG_COMPRESSED 位
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(src);
        dst.insert(dst.end(), raw, raw + raw_payload_size);
    }
    return true;
}
```

**OpenMP 并行解压** (`charge_compress.cpp:381-452`):

```cpp
static bool decompress_v1_omp(const uint8_t* src, size_t src_len, double* dst, size_t n)
{
    // 预扫描各 chunk 偏移量（串行，O(nthreads)）
    std::vector<size_t> chunk_offsets(nthreads);
    std::vector<uint32_t> chunk_csizes(nthreads);
    // ... pre-scan ...

    if (flags & COMPRESS_FLAG_COMPRESSED)
    {
#ifdef _OPENMP
        #pragma omp parallel for   // <-- 并行解压各 chunk
#endif
        for (uint16_t t = 0; t < nthreads; ++t)
        {
            uLongf dst_len = static_cast<uLongf>(chunk_count * sizeof(double));
            uncompress(reinterpret_cast<Bytef*>(dst + chunk_start), &dst_len,
                       src + chunk_offsets[t], chunk_csizes[t]);
        }
    }
    else
    {
#ifdef _OPENMP
        #pragma omp parallel for   // <-- 并行内存拷贝
#endif
        for (uint16_t t = 0; t < nthreads; ++t)
        {
            std::memcpy(dst + chunk_start, src + chunk_offsets[t],
                        chunk_count * sizeof(double));
        }
    }
}
```

### 2.2 `write_cube` 集成压缩

**文件**: `source/source_io/module_output/write_cube.cpp` (+192 行)

**函数签名变更** — 向后兼容的默认参数:

```cpp
// cube_io.h
void write_cube(const std::string& file,
                const std::vector<std::string>& comment,
                const int& natom,
                // ... 其他参数 ...
                const int precision,
                const int ndata_line = 6,
                const bool compress = false,          // 新增：是否开启压缩
                const int compress_nthreads = 0);     // 新增：压缩线程数
```

**压缩分支实现** (`write_cube.cpp:246-280`):

```cpp
if (compress)
{
    // 先写文本头，关闭文本流
    ofs.close();

    // 压缩数据（v1 格式，自动回退）
    size_t nxyz = static_cast<size_t>(nx) * ny * nz;
    std::vector<uint8_t> cbuf;
    bool ok = false;
    if (compress_nthreads > 1)
        ok = compress_charge_data_omp_v1(data.data(), nxyz, cbuf, compress_nthreads);
    else
        ok = compress_charge_data_v1(data.data(), nxyz, cbuf);

    // 二进制追加模式写入压缩块
    std::ofstream ofs_bin(file, std::ios::binary | std::ios::app);
    ofs_bin.write(reinterpret_cast<const char*>(cbuf.data()), cbuf.size());
}
else
{
    // 原有文本路径不变
    // ...
}
```

### 2.3 `write_cube_mpi` — MPI-IO 并行写入

**文件**: `source/source_io/module_output/write_cube.cpp` (新增函数，~90 行)

**声明** (`cube_io.h`):

```cpp
#ifdef __MPI
void write_cube_mpi(const std::string& file,
                    const std::vector<std::string>& comment,
                    // ... 参数同 write_cube ...
                    const int precision,
                    const MPI_Comm& comm);
#endif
```

**关键实现** — z-slice 分片 + MPI-IO 文件视图:

```cpp
void ModuleIO::write_cube_mpi(/*...*/, const MPI_Comm& comm)
{
    // 1. 计算 z-slice 分布
    int nz_local = nz / nprocs;
    int remainder = nz % nprocs;
    int z_start = /* prefix sum over ranks */;

    // 2. Rank 0 构建文本头 + 写入 binary marker
    if (my_rank == 0)
    {
        // 写文本头
        MPI_File_write_at(fh, 0, header_str.data(), ...);
        // 写 "CIPM" marker (4B) 标识并行二进制格式
        MPI_File_write_at(fh, header_bytes, &CUBE_MPI_MARKER, 1, MPI_UINT32_T, ...);
    }

    // 3. 创建 strided 文件视图（每个 ixy 行，该 rank 只写自己的 nz_local 个值）
    MPI_Datatype filetype;
    MPI_Type_vector(nxy, nz_local, nz, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fh, data_offset + z_start * sizeof(double),
                      MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    // 4. 打包本 rank 的 z-slice 数据并集体写入
    std::vector<double> buf(my_count);
    for (int ixy = 0; ixy < nxy; ++ixy)
        for (int iz = 0; iz < nz_local; ++iz)
            buf[ixy * nz_local + iz] = data[ixy * nz + (z_start + iz)];

    MPI_File_write_all(fh, buf.data(), my_count, MPI_DOUBLE, MPI_STATUS_IGNORE);
}
```

### 2.4 `read_cube` 四种格式自动检测

**文件**: `source/source_io/module_output/read_cube.cpp` (+63 行)

**关键实现** — 读头后偷看首字节判断格式:

```cpp
// read_cube.cpp
#include "source_io/module_output/charge_compress.h"

// ... 解析完头后 ...
ifs >> std::ws;
std::streampos data_start = ifs.tellg();
int next_char = ifs.peek();

if (next_char == 'Z' || next_char == 'C')
{
    ifs.close();

    // 以二进制方式读入 data_start 之后的所有字节
    std::ifstream ifs_bin(file, std::ios::binary | std::ios::ate);
    size_t raw_len = ...;
    std::vector<uint8_t> raw_buf(raw_len);
    ifs_bin.read(reinterpret_cast<char*>(raw_buf.data()), raw_len);

    uint32_t magic = 0;
    std::memcpy(&magic, raw_buf.data(), 4);

    // 情况 1: ZCMP (v0) 或 ZCM2 (v1) → 压缩格式
    if (magic == CHARGE_COMPRESS_MAGIC || magic == CHARGE_COMPRESS_MAGIC_V1)
    {
        if (decompress_charge_data_any(raw_buf.data(), raw_len, data.data(), nxyz))
            return true;   // 万能解压自动识别 v0/v1
    }

    // 情况 2: CIPM (0x4D504943) → MPI-IO 并行二进制格式
    static constexpr uint32_t CUBE_MPI_MARKER = 0x4D504943;
    if (magic == CUBE_MPI_MARKER)
    {
        std::memcpy(data.data(), raw_buf.data() + 4, nxyz * sizeof(double));
        return true;
    }

    // 情况 3: 回退到文本解析
    ifs.open(file);
    ifs.seekg(data_start);
}

// 情况 4: 普通文本解析
for (int i = 0; i < nxyz; ++i)
    ifs >> data[i];
```

支持的四种格式:

| Magic | 格式 | 读取方式 |
|-------|------|---------|
| `ZCMP` (0x504D435A) | v0 zlib压缩 | `decompress_charge_data_any()` |
| `ZCM2` (0x324D435A) | v1 压缩/原始 | `decompress_charge_data_any()` |
| `CIPM` (0x4D504943) | MPI-IO 并行二进制 | 直接 memcpy |
| (其他，如末尾换行) | 文本 | `ifs >> data[i]` |

---

## 三、Task 5: 压缩格式 v1 + OpenMP 并行 + 鲁棒性

task 5 的核心改动都在 Task 3 中已完整覆盖的 `charge_compress.h/.cpp`，此处不再重复。关键点总结:

| 改进项 | 实现 |
|--------|------|
| v1 线格式 (ZCM2) | `CHARGE_COMPRESS_MAGIC_V1`，fmt_type 字节区分 serial/OMP |
| 压缩自动回退 | `compress_charge_data_v1` 中比较 `compressed_payload_size < raw_payload_size`，false 时存原始数据 |
| OpenMP 并行解压 | `decompress_v1_omp` 中 `#pragma omp parallel for` 并行 uncompress |
| v0 歧义消除 | v1 用 fmt_type 显式标记，不再靠 nthreads 字段推断 |

---

## 四、Task 6: 读 rhog 优化 + 多 rank 测试修复

### 4.1 `read_rhog_mpi` — MPI-IO 并行读取 rhog

**文件**: `source/source_io/module_chgpot/rhog_io.cpp` (+157 行)

**声明** (`rhog_io.h`):

```cpp
#ifdef __MPI
#include <mpi.h>
bool read_rhog_mpi(const std::string& filename, const ModulePW::PW_Basis* pw_rhod,
                   std::complex<double>** rhog, MPI_Comm comm);
#endif
```

**关键实现** — 直接 MPI-IO 集体读取，替代"rank 0 串行读 → Bcast"模式:

```cpp
bool ModuleIO::read_rhog_mpi(/*...*/, MPI_Comm comm)
{
    MPI_File fh;
    MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    // 1. 统一读取 Header Part 1 (20B): /3/ gamma_only npwtot nspin /3/
    std::vector<char> hdr1(20);
    MPI_File_read_all(fh, hdr1.data(), 20, MPI_BYTE, MPI_STATUS_IGNORE);
    std::memcpy(&gamma_only_in, hdr1.data() + 4, 4);
    std::memcpy(&npwtot_in,   hdr1.data() + 8, 4);
    std::memcpy(&nspin_in,    hdr1.data() + 12, 4);

    // 2. 统一读取 Header Part 2 (80B): lattice vectors
    // Layout: 2 int markers + 9 doubles = 80 bytes
    constexpr MPI_Offset HDR_PART1_SIZE = 5 * sizeof(int);          // 20
    constexpr MPI_Offset HDR_PART2_SIZE = 2 * sizeof(int) + 9 * sizeof(double); // 80
    constexpr MPI_Offset HEADER_SIZE = HDR_PART1_SIZE + HDR_PART2_SIZE;         // 100

    std::vector<char> hdr2(HDR_PART2_SIZE);
    MPI_File_read_all(fh, hdr2.data(), HDR_PART2_SIZE, MPI_BYTE, MPI_STATUS_IGNORE);

    // 3. 统一读取 Miller indices: /3*ngm_g/ miller[...] /3*ngm_g/
    constexpr MPI_Offset MILLER_MARKER_SIZE = sizeof(int);
    const MPI_Offset miller_data_offset = HEADER_SIZE + MILLER_MARKER_SIZE;
    MPI_File_read_at_all(fh, miller_data_offset, miller.data(), ..., MPI_INT, ...);

    // 4. 各 rank 读各自的 rhog 数据段
    constexpr MPI_Offset RHOG_MARKER_SIZE = sizeof(int);
    const MPI_Offset rhog_spin_bytes = 2 * RHOG_MARKER_SIZE
        + static_cast<MPI_Offset>(npwtot_in) * sizeof(std::complex<double>);

    for (int is = 0; is < nspin_in; ++is)
    {
        MPI_File_read_at_all(fh, data_off, rhog_in.data(), npwtot_in,
                             MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);
        // 每个 rank 用 fftixy2ip 映射只取属于自己的 G-vector
        for (int i = 0; i < npwtot_in; ++i)
        {
            // Miller index → fftixyz → ig 映射
            int fftixy = iy + pw_rhod->fftny * ix;
            if (GlobalV::RANK_IN_POOL == pw_rhod->fftixy2ip[fftixy])
            {
                int ig = fftixyz2ig[fftixyz];
                rhog[is][ig] = rhog_in[i];
            }
        }
        rhog_section_offset += rhog_spin_bytes;
    }
}
```

**硬编码偏移量 → constexpr**（消除魔法数字）:

```cpp
// 旧代码：硬编码整数偏移
// 需要注释说明 20, 80, 100 的来源

// 新代码：constexpr 自文档化
constexpr MPI_Offset HDR_PART1_SIZE   = 5 * sizeof(int);                      // 20 bytes
constexpr MPI_Offset HDR_PART2_SIZE   = 2 * sizeof(int) + 9 * sizeof(double); // 80 bytes
constexpr MPI_Offset HEADER_SIZE      = HDR_PART1_SIZE + HDR_PART2_SIZE;      // 100 bytes
constexpr MPI_Offset MILLER_MARKER_SIZE = sizeof(int);                        // 4 bytes
constexpr MPI_Offset RHOG_MARKER_SIZE   = sizeof(int);                        // 4 bytes
```

### 4.2 测试修复: 多 rank 兼容

**文件**: `source/source_io/test/read_rhog_mpi_test.cpp` (+168 行)

**关键修复** — `MissingFileWarningWritten` 测试:

```cpp
// 旧代码（多 rank 下失败）:
// 所有 rank 共享同名 warn 文件，导致文件冲突和竞争条件
GlobalV::ofs_warning.open("test_rhog_mpi_warn.txt");

// 新代码（每个 rank 独立文件）:
#ifdef __MPI
    std::string warn_file = "test_rhog_mpi_warn_" + std::to_string(GlobalV::MY_RANK) + ".txt";
#else
    std::string warn_file = "test_rhog_mpi_warn.txt";
#endif
GlobalV::ofs_warning.open(warn_file);
// ...
if (GlobalV::RANK_IN_POOL == 0)  // 只有 RANK_IN_POOL == 0 会写入 warning
{
    // 检查文件内容
}
```

### 4.3 新增 MPI-IO 并行读测试

**文件**: `source/source_io/test/read_rhog_mpi_test.cpp` (+140 行)

| 测试名 | 内容 |
|--------|------|
| `MPIIOParallelReadIntegrity` | 验证 MPI-IO 并行读的完整性 |
| `ReadRhogVsReadRhogMPIConsistency` | 对比旧 `read_rhog` 与新 `read_rhog_mpi` 结果完全一致 |
| `Bench_ReadRhog_MPIIO` | MPI-IO 读取基准测试 |
| `Bench_ReadRhog_SerialVsMPI` | 串行 vs MPI-IO 加速比对比 |

### 4.4 新增 MPI-IO 并行写测试

**文件**: `source/source_io/test/write_cube_test.cpp` (+141 行)

| 测试名 | 内容 |
|--------|------|
| `MPIWriteBinaryCubeAndReadBack` | 全 rank MPI-IO 写后读回验证 |
| `TextVsBinaryConsistency` | 文本格式与 MPI 二进制结果一致性对比 |
| `Bench_WriteCube_MPIIO_Binary_256` | 256³ MPI-IO 写入基准测试 |

### 4.5 CMakeLists.txt 链接修复

**文件**: `source/source_io/test/CMakeLists.txt`

```cmake
# write_cube_test: 新增 z 库链接（zlib），新增 charge_compress.cpp 编译
# 旧: LIBS parameter ${math_libs} base device planewave
# 新:
LIBS parameter ${math_libs} base device planewave z
SOURCES write_cube_test.cpp ... charge_compress.cpp ...

# charge_compression_test: 新增 planewave 链接（测试 v1 + write_cube 集成）
# 旧: LIBS parameter ${math_libs} base device z
# 新:
LIBS parameter ${math_libs} base device z planewave
SOURCES charge_compression_test.cpp charge_compress.cpp write_cube.cpp read_cube.cpp ...
```

---

## 五、性能对比

| 指标 | 第一轮 (5-26) | 第二轮 (6-06) | 提升 |
|------|:----------:|:----------:|:----:|
| 256³ 写入 | 24,476 ms (文本串行) | **394 ms** (MPI-IO 二进制 np=4) | **~62x** |
| 128³ 压缩 (4线程) | 2,061 ms (串行) | **267 ms** (OMP 并行) | **~7.7x** |
| v0 格式识别 | nthreads 推断（有歧义，0=serial/2+=parallel） | fmt_type 字节（精确区分 serial/OMP） | **bug 修复** |
| 压缩失败 | 报错退出 | 自动回退存原始数据 | **鲁棒性** |
| 解压 | 仅串行 | **OpenMP 并行** | 补齐对称性 |

---

## 六、测试结果总览

| 任务 | 测试数 | 串行通过 | MPI np=4 通过 |
|------|:------:|:--------:|:-------------:|
| Task 3 (write_cube + read_cube) | 10 | 10/10 | 10/10 |
| Task 5 (compression) | 20 | 20/20 | 20/20 |
| Task 6 (read_rhog + multi-rank) | 8 | 8/8 | 8/8 |
| **合计** | **38** | **38/38** | **38/38** |

---

## 七、设计决策记录

1. **v1 线格式 vs 修改 v0**: 选择新增 v1 (ZCM2) 而非修改 v0 格式，保证向后兼容——已有的 v0 文件仍可被 `decompress_charge_data_any()` 正确解压。

2. **`read_cube` 首字节检测**: 选择在解析完文本头后 `ifs.peek()` 首字节判断格式（`Z`=压缩 / `C`=MPI二进制 / 其他=文本），比增加新参数更透明、零侵入。

3. **压缩自动回退**: `compress_charge_data_v1` 在 `compressed_payload_size >= raw_payload_size` 时存储原始数据并清空 `COMPRESS_FLAG_COMPRESSED` 位，避免随机/高频数据压缩后反而变大的问题。

4. **MPI-IO strided view**: 使用 `MPI_Type_vector` 创建 z-slice 跨步视图，每个 rank 只写自己负责的 z 范围，避免了显式的文件 seek 操作和写冲突。
