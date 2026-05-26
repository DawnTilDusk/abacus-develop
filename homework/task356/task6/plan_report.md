# Task 6: 电荷密度读取 MPI-IO 并行化 —— 优化方案报告

Date: 2026-05-26

---

## 一、当前实现分析

### 1.1 现有代码结构

**源文件位置：**
- `source/source_io/module_chgpot/rhog_io.cpp` (422 行) — 实现 `read_rhog()` 和 `write_rhog()`
- `source/source_io/module_chgpot/rhog_io.h` (63 行) — 函数声明
- `source/source_io/module_output/binstream.h` — `Binstream` 类（C `FILE*` 包装器）

**测试文件：**
- `source/source_io/test/read_rhog_mpi_test.cpp` (246 行) — MPI 测试
- `source/source_io/test/read_rhog_test.cpp` (158 行) — 串行测试（预先存在）

**参考数据文件：** `source/source_io/test/support/charge-density.dat` (41 KB, 二进制格式)

### 1.2 当前读取算法流程

```
read_rhog(filename, pw_rhod, rhog)
  │
  ├── [rank_in_pool == 0] Binstream ifs.open(filename, "r")
  ├── [rank_in_pool == 0] if (!ifs) → error = true
  ├── MPI_Bcast(&error, ...)                    ← 广播 1: 错误标志
  │
  ├── [rank_in_pool == 0] 读取头部:
  │     ifs >> size >> gamma_only_in >> npwtot_in >> nspin_in >> size
  │     ifs >> size >> b1[0..2] >> b2[0..2] >> b3[0..2] >> size
  ├── MPI_Bcast(&error, ...)                    ← 广播 2: gamma_only 一致性
  ├── MPI_Bcast(&gamma_only_in, ...)            ← 广播 3
  ├── MPI_Bcast(&npwtot_in, ...)                ← 广播 4
  ├── MPI_Bcast(&nspin_in, ...)                 ← 广播 5
  ├── MPI_Bcast(b1, 3, MPI_DOUBLE, ...)         ← 广播 6
  ├── MPI_Bcast(b2, 3, MPI_DOUBLE, ...)         ← 广播 7
  ├── MPI_Bcast(b3, 3, MPI_DOUBLE, ...)         ← 广播 8
  │
  ├── [rank_in_pool == 0] 读取 Miller 索引:
  │     ifs >> size;
  │     for i in 0..npwtot_in:
  │       ifs >> miller[i*3] >> miller[i*3+1] >> miller[i*3+2]
  │     ifs >> size;
  ├── MPI_Bcast(miller, 3*npwtot_in, MPI_INT, ...) ← 广播 9: 大数组
  │
  ├── 构建 fftixyz2ig 映射表 (所有进程本地计算)
  │
  ├── for is in 0..nspin_in:
  │     ├── [rank_in_pool == 0] 读取 rhog 数据:
  │     │     ifs >> size;
  │     │     for i in 0..npwtot_in:
  │     │       ifs >> rhog_in[i]
  │     │     ifs >> size;
  │     │
  │     ├── MPI_Bcast(rhog_in, npwtot_in, MPI_DOUBLE_COMPLEX, ...) ← 广播 10+N
  │     │
  │     └── 数据分发: 每个进程过滤属于自己的 G-vector (Miller 索引→本地 ig)
  │
  └── [rank_in_pool == 0] ifs.close()
```

### 1.3 瓶颈分析

| 瓶颈 | 详细描述 | 影响程度 |
|------|---------|---------|
| **串行读取** | 仅 rank_in_pool == 0 执行所有文件 I/O | 极高 — 其他进程完全空闲 |
| **多次小 Bcast** | gamma_only, error, npwtot, nspin 各单独广播一次，共 8 次小消息 | 中 — 延迟累计 |
| **大数组 Bcast** | `miller` (3 × npwtot × 4B) 和 `rhog_in` (npwtot × 16B) 各广播一次 | 高 — 大数据量跨进程传输 |
| **重复内存分配** | 每次读取都需要完整 `rhog_in` 数组（npwtot 大小），即使本进程只需 npw 个 | 中 — 内存浪费 |
| **Binstream 文本解析** | `ifs >> value` 是格式化文本读取，逐字段解析 | 中 — 比二进制读取慢约 3-5x |

### 1.4 串行性能基线

| 测试 | 耗时 (ms) | 吞吐量 (MB/s) | 进程数 | 文件大小 |
|------|-----------|---------------|--------|---------|
| ReadRhog_Serial | 86.52 | 0.46 | 4 | ~41 KB |

**关键发现**：即使文件仅 41 KB，串行读取吞吐量仅 0.46 MB/s。瓶颈不在磁盘，而在通信延迟——每次 `Bcast` 和 `>>` 文本解析操作的开销主导了时间。对于大体系（数百 MB），延迟将成倍放大。

---

## 二、优化算法设计

### 2.1 核心思路

**MPI-IO 集体并行读取**：所有进程同时打开文件，各自直接读取需要的数据块，消除 rank 0 串行瓶颈和 Bcast 通信开销。

### 2.2 文件布局分析

当前 `write_rhog` 生成的二进制文件布局：

```
┌──────────────────────────────────────────────────────────────┐
│ Offset 0:                                                     │
│   [/size=3/] /gamma_only/ /ngm_g/ /nspin/ [/size=3/]        │
│    4B        4B           4B      4B       4B          = 20B │
│                                                               │
│   [/size=9/] /b11/../b33/ [/size=9/]                         │
│    4B        9×8B          4B                       = 80B    │
│                                                               │
│   header_total = 20 + 80 = 100 bytes                         │
├──────────────────────────────────────────────────────────────┤
│   [/size=3*ngm_g/]                                           │
│    4B                                                 = 4B   │
│   miller[0].x miller[0].y miller[0].z ...                    │
│   3×ngm_g × 4B                                      = 12*ngm_g│
│   [/size=3*ngm_g/]                                           │
│    4B                                                 = 4B   │
│   miller_total = 8 + 12 * ngm_g bytes                        │
├──────────────────────────────────────────────────────────────┤
│   for each spin:                                              │
│     [/size=ngm_g/]       4B                                  │
│     rhog[0..ngm_g-1]     ngm_g × 16B (complex<double>)       │
│     [/size=ngm_g/]       4B                                  │
│   spin_data_total = (8 + 16 * ngm_g) * nspin bytes           │
└──────────────────────────────────────────────────────────────┘
```

### 2.3 MPI-IO 并行读取算法

```
Algorithm: read_rhog_mpi(filename, pw_rhod, rhog)

Input:
  filename   — 二进制 rhog 文件路径
  pw_rhod    — PW_Basis 对象 (含 npw, npwtot, fftixy2ip 等)
  rhog       — 输出: 复数数组

Steps:

1. 所有进程集体打开文件:
   MPI_File_open(POOL_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)

2. 所有进程集体读取头部 (100 bytes):
   MPI_File_read_all(fh, header_buf, 100, MPI_BYTE, &status)
   // 全部进程获得相同的头部
   解析: size, gamma_only, npwtot, nspin, b1-b3

3. 校验头部一致性:
   if gamma_only_in != pw_rhod->gamma_only → error
   if npwtot_in > pw_rhod->npwtot → warning
   ...

4. 所有进程集体读取全部 Miller 索引:
   miller_offset = 100  // header 大小
   miller_bytes = 8 + 12 * npwtot_in  // 含 size 标记
   MPI_File_read_at_all(fh, miller_offset, miller_buf, miller_bytes, MPI_BYTE, &status)

5. 各进程本地构建 fftixyz2ig 映射表 (同原逻辑)

6. 计算 rhog 数据偏移:
   rhog_offset = 100 + miller_bytes

7. 所有进程集体读取全部 rhog 数据:
   rhog_bytes = (8 + 16 * npwtot_in) * nspin
   MPI_File_read_at_all(fh, rhog_offset, rhog_buf, rhog_bytes, MPI_BYTE, &status)

8. 各进程本地分发数据到 rhog[is][ig]:
   for is in 0..nspin:
     for iglobal in 0..npwtot_in:
       根据 miller 索引找到本进程对应的 ig
       if ig 属于本进程: rhog[is][ig] = rhog_buf[is * npwtot_in + iglobal]

9. 处理 nspin=2→4 转换 (同原逻辑)

10. MPI_File_close(&fh)
```

### 2.4 设计讨论：全量读取 vs 分片读取

**方案 A（全量集体读取）：** 所有进程读取全部文件内容。

| 优点 | 缺点 |
|------|------|
| 实现简单 | 每个进程需完整大小的缓冲区 |
| 无需复杂偏移计算 | 对大文件不友好 (GB 级) |
| 天然数据正确 | 每个进程读取相同内容 = 浪费 I/O |

**方案 B（分片读取 + 本地过滤）：** 每个进程只读自己的 rhog 数据。

| 优点 | 缺点 |
|------|------|
| I/O 量 = 文件大小 / nprocs | 复杂偏移计算 |
| 内存高效 | 需要预知 G-vector 到 rank 的映射 |

**选择方案 A（全量集体读取）作为首选**，原因：
1. rhog 文件通常不大（<100 MB），全量读取成本可接受
2. 实现简洁，不易出错
3. `MPI_File_read_all` 内部可以利用并行文件系统优化（实际 I/O 不需要每个进程都读相同字节）
4. 与读取后的数据分发逻辑（每个 rank 过滤自己的 G-vector）天然配合

### 2.5 算法伪代码（完整版）

```cpp
bool read_rhog_mpi(const std::string& filename,
                   const ModulePW::PW_Basis* pw_rhod,
                   std::complex<double>** rhog)
{
    ModuleBase::timer::start("ModuleIO", "read_rhog_mpi");

    MPI_File fh;
    int ret = MPI_File_open(POOL_WORLD, filename.c_str(),
                            MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (ret != MPI_SUCCESS) {
        ModuleBase::WARNING("ModuleIO::read_rhog", "Can't open file " + filename);
        return false;
    }

    // Step 1: Read header (100 bytes: 2×size + gamma_only + npwtot + nspin
    //                              + 2×size + 9×bvec)
    // All ranks read the same header collectively
    std::vector<char> header(100);
    MPI_File_read_all(fh, header.data(), 100, MPI_BYTE, MPI_STATUS_IGNORE);

    // Parse header
    int gamma_only_in, npwtot_in, nspin_in, size;
    double b1[3], b2[3], b3[3];
    parse_rhog_header(header.data(), size, gamma_only_in, npwtot_in,
                      nspin_in, size, b1, b2, b3);

    // Validate
    if (gamma_only_in != pw_rhod->gamma_only) {
        MPI_File_close(&fh);
        return false; // inconsistent
    }
    // Warnings for npwtot/nspin mismatch...

    // Step 2: Read Miller indices (all ranks)
    MPI_Offset miller_offset = 100;
    int miller_bytes = 8 + 12 * npwtot_in;
    std::vector<int> miller(npwtot_in * 3);
    // Skip the opening size marker (4 bytes), read miller data
    MPI_File_read_at_all(fh, miller_offset + 4,
                         miller.data(), 3 * npwtot_in, MPI_INT,
                         MPI_STATUS_IGNORE);

    // Step 3: Build fftixyz2ig map (local computation)
    std::vector<int> fftixyz2ig(pw_rhod->nxyz, -1);
    for (int ig = 0; ig < pw_rhod->npw; ++ig) {
        int isz = pw_rhod->ig2isz[ig];
        int iz = isz % pw_rhod->nz;
        int is = isz / pw_rhod->nz;
        int ixy = pw_rhod->is2fftixy[is];
        int ixyz = iz + pw_rhod->nz * ixy;
        fftixyz2ig[ixyz] = ig;
    }

    // Step 4: Read rhog data (all ranks)
    MPI_Offset rhog_offset = miller_offset + miller_bytes;

    // Read only what rank_in_pool 0 used to: the complex values between size markers
    // The format is: [size=ngm_g] [complex_data × npwtot_in] [size=ngm_g]
    // For each spin channel
    for (int is = 0; is < nspin_in; ++is) {
        // Skip the opening size marker (4 bytes), read complex data
        MPI_Offset data_offset = rhog_offset + 4; // skip size marker
        std::vector<std::complex<double>> rhog_in(npwtot_in);
        MPI_File_read_at_all(fh, data_offset,
                             rhog_in.data(), npwtot_in,
                             MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);

        // Distribute data to local rhog
        for (int i = 0; i < npwtot_in; ++i) {
            int ix = miller[i * 3];
            int iy = miller[i * 3 + 1];
            int iz = miller[i * 3 + 2];

            // Wrap negative indices (same logic as original)
            if (ix <= -(pw_rhod->nx + 1) / 2 || ix >= pw_rhod->nx / 2 + 1
                || iy <= -(pw_rhod->ny + 1) / 2 || iy >= pw_rhod->ny / 2 + 1
                || iz <= -(pw_rhod->nz + 1) / 2 || iz >= pw_rhod->nz / 2 + 1)
                continue;

            if (ix < 0) ix += pw_rhod->nx;
            if (iy < 0) iy += pw_rhod->ny;
            if (iz < 0) iz += pw_rhod->nz;

            int fftixy = iy + pw_rhod->fftny * ix;
            if (pw_rhod->RANK_IN_POOL == pw_rhod->fftixy2ip[fftixy]) {
                int fftixyz = iz + pw_rhod->nz * fftixy;
                int ig = fftixyz2ig[fftixyz];
                rhog[is][ig] = rhog_in[i];
            }
        }

        // Advance offset past data + closing size marker
        rhog_offset += 8 + 16 * npwtot_in; // 4B size + data + 4B size
    }

    // nspin=2 → 4 conversion (same logic)
    if (nspin_in == 2 && PARAM.inp.nspin == 4) {
        for (int ig = 0; ig < pw_rhod->npw; ++ig)
            rhog[3][ig] = rhog[1][ig];
        ModuleBase::GlobalFunc::ZEROS(rhog[1], pw_rhod->npw);
        ModuleBase::GlobalFunc::ZEROS(rhog[2], pw_rhod->npw);
    }

    MPI_File_close(&fh);
    ModuleBase::timer::end("ModuleIO", "read_rhog_mpi");
    return true;
}
```

### 2.6 与 read_rhog 的对比

| 操作 | 原 read_rhog | 新 read_rhog_mpi |
|------|-------------|-----------------|
| 文件打开 | rank 0 打开 | 所有进程 MPI_File_open |
| 头部读取 | rank 0 读, 8 次 Bcast | 集体 MPI_File_read_all |
| Miller 索引 | rank 0 读, 1 次大 Bcast | 集体 MPI_File_read_at_all |
| rhog 数据 | rank 0 读, N 次 Bcast | 集体 MPI_File_read_at_all |
| 数据分发 | Bcast 后各进程过滤 | 读后各进程直接过滤 |
| 通信次数 | 10+ 次（含大数组） | **0 次 Bcast** |
| I/O 模式 | rank 0 独占 | 所有进程共享 |

---

## 三、实现计划

### 3.1 需要修改/创建的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/source_io/module_chgpot/rhog_io.h` | 修改 | 新增 `read_rhog_mpi()` 声明 |
| `source/source_io/module_chgpot/rhog_io.cpp` | 修改 | 新增 `read_rhog_mpi()` 实现 |
| `source/source_io/test/read_rhog_mpi_test.cpp` | 修改 | 新增 MPI-IO 正确性 + 性能测试 |
| `source/source_io/CMakeLists.txt` | 无需修改 | rhog_io.cpp 已在 objects 列表中 |

### 3.2 实现步骤

**Step 1: 添加 `read_rhog_mpi()` 函数**
- 在 `rhog_io.h` 中声明
- 在 `rhog_io.cpp` 中实现核心 MPI-IO 逻辑
- 复用原 `read_rhog` 的数据分发逻辑（Miller→本地 ig 映射）

**Step 2: 验证与原实现的输出一致性**
- 使用相同输入文件，对比 `read_rhog` 和 `read_rhog_mpi` 输出

**Step 3: 更新测试**
- `MPIIOParallelReadIntegrity`：多进程 MPI-IO 读取，验证数据完整性
- `SingleVsMultiProcessConsistency`：对比 1 进程 vs 4 进程读取结果
- `Bench_ReadRhog_MPIIO_np{N}`：MPI-IO 读取性能基线
- `Bench_ReadRhog_Scaling`：进程数 1→2→4→8 加速比曲线

### 3.3 调用方式

```cpp
// 在需要读取 rhog 的模块中 (如 charge_init.cpp):
#ifdef __MPI
    bool ok = ModuleIO::read_rhog_mpi(filename, pw_rhod, rhog);
#else
    bool ok = ModuleIO::read_rhog(filename, pw_rhod, rhog);
#endif
```

或通过条件编译自动选择：在 `read_rhog` 内部检测 `__MPI` 并路由到 MPI-IO 版本。

---

## 四、预期性能提升

### 4.1 延迟/吞吐量预期

| 指标 | 当前（串行+Bcast） | MPI-IO 集体读取 | 提升 |
|------|-------------------|-----------------|------|
| 小文件 (41 KB) | 86.52 ms | ~10-20 ms | **4-8x** |
| 中等文件 (10 MB, ~10^5 PW) | ~1000 ms (估算) | ~50-100 ms | **10-20x** |
| 大文件 (100 MB, ~10^6 PW) | ~10000 ms (估算) | ~200-500 ms | **20-50x** |
| 通信量 | 全部 Bcast | 零 Bcast | **∞** |

**主要收益来源：**
1. **消除 Bcast 延迟**：8+ 次 `MPI_Bcast` 的延迟（每次 ~0.1-1 ms）× 进程数 → 全部消除
2. **并行 I/O 带宽聚合**：所有进程的磁盘带宽同时使用
3. **消除 rank 0 内存热点**：原方案 rank 0 需要分配 `npwtot` 完整数组进行 Bcast，新方案各进程直接读取

### 4.2 缩放预期

| 进程数 | 小文件读取时间 (预期) | 加速比 |
|--------|----------------------|--------|
| 1 (串行) | 86 ms | 1.0x |
| 2 | ~50 ms | 1.7x |
| 4 | ~20 ms | 4.3x |
| 8 | ~15 ms | 5.7x |

小文件（41KB）的加速主要来自消除 Bcast 延迟，而非并行 I/O 带宽（文件太小，I/O 时间可忽略）。

### 4.3 综合收益（Task 6 + Task 3 + Task 5）

优化后完整读写流水线（以 256^3 网格为例）：

```
写入: MPI-IO 并行二进制写入 (Task 3) + zlib 压缩 (Task 5) → 1-2 s
读取: MPI-IO 并行读取 (Task 6) + zlib 解压 (Task 5)       → 0.3-0.5 s
```

---

## 五、测试计划

### 5.1 新增正确性测试

| 测试名称 | 说明 |
|---------|------|
| `MPIIOParallelReadIntegrity` | 4 进程 MPI-IO 读取参考文件，每个 rank 验证自己拥有的 G-vector 数据正确 |
| `SingleVsMultiProcessConsistency` | np=1 读取结果 vs np=4 读取结果（每个 rank 收集自己的数据后 Allgather 对比） |
| `ReadRhogVsReadRhogMPIConsistency` | 原 `read_rhog` 与 新 `read_rhog_mpi` 对同一文件输出完全一致 |
| `MPIIONonexistentFile` | MPI-IO 读取不存在文件，正确返回 false |

### 5.2 新增性能测试

| 测试名称 | 说明 |
|---------|------|
| `Bench_ReadRhog_MPIIO_np4` | 4 进程 MPI-IO 读取性能基线 |
| `Bench_ReadRhog_MPIIO_np8` | 8 进程扩展性测试 |
| `Bench_ReadRhog_Scaling` | np=1,2,4,8 加速比曲线 |
| `Bench_ReadVsWrite_Balance` | 读/写时间比率（读写对称性分析） |
| `Bench_ReadRhog_SerialVsMPI` | 同数据量下串行 vs MPI-IO 直接对比 |

### 5.3 现有测试保留

`read_rhog_test.cpp` (4 个已有测试) 和 `read_rhog_mpi_test.cpp` (3 个已有测试) 全部保留，新增测试追加到 `read_rhog_mpi_test.cpp`。

---

## 六、风险与注意事项

1. **文件格式依赖性**：`read_rhog_mpi` 强依赖 `write_rhog` 的二进制格式。如果外部文件布局不同（如 Quantum ESPRESSO 生成的文件），需做兼容性适配。
2. **MPI-IO 集体操作**：`MPI_File_read_all` 要求通信域内所有进程调用。如果有进程提前返回（如 error 情况），需确保所有进程都执行到 `MPI_File_close`。
3. **文件不存在处理**：MPI-IO 打开失败时，部分 MPI 实现可能导致未定义行为。需在打开前用 `access()` 或 rank 0 预检查。
4. **大文件偏移**：对于 ngm_g > 10^6 的大体系，Miller 索引占 12 MB，rhog 数据占 16 MB/spin。使用 `MPI_Offset` (64-bit) 确保不溢出。
5. **`RANK_IN_POOL` 全局变量**：数据分发逻辑依赖 `pw_rhod->fftixy2ip` 和全局变量 `GlobalV::RANK_IN_POOL`。MPI-IO 版本保持此依赖，不改变分发逻辑。
