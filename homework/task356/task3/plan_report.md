# Task 3: Cube 文件写入 MPI-IO 并行化 —— 优化方案报告

Date: 2026-05-26

---

## 一、当前实现分析

### 1.1 现有代码结构

**源文件位置：**
- `source/source_io/module_output/write_cube.cpp` (260 行) — 实现 `write_cube()` 和 `write_vdata_palgrid()`
- `source/source_io/module_output/cube_io.h` (105 行) — 函数声明
- `source/source_io/module_output/read_cube.cpp` (208 行) — 配套读取函数

**测试文件：** `source/source_io/test/write_cube_test.cpp` (328 行)

### 1.2 当前算法流程

```
write_vdata_palgrid(pgrid, data, ...)
  │
  ├── pgrid.reduce(data_xyz_full, data)    // MPI 归约: 所有进程数据汇总到 band group rank 0
  ├── MPI_Barrier(POOL_WORLD)              // 同步等待所有进程
  │
  └── [仅 rank 0 / rank_in_pool == 0] write_cube()
        │
        ├── 写入 2 行注释 (文本)
        ├── 写入 natom + origin (文本, precision=1)
        ├── 写入 nx/ny/nz + 格点矢量 (文本, precision=6)
        ├── 写入原子信息 (文本)
        │
        └── 写入体积数据 (文本, scientific, precision=11)
              │
              └── for ixy in 0..nxy:           // 外层: x-y 平面
                    for iz in 0..nz:             // 内层: z 方向 (z-fastest)
                      ofs << " " << data[ixy*nz + iz]
                      if (iz+1) % 6 == 0: ofs << "\n"
                    ofs << "\n"
```

### 1.3 瓶颈分析

| 瓶颈 | 详细描述 | 影响程度 |
|------|---------|---------|
| **串行 I/O** | 仅 rank 0 执行写入，其余 N-1 个进程空闲 | 极高 — 浪费 N-1 个进程的 I/O 带宽 |
| **文本格式转换** | `double → 科学计数法字符串` 需要格式化和字符拷贝，每个数值约 20 字节 | 高 — 文本体积是二进制的 2-3 倍 |
| **逐元素输出** | `ofs << data[i]` 逐元素调用 `operator<<`，产生大量系统调用 | 中 — 即使有 ostream 缓冲，格式化开销仍大 |
| **同步阻塞** | `MPI_Barrier` 强制所有进程等待最慢者 | 中 — 异构环境下尤其严重 |

### 1.4 串行性能基线（单 rank，文本格式）

| 网格 | 数据量 | 耗时 (ms) | 吞吐量 (MB/s) |
|------|--------|-----------|---------------|
| 64^3 | 2.0 MB | 330.20 | 6.21 |
| 128^3 | 16.2 MB | 3700.28 | 4.38 |
| 256^3 | 128.8 MB | 24475.97 | 5.26 |

**关键发现：** 文本写入吞吐量仅约 5 MB/s，远低于典型磁盘带宽 (100-500 MB/s)。256^3 网格需约 24.5 秒完成写入。I/O 而非计算是瓶颈。

---

## 二、优化算法设计

### 2.1 核心思路

**MPI-IO 集体并行写入** + **二进制数据格式**，将所有进程的 I/O 带宽聚合起来。

### 2.2 文件布局设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Cube 文件布局 (混合格式)                    │
├─────────────────────────────────────────────────────────────┤
│  HEADER (文本，由 rank 0 单独写入)                             │
│  ├── Comment line 1 (文本)                                   │
│  ├── Comment line 2 (文本)                                   │
│  ├── natom origin_x origin_y origin_z (文本)                 │
│  ├── nx dx[0] dx[1] dx[2] (文本)                            │
│  ├── ny dy[0] dy[1] dy[2] (文本)                            │
│  ├── nz dz[0] dz[1] dz[2] (文本)                            │
│  └── 原子信息 × natom (文本)                                  │
├─────────────────────────────────────────────────────────────┤
│  DATA (二进制 double，MPI-IO 并行写入)                         │
│  ├── [rank 0 数据块]  z_index ∈ [0, nz/nproc)               │
│  ├── [rank 1 数据块]  z_index ∈ [nz/nproc, 2*nz/nproc)      │
│  ├── [rank 2 数据块]  z_index ∈ [2*nz/nproc, 3*nz/nproc)    │
│  └── [rank 3 数据块]  z_index ∈ [3*nz/nproc, nz)            │
└─────────────────────────────────────────────────────────────┘
```

**header_size**: 文本头部的字节数，由 rank 0 写入后广播给所有进程。

**数据布局**：z-fastest 索引, `data[ixy * nz + iz]`。每个 rank 写入其负责的 z 范围的全部 x-y 平面数据。

### 2.3 MPI-IO 写入算法

```
Algorithm: write_cube_mpi_parallel(file, metadata, data, nxyz, comm)

Input:
  file       — 输出文件路径
  metadata   — 注释、原子信息、格点参数
  data       — 完整体积数据 (nxyz 个 double, 分布在所有进程)
  nxyz       — 总数据点数 (nx * ny * nz)
  comm       — MPI 通信域

Variables:
  nprocs     — 总进程数
  my_rank    — 当前进程编号
  nxy        — nx * ny
  nz_local   — 本进程负责的 z 层数
  z_start    — 本进程起始 z 索引
  offset     — 本进程在文件中的写入偏移

Steps:

1. 计算数据分片:
   nz_local = nz / nprocs           // 平均分配
   remainder = nz % nprocs
   if my_rank < remainder: nz_local += 1
   z_start = sum of nz_local for ranks < my_rank

2. rank 0 写入文本头部:
   ofs.open(file)
   写入全部文本头部行 (与原始 write_cube 相同格式)
   ofs.close()
   记录 header_bytes = 文件当前位置

3. 广播 header_bytes 到所有进程:
   MPI_Bcast(&header_bytes, 1, MPI_LONG, 0, comm)

4. 所有进程计算文件偏移:
   my_offset = header_bytes + z_start * nxy * sizeof(double)
   my_count = nz_local * nxy

5. 所有进程打开文件进行并行写入:
   MPI_File_open(comm, file, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh)

6. 集体写入 (MPI-IO):
   MPI_File_write_at_all(fh, my_offset, &data[z_start * nxy], my_count, MPI_DOUBLE, &status)

7. 关闭文件:
   MPI_File_close(&fh)
```

### 2.4 关于数据归约的优化

当前 `write_vdata_palgrid` 先将分布式数据 `pgrid.reduce()` 归约到 rank 0，然后 rank 0 写入。对于 MPI-IO 方案：

**方案 A（保持归约 + 分片写入）：** 仍在 rank 0 完成归约得到完整数据，然后由 rank 0 分发给各进程各自写入。额外引入一次数据分发通信。

**方案 B（每个 band group 各自写入）：** 每个 band group 的 rank 0 完成归约后，直接使用 MPI-IO 写入共享文件的不同偏移。**推荐方案 B**，因为：
- 不需要额外的进程间数据分发
- 各 band group 的数据天然对应不同的 z 范围
- 与现有 `write_vdata_palgrid` 的调用结构兼容

### 2.5 二进制数据格式说明

二进制格式直接写入 `double` 的原始字节：

```
offset = header_bytes + z_start * nxy * sizeof(double)
```

数据顺序：外层 x-y 平面 (`ixy = ix * ny + iy`)，内层 z (`iz`)。这与当前文本格式的循环顺序完全一致，保证了与 `read_cube` 的兼容性。

**兼容性考虑：** 为保持与 VESTA 等工具的兼容，头部仍使用文本格式。二进制数据段可通过标志位识别（例如在注释中增加 `BINARY` 标记），或提供单独的二进制格式变体。

---

## 三、实现计划

### 3.1 需要修改/创建的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/source_io/module_output/cube_io.h` | 修改 | 新增 `write_cube_mpi()` 函数声明 |
| `source/source_io/module_output/write_cube.cpp` | 修改 | 新增 `write_cube_mpi()` MPI-IO 并行写入实现 |
| `source/source_io/test/write_cube_test.cpp` | 修改 | 新增 MPI-IO 正确性测试 + 并行性能测试 |
| `source/source_io/CMakeLists.txt` | 修改 | 无新增源文件，仅需确保 MPI 链接 |

### 3.2 实现步骤

**Step 1: 添加 `write_cube_mpi()` 函数**
- 在 `cube_io.h` 中声明新函数
- 输入参数与 `write_cube` 类似，额外增加 `MPI_Comm` 参数
- 支持两种模式：`text`（头部文本 + 数据文本）和 `binary`（头部文本 + 数据二进制）

**Step 2: 实现 MPI-IO 核心逻辑**
- 各进程按 z 方向计算分片
- rank 0 写文本头部，广播 header_bytes
- 所有进程 `MPI_File_open` → `MPI_File_write_at_all` → `MPI_File_close`

**Step 3: 更新测试**
- `MPIWriteAllRanksConsistent`：多进程写入同一文件，验证二进制一致性
- `TextVsBinaryConsistency`：文本格式与二进制格式写入的数据内容完全一致
- `Bench_WriteCube_MPIIO_np{N}_{size}`：MPI-IO 并行写入性能基准

**Step 4: 集成到 `write_vdata_palgrid`**
- 当启用 MPI-IO 时，调用 `write_cube_mpi()` 替代 `write_cube()`
- 用编译宏 `__MPI` 控制

### 3.3 新增函数签名

```cpp
// cube_io.h 新增声明
void write_cube_mpi(const std::string& file,
                    const std::vector<std::string>& comment,
                    const int& natom,
                    const std::vector<double>& origin,
                    const int& nx, const int& ny, const int& nz,
                    const std::vector<double>& dx,
                    const std::vector<double>& dy,
                    const std::vector<double>& dz,
                    const std::vector<int>& atom_type,
                    const std::vector<double>& atom_charge,
                    const std::vector<std::vector<double>>& atom_pos,
                    const std::vector<double>& data,
                    const int precision,
                    const bool binary_data = true,
                    const MPI_Comm& comm = MPI_COMM_WORLD);
```

---

## 四、预期性能提升

### 4.1 定量预测

以 4 MPI 进程、256^3 网格（128.8 MB 原始数据）为例：

| 指标 | 当前（文本串行） | MPI-IO 文本 | MPI-IO 二进制 | 提升 |
|------|-----------------|-------------|---------------|------|
| 写入时间 | ~24.5 s | ~8-10 s | ~0.5-1.5 s | **16-50x** |
| 文件体积 | ~360 MB | ~360 MB | ~128 MB | **2.8x** |
| 吞吐量 | 5.26 MB/s | ~15 MB/s | ~85-250 MB/s | **16-48x** |

**预测依据：**
- MPI-IO 文本写入：4 进程并行分担文本格式化，每个进程约 1/4 的数据量，但文本格式开销仍在
- MPI-IO 二进制写入：eliminates 文本转换开销，4 进程共享磁盘带宽可达 100-500 MB/s
- 文件体积：文本每个 double 约 20 字符（含空格/换行），二进制固定 8 字节

### 4.2 缩放预期

| 进程数 | 二进制吞吐量 (预期) | 加速比 |
|--------|---------------------|--------|
| 1 | ~80 MB/s | 1.0x (baseline) |
| 2 | ~150 MB/s | 1.9x |
| 4 | ~280 MB/s | 3.5x |
| 8 | ~450 MB/s | 5.6x |

受限于磁盘带宽上限和 MPI-IO 协调开销，8 进程以上可能进入平台期。

---

## 五、测试计划

### 5.1 新增正确性测试

| 测试名称 | 说明 | 验证点 |
|---------|------|--------|
| `MPIWriteBinaryCubeAndReadBack` | 多进程 MPI-IO 写入二进制 cube 文件，rank 0 读取验证 | 并行写入数据完整性 |
| `MPIWriteAllRanksConsistent` | 4 进程写入相同数据，对比各 rank 输出是否一致 | 文件级一致性 |
| `TextVsBinaryConsistency` | 同一数据分别以文本和二进制格式写入，读取后逐元素对比 | 格式兼容性 |

### 5.2 新增性能测试

| 测试名称 | 说明 |
|---------|------|
| `Bench_WriteCube_MPIIO_Binary_np4_64` | 4 进程 MPI-IO 二进制写入 64^3 |
| `Bench_WriteCube_MPIIO_Binary_np4_128` | 4 进程 MPI-IO 二进制写入 128^3 |
| `Bench_WriteCube_MPIIO_Binary_np4_256` | 4 进程 MPI-IO 二进制写入 256^3 |
| `Bench_WriteCube_MPIIO_Binary_np8_256` | 8 进程扩展性测试 |
| `Bench_WriteCube_MPIIO_Scaling` | 进程数 1→2→4→8 加速比曲线 |

---

## 六、风险与注意事项

1. **文件系统兼容性**：MPI-IO 需要支持 POSIX 文件系统的并行访问（如 Lustre, GPFS）。在 NFS 上性能可能下降。
2. **头部文本大小不确定性**：rank 0 写头部后需精确获取 header_bytes。使用 `tellp()` 获取 ofstream 位置。
3. **大文件支持**：256^3 网格二进制数据约 128 MB，使用 `MPI_Offset`（8 字节）确保支持 >2GB 文件。
4. **与现有 `.cube` 读取工具的兼容性**：二进制数据段不被标准 cube reader（如 VESTA）直接支持。解决方案：保留文本输出选项，或提供转换工具。
