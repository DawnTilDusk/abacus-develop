# Task 3: Cube 文件 MPI-IO 并行写入 —— 优化实现报告

Date: 2026-05-26

---

## 一、实现概要

### 1.1 实际完成内容

| 内容 | 状态 |
|------|------|
| `write_cube_mpi()` — MPI-IO 并行写入函数 | 完成 |
| MPI file view 实现跨 rank z-slice 交替写入 | 完成 |
| binary marker (CIPM) 文件格式标识 | 完成 |
| `read_cube` 自动识别 CIPM 格式 | 完成 |
| 正确性/性能测试 (10 个测试, 全部通过) | 完成 |

### 1.2 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `source/source_io/module_output/write_cube.cpp` | 修改 | 新增 `write_cube_mpi()` 函数 |
| `source/source_io/module_output/read_cube.cpp` | 修改 | 新增 CIPM 二进制格式检测 |
| `source/source_io/module_output/cube_io.h` | 修改 | 新增 `write_cube_mpi()` 声明, 添加 `<mpi.h>` |
| `source/source_io/test/write_cube_test.cpp` | 修改 | 新增 MPI-IO 正确性 + 性能测试 |
| `source/source_io/test/CMakeLists.txt` | 修改 | 添加 zlib 链接 |

---

## 二、核心算法实现

### 2.1 算法设计

**write_cube_mpi() 流程：**

```
1. 所有 rank 计算 z 分片
   nz_local = nz / nprocs  (平均分配)
   余数分配给前 remainder 个 rank
   z_start = sum of nz_local for ranks < my_rank

2. Rank 0 生成文本头部字符串 (ostringstream)
   → 包含注释、原子信息、格点矢量
   → 不含数据行 (数据用二进制写入)

3. MPI_Bcast header_bytes 到所有 rank

4. 所有 rank 集体打开文件 (MPI_File_open)

5. Rank 0 写入头部 + CIPM marker
   MPI_File_write_at(fh, 0, header_str, ..., MPI_CHAR)
   MPI_File_write_at(fh, header_bytes, &CIPM_MARKER, 1, MPI_UINT32_T)

6. MPI_Barrier (确保头部写完)

7. 所有 rank 设置 strided file view (MPI_File_set_view)
   - 使用 MPI_Type_vector(nxy, nz_local, nz, MPI_DOUBLE) 创建视图
   - 每个 rank 写入 nz_local 个 z 值，步长 = nz

8. 所有 rank 集体写入 (MPI_File_write_all)
   各 rank 写入自己的 nxy × nz_local 个 double

9. MPI_File_close
```

**代码位置：** `write_cube.cpp:299-402`

### 2.2 文件布局

```
┌────────────────────────────────────────────────────┐
│ 文本头部 (同标准 Cube 格式)                          │
│   Comment line 1                                    │
│   Comment line 2                                    │
│   natom origin_x origin_y origin_z                  │
│   nx dx[0] dx[1] dx[2]                             │
│   ny dy[0] dy[1] dy[2]                             │
│   nz dz[0] dz[1] dz[2]                             │
│   原子信息 × natom                                   │
├────────────────────────────────────────────────────┤
│ CIPM marker: 0x4D504943 (4 字节, "CIPM" LE)        │
├────────────────────────────────────────────────────┤
│ 二进制数据 (nxyz 个 double, z-fastest 排序, LE)      │
│   [ixy=0, z=0..nz-1]                               │
│   [ixy=1, z=0..nz-1]                               │
│   ...                                               │
│ MPI-IO: 每个 rank 通过 file view 写入自己的 z 切片    │
└────────────────────────────────────────────────────┘
```

### 2.3 MPI File View 设计

这是本实现的核心设计点。z-fastest 数据布局要求所有 `nz` 个值在文件中连续排列。每个 rank 拥有其中 `nz_local` 个连续 z 值。

对 rank r (z_start 到 z_start+nz_local-1)：

```
MPI_Type_vector(nxy,        // 块数 (每个 ixy 一个块)
                nz_local,   // 每块的元素数
                nz,         // 步长 (跳过其他 rank 的 nz_local 值)
                MPI_DOUBLE,
                &filetype);
```

设文件视图偏移：`data_offset + z_start * sizeof(double)`

效果：rank r 写入的 buffer 被映射到文件位置：
- 位置 0: ixy=0, z=z_start (映射到文件位置 z_start)
- 位置 nz_local: ixy=1, z=z_start (映射到文件位置 nz + z_start)
- 位置 2*nz_local: ixy=2, z=z_start (映射到文件位置 2*nz + z_start)

### 2.4 read_cube 自动检测

```cpp
// 读取头部后, peek 下一字节
// 标准文本数据以数字/减号/小数点开头
// 'Z' → zlib 压缩 (ZCMP magic)
// 'C' → MPI 二进制 (CIPM marker 第一字节)
```

CIPM marker = 0x4D504943 ("CIPM" 小端序), 字节序为 `[0x43, 0x49, 0x50, 0x4D]` = `['C', 'I', 'P', 'M']`。首字节 'C' 永不合法出现在文本数字开头，因此可安全区分。

**代码位置：** `read_cube.cpp:203-258`

---

## 三、性能测试结果

### 3.1 串行文本写入基线 (单 rank)

| 网格 | 数据量 | 耗时 (ms) | 吞吐量 (MB/s) |
|------|--------|-----------|---------------|
| 64^3 | 2.0 MB | 283.7 | 7.2 |
| 128^3 | 16.2 MB | 1919.9 | 8.4 |
| 256^3 | 128.8 MB | 16748.8 | 7.7 |

### 3.2 MPI-IO 二进制并行写入 (4 ranks)

| 网格 | 耗时 (ms) | 吞吐量 (MB/s) | vs 串行文本 |
|------|-----------|---------------|------------|
| 256^3 | ~2,300 | ~56 | **7.3x** |

**分析：**
- 吞吐量从文本格式的 ~7.7 MB/s 提升至二进制 MPI-IO 的 ~56 MB/s
- 7.3x 加速来自三个方面：
  1. **二进制格式消除格式化开销**：8 字节/double vs 文本 ~20 字节/double (2.5x)
  2. **MPI-IO 并行 I/O**：4 个 rank 并行写入 (理论 4x)
  3. **消除文本转换**：无需 double → 科学计数法字符串转换
- 实际加速略低于理论值，受限于测试环境的文件系统并行写入能力

### 3.3 正确性验证

| 测试 | 结果 | 说明 |
|------|------|------|
| `MPIWriteBinaryCubeAndReadBack` | PASS (4/4 ranks) | 4 进程 MPI-IO 写入 → rank 0 读取验证 |
| `TextVsBinaryConsistency` | PASS (4/4 ranks) | 文本 vs 二进制输出内容完全一致 (误差 < 1e-6) |
| `Bench_WriteCube_MPIIO_Binary_256` | PASS | 256^3 MPI-IO 性能基准 |

---

## 四、设计决策

1. **MPI file view 而非 scatter+gather**：直接用 MPI file view 实现 strided write，避免了额外的数据通信步骤，最大化利用并行文件系统性能。

2. **CIPM marker**：自定义 4 字节文件标记 (0x4D504943)，使 `read_cube` 可自动识别 MPI 二进制格式，无需修改 Cube 注释行。

3. **文本头部保留**：保持标准 Cube 格式头部，VESTA 可读取元数据信息（网格大小、原子位置等）。仅数据段使用二进制格式。

4. **无 DELETE_ON_CLOSE**：最初实现中误用了 `MPI_MODE_DELETE_ON_CLOSE`，已在最终版本中移除，使用标准 `MPI_MODE_WRONLY | MPI_MODE_CREATE`。
