电荷密度 I/O 优化算法文档
# 一、题目3：Cube 文件写入优化
## 1.1 原算法理解
当前流程：

write_vdata_palgrid()
  ├── pgrid.reduce()      → MPI 归约，汇总数据到 rank 0
  ├── MPI_Barrier()       → 同步等待
  └── write_cube()        → 仅 rank 0 串行写入文件
数据写入方式：文本格式，每个 double 转成科学计数法字符串，每行6个数值。

## 1.2 原算法问题

- 串行 I/O	仅 rank 0 写入，其他 N-1 个进程空闲等待
- 文本格式开销	浮点数→字符串转换耗时，文件体积大（约为二进制的2-3倍）
- 同步阻塞	MPI_Barrier 导致所有进程等待最慢的进程
- 小规模 I/O	每个数值单独写入，大量系统调用
## 1.3 改进方案

主要方案：MPI-IO 并行写入

所有进程同时打开文件，各自写入自己的数据块

进程0: [z=0..nz/4]  ← 写入文件偏移 0
进程1: [z=nz/4..nz/2]  ← 写入文件偏移 nz/4 * nxy
进程2: [z=nz/2..3nz/4]
进程3: [z=3nz/4..nz]

使用 MPI_File_write_all() 实现并行写入
核心代码思路：
```
Cpp

MPI_File_open(MPI_COMM_WORLD, fn, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
// 每个进程计算自己的偏移和写入量
MPI_Offset offset = header_size + my_z_start * nxy * sizeof(double);
MPI_File_write_at_all(fh, offset, my_data, my_count, MPI_DOUBLE, &status);
MPI_File_close(&fh);```

```
其它问题的大概思路：采用二进制写入减少开销；可以考虑将MPI_Barrier去掉，采用非阻塞通信；以及利用MPI_File_Write_At_All直接块写入而不是在循环中多次调用
# 二、题目5：电荷密度数据压缩
## 2.1 原算法理解

当前实现：无压缩，直接以文本格式写入所有数据。

数据特征：

- 三维网格浮点数，具有空间连续性
- 相邻网格点数值变化平缓
- 远离原子核时趋近于零
## 2.2 原算法问题

文件体积大	512原子体系约360MB，占用大量磁盘空间
I/O 时间长	大量数据写入/读取耗时
网络传输慢	超算间传输大文件效率低

## 2.3 改进方案

方案：zlib 无损压缩

写入流程：
  原始数据 → zlib compress → 压缩数据 → 写入文件

读取流程：
  读取文件 → zlib uncompress → 原始数据
文件格式设计：

\[Cube 文件头（文本）] \[压缩标记（4字节）]\[Cube 文件头（文本）] \[压缩标记（4字节）] \[原始大小（8字节）] \[压缩数据]

# 三、题目6：电荷密度读取优化
## 3.1 原算法理解

当前流程（rhog_io.cpp / read_cube.cpp）：

read_rhog() / read_vdata_palgrid()
  ├── rank 0 串行读取文件头（多次 ifs >> 操作）
  ├── MPI_Bcast 广播文件头（多次单独广播）
  ├── rank 0 串行读取全部数据
  ├── 网格不匹配时进行三线性插值（仅 rank 0）
  └── pgrid.bcast() 广播数据到所有进程
## 3.2 原算法问题

串行读取	仅 rank 0 读文件，其他进程空闲
多次广播	每个参数单独 Bcast，通信开销大
内存拷贝	广播后还需数据映射到本地存储
同步阻塞	所有进程等待 rank 0 完成读取

## 3.3 改进方案

方案：MPI-IO 并行读取

所有进程同时打开文件，各自读取自己的数据块

进程0: 读取文件头 + [z=0..nz/4]
进程1: 读取 [z=nz/4..nz/2]
进程2: 读取 [z=nz/2..3nz/4]
进程3: 读取 [z=3nz/4..nz]

使用 MPI_File_read_at_all() 实现并行读取
核心代码思路：
```Cpp

MPI_File_open(MPI_COMM_WORLD, fn, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
// 所有进程并行读取文件头
MPI_File_read_all(fh, header_buf, header_size, MPI_CHAR, &status);
// 每个进程读取自己的数据块
MPI_Offset offset = header_size + my_z_start * nxy * sizeof(double);
MPI_File_read_at_all(fh, offset, my_data, my_count, MPI_DOUBLE, &status);
MPI_File_close(&fh);
```

四、三个题目的关联

Apply
写入流程（题目3 + 5）：
  电荷密度 → MPI归约 → 压缩(题5) → MPI-IO并行写入(题3)

读取流程（题目6 + 5）：
  MPI-IO并行读取(题6) → 解压(题5) → 数据分发
