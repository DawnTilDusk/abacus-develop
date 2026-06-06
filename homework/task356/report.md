# 题目4 异步 I/O 与计算重叠 — 实现报告

## 项目信息

| 项目 | 内容 |
|------|------|
| **题目** | 异步 I/O 与计算重叠 |
| **难度** | ⭐⭐⭐ |
| **涉及模块** | `source/source_io/module_async_io/` (新建) |
| **修改文件** | `write_cube.cpp`, `cube_io.h`, `source_io/CMakeLists.txt` |

---

## 一、问题背景与现有瓶颈

在 ABACUS 中，电荷密度写入的调用链为：

```
Charge::save_rho_before_sum_band()
    ↓
ModuleIO::write_vdata_palgrid()
    ↓
pgrid.reduce()          ← MPI 归约 (所有进程参与)
    ↓
MPI_Barrier             ← 所有进程等待 (瓶颈①)
    ↓
write_cube()            ← 仅 rank 0 写入 (瓶颈②)
```

### 关键性能瓶颈

| 瓶颈 | 位置 | 描述 |
|------|------|------|
| **MPI 同步阻塞** | `write_cube.cpp:53-58` | 所有进程在 `MPI_Barrier` 等待 rank 0 完成写入 |
| **串行 I/O** | `write_cube.cpp:64-176` | 仅 rank 0 执行文件写入，其他进程空闲 |
| **大量格式化字符串操作** | `write_cube.cpp:247-258` | 每行格式化输出，CPU 开销大 |
| **无计算重叠** | — | I/O 期间主计算线程被阻塞 |

---

## 二、设计方案

### 2.1 架构概览

采用**生产者-消费者模式**，引入独立 I/O 工作线程：

```
主计算线程 (生产者)                         I/O 工作线程 (消费者)
    │                                           │
    │  1. MPI 归约 (同步)                       │
    │  2. 构建 IOBuffer                         │
    │  3. submit() ──── queue ────────►          │  1. 取出任务
    │  4. 立即返回 ◄────── notify ─────           │  2. 执行 write_cube()
    │  5. 继续计算                               │  3. 标记完成
    │                                           │
    │  必要时调用 wait_all()                     │
```

### 2.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `IOBuffer` | `io_buffer.h` | 封装待写入数据+元信息，支持 move 语义 |
| `IIOTask` | `io_task.h` | 抽象 I/O 任务基类 (策略模式) |
| `CubeWriteTask` | `io_task.h` | Cube 文本格式写入实现 |
| `BinaryWriteTask` | `io_task.h` | 二进制格式写入实现 |
| `AsyncIOManager` | `async_io_manager.h/cpp` | 核心管理器, 管理线程+队列+同步 |

### 2.3 同步机制

| 机制 | C++11 工具 | 用途 |
|------|-----------|------|
| **互斥锁** | `std::mutex` | 保护任务队列的并发访问 |
| **条件变量** | `std::condition_variable` | 生产者→消费者任务到达通知 |
| **原子标志** | `std::atomic<bool>` | 控制线程生命周期 |
| **Move 语义** | `std::move` / `unique_ptr` | 零拷贝传递大数据 |

---

## 三、文件变更清单

### 3.1 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `source/source_io/module_async_io/io_buffer.h` | 186 | I/O 缓冲区的数据结构和元信息 |
| `source/source_io/module_async_io/io_task.h` | 223 | 抽象任务基类和具体任务实现 |
| `source/source_io/module_async_io/async_io_manager.h` | 118 | 异步 I/O 管理器声明 (单例) |
| `source/source_io/module_async_io/async_io_manager.cpp` | 166 | 管理器实现: 线程、队列、同步 |
| `source/source_io/module_async_io/CMakeLists.txt` | 28 | 模块构建配置 |
| `source/source_io/module_async_io/test/test_async_io.cpp` | 301 | 单元测试 (8个测试用例) |
| `source/source_io/module_async_io/test/CMakeLists.txt` | 13 | 测试构建配置 |

### 3.2 修改文件

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `source/source_io/module_output/cube_io.h` | 新增声明 | 添加 `write_vdata_palgrid_async()` 声明 |
| `source/source_io/module_output/write_cube.cpp` | 新增实现 | 添加 `write_vdata_palgrid_async()` 实现 |
| `source/source_io/CMakeLists.txt` | 添加子目录 | 加入 `module_async_io` 子目录 |

---

## 四、核心实现详解

### 4.1 `AsyncIOManager` 类

```cpp
class AsyncIOManager {
    // --- 生命周期 ---
    static AsyncIOManager& instance();   // 单例
    void start(size_t max_queue_size=4); // 启动 I/O 线程
    void stop();                         // 优雅停止
    void wait_all();                     // 等待所有任务完成

    // --- 任务提交 ---
    bool submit_cube_write(IOBuffer&& buf);   // 提交 Cube 写入
    bool submit_binary_write(IOBuffer&& buf); // 提交二进制写入
    bool submit_task(std::unique_ptr<IIOTask>); // 通用提交

    // --- 状态查询 ---
    bool is_running() const;
    bool is_idle() const;
    size_t pending_count() const;
};
```

### 4.2 工作线程流程 (`io_loop`)

```
while (running_) {
    lock(queue_mutex_);
    cv_.wait(lock, []{ return !queue.empty() || !running_; });
    if (!running_ && queue.empty()) break;
    task = pop(queue);
    unlock(queue_mutex_);

    task->execute();  // 在锁外执行阻塞 I/O

    lock(queue_mutex_);
    stats_completed_++;
    cv_.notify_all();  // 通知可能的 wait_all()
}
```

### 4.3 写入路径对比

| 步骤 | 同步版本 (`write_vdata_palgrid`) | 异步版本 (`write_vdata_palgrid_async`) |
|------|---------------------------------|----------------------------------------|
| MPI 归约 | `pgrid.reduce()` | `pgrid.reduce()` (相同) |
| 同步 | `MPI_Barrier` (所有进程等待) | **跳过 Barrier** |
| 文件写入 | `write_cube()` (rank 0, 阻塞) | **提交到 AsyncIOManager** |
| 返回 | 文件写完才返回 | **立即返回** |
| 其他进程 | 等待 Barrier | **直接返回** |

---

## 五、使用方式

### 5.1 初始化

在程序启动时 (例如 `main.cpp` 或 `ESolver::init()`)：

```cpp
#include "source_io/module_async_io/async_io_manager.h"

// 启动异步 I/O 管理器，设置队列容量为 4
AsyncIOManager::instance().start(4);
```

### 5.2 写入电荷密度

在需要输出 Cube 文件的位置，将 `write_vdata_palgrid()` 替换为：

```cpp
ModuleIO::write_vdata_palgrid_async(
    pgrid, rho[is], is, nspin, istep,
    fn, ef, ucell, precision, out_fermi, reduce_all_pool
);
// 调用后立即返回，文件由后台线程写入
```

### 5.3 确保写入完成

在程序退出、或需要立即读取刚写入的文件时：

```cpp
// 等待所有异步 I/O 任务完成
AsyncIOManager::instance().wait_all();
```

### 5.4 程序退出

```cpp
// 停止 I/O 工作线程 (会等待所有任务完成)
AsyncIOManager::instance().stop();
```

---

## 六、单元测试

### 6.1 测试用例

| 测试 | 文件 | 描述 |
|------|------|------|
| `StartStop` | `test_async_io.cpp` | 验证管理器启停生命周期 |
| `DoubleStart` | `test_async_io.cpp` | 验证重复启动的安全性 |
| `CubeWriteAndVerify` | `test_async_io.cpp` | 异步写入 Cube 文件并验证数据完整 |
| `BinaryWriteAndVerify` | `test_async_io.cpp` | 异步写入二进制文件并验证数据完整 |
| `MultipleTasks` | `test_async_io.cpp` | 多文件写入的顺序和完整性 |
| `QueueFullRejection` | `test_async_io.cpp` | 队列满时的行为 |
| `ErrorPropagation` | `test_async_io.cpp` | 错误路径处理 |
| `StateQueries` | `test_async_io.cpp` | `is_idle()` 和 `pending_count()` 状态查询 |

### 6.2 运行测试

```bash
ctest -R MODULE_IO_async_io -V
```

---

## 七、性能预期

| 指标 | 优化前 | 优化后 (预期) | 提升 |
|------|--------|-------------|------|
| 每次写入的时间 | T_sync | T_async ≈ 0 (主线程) | 主线程被阻塞时间降至 ~0 |
| 其他进程等待时间 | ~T_sync | ~0 | MPI_Barrier 被移除 |
| 大体系 (512原子) | ~360 MB 写入 | 后台写入 | 计算可继续 |

**理论加速比**: 在 I/O 占总时间 30-40% 的大体系上，异步 I/O 可使有效 I/O 等待时间降低 > 50%。

---

## 八、注意事项

1. **该模块完全独立于 MPI**：不依赖任何 `MPI_*` 调用，因此可在非 MPI 构建中正常使用。
2. **线程安全**：所有共享数据通过 `std::mutex` 保护，工作线程是唯一的消费者。
3. **异常安全**：I/O 工作线程捕获所有异常并通过 `IOBuffer::error_message` 传播。
4. **队列入满回退**：当队列满时，`submit()` 返回 false，调用者应回退到同步写入。
5. **不修改现有接口**：原有的 `write_vdata_palgrid()` 保持不变，保持完全向后兼容。

---

## 九、与现有代码的关系

```
                    charge_init.cpp / charge.cpp
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
        write_vdata_palgrid()    write_vdata_palgrid_async()   ← 新增
                    │                 │
                    │           ┌─────┴──────┐
                    ▼           ▼            ▼
              write_cube()   write_cube()  AsyncIOManager   ← 新增
                    │           │            │
                    ▼           ▼            ▼
               std::ofstream  std::ofstream  I/O Worker Thread
```

- 蓝色表示原有代码，绿色表示新增代码
- `AsyncIOManager` 完全独立，不依赖 ABACUS 的 MPI/并行框架
- `CubeWriteTask` 内部调用等效于 `write_cube()` 的写入逻辑

---

## 十、扩展方向

1. **环形缓冲**：对于 MD 模拟等连续多步输出场景，可将双缓冲扩展为环形缓冲。
2. **MPI-IO 集成**：将异步 I/O 与 MPI-IO 结合，实现真正的并行写入。
3. **压缩支持**：在工作线程中集成 zlib/SZ 压缩，进一步减少 I/O 时间。
4. **优先级队列**：区分关键写入 (检查点) 和非关键写入 (可视化文件)，高优先级任务先执行。
