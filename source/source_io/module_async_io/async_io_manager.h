#ifndef ASYNC_IO_MANAGER_H
#define ASYNC_IO_MANAGER_H

#include "io_buffer.h"
#include "io_task.h"

#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <string>
#include <vector>

// ============================================================================
// AsyncIOManager: 异步 I/O 管理器 — 生产者-消费者模式
//
// 设计目标:
//   使电荷密度文件的写入/读取与主计算线程重叠执行，
//   减少 I/O 等待时间。
//
// 工作流程:
//   主线程 (生产者):
//     1. 构建 IOBuffer (包含待写入的数据和元信息)
//     2. 调用 submit() 将任务提交到队列
//     3. 立即返回，继续计算
//
//   I/O 工作线程 (消费者):
//     1. 等待队列中有任务
//     2. 取出任务并执行阻塞 I/O
//     3. 标记完成
//
// 同步机制:
//   - std::mutex: 保护任务队列的并发访问
//   - std::condition_variable: 生产者→消费者通知
//   - std::atomic<bool>: 控制线程生命周期
//
// 线程安全:
//   - submit() 和 wait_all() 可从多个线程调用
//   - 工作线程是唯一的消费者
//
// 使用示例:
//   AsyncIOManager& mgr = AsyncIOManager::create();
//   mgr.start();
//
//   // 在需要写入的地方:
//   IOBuffer buf = IOBuffer::make_cube_write(std::move(data), fn, is, istep);
//   buf.set_cube_header(comment, nat, origin, nx, ny, nz, dx, dy, dz, ...);
//   mgr.submit(std::move(buf));
//
//   // 在需要数据保证写入完成的地方:
//   mgr.wait_all();
//
//   // 程序结束前:
//   mgr.stop();
// ============================================================================

/// @brief 异步 I/O 管理器 (单例模式)
class AsyncIOManager
{
  public:
    // ======================== 生命周期管理 ========================

    /// @brief 获取全局唯一实例
    static AsyncIOManager& instance()
    {
        static AsyncIOManager inst;
        return inst;
    }

    // 禁止拷贝/赋值
    AsyncIOManager(const AsyncIOManager&) = delete;
    AsyncIOManager& operator=(const AsyncIOManager&) = delete;

    /// @brief 启动 I/O 工作线程
    /// @param max_queue_size  队列最大容量 (0 表示无限制)
    void start(size_t max_queue_size = 4);

    /// @brief 等待所有已提交的任务完成
    void wait_all();

    /// @brief 获取当前待处理任务数
    size_t pending_count() const;

    /// @brief 检查 I/O 工作线程是否正在运行
    bool is_running() const { return running_.load(); }

    /// @brief 检查队列是否为空且没有任务正在执行
    bool is_idle() const;

    /// @brief 检查队列是否有空位可提交 (非阻塞)
    /// @return true 表示队列未满，可以提交新任务
    bool can_submit() const;

    /// @brief 优雅停止 I/O 工作线程 (等待所有任务完成)
    void stop();

    /// @brief 析构函数 — 自动等待并停止工作线程
    ~AsyncIOManager();

    // ======================== 任务提交 ========================

    /// @brief 提交一个 Cube 写入任务 (最常用的异步写入接口)
    /// @param buf  包含数据和元信息的缓冲区 (move 语义)
    /// @return true 表示提交成功, false 表示队列满
    bool submit_cube_write(IOBuffer&& buf);

    /// @brief 提交一个二进制写入任务
    /// @param buf  包含数据和元信息的缓冲区 (move 语义)
    /// @return true 表示提交成功, false 表示队列满
    bool submit_binary_write(IOBuffer&& buf);

    /// @brief 提交一个通用 I/O 任务
    /// @param task  任务对象 (move 语义)
    /// @return true 表示提交成功, false 表示队列满
    bool submit_task(std::unique_ptr<IIOTask> task);

    /// @brief 提交一个 rhog 二进制读取任务
    /// @param buf  包含文件路径和预分配空间的缓冲区
    /// @return true 表示提交成功, false 表示队列满
    bool submit_rhog_read(IOBuffer&& buf);

    // ======================== 读取结果获取 ========================

    /// @brief 获取一个已完成的任务 (如果队列非空)
    /// @return 已完成任务的缓冲区, 如果无已完成任务则返回空 IOBuffer
    IOBuffer pop_completed();

    /// @brief 等待并获取下一个已完成的任务 (阻塞直到有任务完成)
    /// @return 已完成任务的缓冲区
    IOBuffer wait_next_completed();

    /// @brief 获取已完成任务数 (非阻塞查询)
    size_t completed_count() const;

    // ======================== 性能统计 ========================

    /// @brief 获取总提交任务数
    size_t total_submitted() const { return stats_total_submitted_.load(); }

    /// @brief 获取已完成任务数
    size_t total_completed() const { return stats_total_completed_.load(); }

    /// @brief 获取因队列满而被拒绝的任务数
    size_t total_rejected() const { return stats_total_rejected_.load(); }

    /// @brief 重置性能计数器
    void reset_stats();

  private:
    // ======================== 私有构造 ========================
    AsyncIOManager() = default;

    // ======================== 工作线程逻辑 ========================

    /// @brief I/O 工作线程主循环
    void io_loop();

    // ======================== 成员变量 ========================

    // ---- 线程管理 ----
    std::thread io_worker_;                 ///< 独立 I/O 工作线程
    std::atomic<bool> running_{false};      ///< 控制线程运行

    // ---- 任务队列 (生产者-消费者) ----
    std::queue<std::unique_ptr<IIOTask>> task_queue_;  ///< 待处理任务队列
    size_t max_queue_size_ = 4;                         ///< 队列最大容量

    // ---- 同步原语 ----
    mutable std::mutex queue_mutex_;         ///< 保护 task_queue_ 和 completed_queue_ 的并发访问
    std::condition_variable cv_;             ///< 任务到达 / 完成通知
    std::atomic<size_t> in_flight_tasks_{0}; ///< 正在执行(已出队但未完成)的任务数

    // ---- 完成队列 (I/O 工作线程写入, 主线程消费) ----
    std::queue<IOBuffer> completed_queue_;   ///< 已完成任务的缓冲区队列

    // ---- 性能统计 ----
    std::atomic<size_t> stats_total_submitted_{0};   ///< 总提交任务数
    std::atomic<size_t> stats_total_completed_{0};   ///< 总完成任务数
    std::atomic<size_t> stats_total_rejected_{0};    ///< 总被拒绝任务数
};

#endif // ASYNC_IO_MANAGER_H
