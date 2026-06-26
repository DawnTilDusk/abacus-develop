#ifndef OMP_TASK_MANAGER_H
#define OMP_TASK_MANAGER_H

// ============================================================================
// OMPTaskManager: 基于 OpenMP 4.0 task+depend 的异步 I/O 管理器
//
// 设计目标:
//   用 #pragma omp task depend(in/out) 替代 std::thread 线程池,
//   实现计算与 I/O 重叠。所有线程来自 #pragma omp parallel 区域。
//
// 核心机制:
//   1. parallel + single: 单线程进入, spawn 出 N 个 I/O worker task,
//      然后执行主计算 (Driver::init → SCF 循环)。
//   2. 递归任务链 (io_worker_chain): 每个 I/O worker task 取一个任务,
//      执行为 omp task, 完成后递归生成下一个自己, 链表式持续运行。
//   3. depend(inout: global_serial_dep_): SERIALIZE_ALL 任务声明
//      同一个 depend 变量, OpenMP 运行时保证它们串行化;
//      INDEPENDENT 任务不加 depend, 多个 workers 并行执行。
//
// 约束:
//   - C++11 标准 + OpenMP 4.0
//   - 不用 std::thread (所有线程由 OpenMP 运行时管理)
//   - header-only, 新文件在独立目录 omp_task/, 不影响现有实现
//
// 与 AsyncIOManager (std::thread) 的接口兼容:
//   start/stop → init（在 parallel 区域外调用）
//   submit_*   → submit_*（在 parallel single 区域内调用）
//   wait_all   → signal_done + 隐式 taskwait
// ============================================================================

#include "io_buffer.h"
#include "io_task.h"

#include <memory>
#include <mutex>            // C++11: 保护队列
#include <condition_variable> // C++11: 任务到达通知
#include <queue>             // C++11: 任务队列
#include <atomic>            // C++11: 状态标志
#include <string>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

/// @brief 基于 OpenMP 4.0 task + depend 的异步 I/O 管理器 (单例)
class OMPTaskManager
{
  public:
    // ======================== 生命周期 ========================

    static OMPTaskManager& instance()
    {
        static OMPTaskManager inst;
        return inst;
    }

    OMPTaskManager(const OMPTaskManager&) = delete;
    OMPTaskManager& operator=(const OMPTaskManager&) = delete;

    /// @brief 初始化管理器 (进入 parallel 区域前调用)
    /// @param num_workers  I/O worker 链数量
    void init(size_t num_workers = 4);

    /// @brief 在 parallel single 区域内调用, spawn 出 N 个 I/O worker task
    ///        调用方 THEN 执行主计算, I/O workers 与之并发
    void spawn_io_workers();

    /// @brief 通知 I/O workers: 主线程不再提交新任务
    ///        workers 在清空队列后自行终止
    void signal_done();

    /// @brief 强制停止 (异常退出路径)
    void signal_stop();

    /// @brief 重置为未初始化状态 (仅用于单元测试)
    void reset_for_testing()
    {
        running_.store(false, std::memory_order_release);
        done_.store(false);
        stop_.store(false);
        in_flight_.store(0);
        submitted_.store(0);
        completed_.store(0);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            while (!task_queue_.empty()) task_queue_.pop();
            while (!completed_queue_.empty()) completed_queue_.pop();
        }
    }

    /// @brief 阻塞等待所有已提交任务完成
    void wait_all();

    bool is_running() const { return running_.load(std::memory_order_acquire); }

    // ======================== 任务提交 ========================

    /// @brief 提交 Cube 写入任务
    bool submit_cube_write(std::unique_ptr<IIOTask> task);

    /// @brief 提交二进制写入任务
    bool submit_binary_write(std::unique_ptr<IIOTask> task);

    /// @brief 提交 rhog 二进制读取任务
    bool submit_rhog_read(std::unique_ptr<IIOTask> task);

    // ======================== 读取结果获取 ========================

    IOBuffer pop_completed();
    IOBuffer wait_next_completed();

    // ======================== 性能统计 ========================

    size_t total_submitted() const { return submitted_.load(std::memory_order_relaxed); }
    size_t total_completed() const { return completed_.load(std::memory_order_relaxed); }
    size_t in_flight()       const { return in_flight_.load(std::memory_order_relaxed); }

  private:
    OMPTaskManager() = default;
    ~OMPTaskManager() = default;

    // ======================== 内部方法 ========================

    /// @brief 提交通用任务 (加锁入队)
    bool submit_task(std::unique_ptr<IIOTask> task);

    /// @brief I/O 递归任务链 (每个 worker 的执行主循环)
    ///
    /// 设计模式: 递归链, 而非 while(true) 循环。
    ///   - 每个链步骤: 取任务 → spawn 执行 task → spawn 下一个自己 → return
    ///   - 这样每个 OMP task 有明确的开始/结束, 隐式 barrier 不会死锁
    ///
    /// 终止条件: done_==true && task_queue_.empty()
    void io_worker_chain();

    // ======================== 成员变量 ========================

    size_t num_workers_{4};

    // ---- 状态控制 (atomic) ----
    std::atomic<bool> running_{false};   ///< 管理器是否已初始化
    std::atomic<bool> done_{false};      ///< 计算线程已完成提交
    std::atomic<bool> stop_{false};      ///< 强制停止 (异常路径)
    std::atomic<size_t> in_flight_{0};   ///< 正在执行的 OMP task 数
    std::atomic<size_t> submitted_{0};   ///< 总提交数
    std::atomic<size_t> completed_{0};   ///< 总完成数

    // ---- 任务队列 ----
    std::queue<std::unique_ptr<IIOTask>> task_queue_;
    mutable std::mutex queue_mutex_;//此处mutable是加锁的规范用法，此数据仅仅为了保证数据的加锁，并没有其他的作用
    std::condition_variable cv_;

    // ---- 完成队列 (读取场景) ----
    std::queue<IOBuffer> completed_queue_;

    // ---- OpenMP depend 变量 (SERIALIZE_ALL 共享) ----
    // 所有 SERIALIZE_ALL task 声明 depend(inout: global_serial_dep_),
    // OpenMP 运行时保证同一时刻只有一个此类 task 执行。
    // INDEPENDENT task 不加 depend, 不受此约束。
    int global_serial_dep_{0};
};

// ============================================================================
// inline 实现
// ============================================================================

inline void OMPTaskManager::init(size_t num_workers)
{
    num_workers_ = (num_workers > 0) ? num_workers : 1;
    running_.store(true, std::memory_order_release);
    done_.store(false);
    stop_.store(false);
    in_flight_.store(0);
    submitted_.store(0);
    completed_.store(0);

    // 清空残留队列 (防止上次未正常清理)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!task_queue_.empty()) task_queue_.pop();
        while (!completed_queue_.empty()) completed_queue_.pop();
    }

#ifdef _OPENMP
    std::cout << "OMPTaskManager: initialized with " << num_workers_
              << " I/O workers (OpenMP " << _OPENMP << ")" << std::endl;
#else
    std::cout << "OMPTaskManager: initialized with " << num_workers_
              << " I/O workers (OpenMP NOT available)" << std::endl;
#endif
}

inline void OMPTaskManager::spawn_io_workers()
{
    if (!running_.load(std::memory_order_acquire))
    {
        std::cerr << "OMPTaskManager: cannot spawn, not initialized" << std::endl;
        return;
    }

    for (size_t i = 0; i < num_workers_; ++i)
    {
        #pragma omp task untied
        {
            io_worker_chain();
        }
    }
}

inline void OMPTaskManager::signal_done()
{
    done_.store(true, std::memory_order_release);
    cv_.notify_all();
}

inline void OMPTaskManager::signal_stop()
{
    stop_.store(true, std::memory_order_release);
    done_.store(true, std::memory_order_release);
    cv_.notify_all();
}

inline void OMPTaskManager::wait_all()
{
    // 自旋 + 短暂让步, 等待所有 in-flight task 完成
    while (in_flight_.load(std::memory_order_acquire) > 0)
    {
        #pragma omp taskyield
    }
    // 确保队列也空
    while (true)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (task_queue_.empty()) break;
    }
}

// ======================== 任务提交 ========================

inline bool OMPTaskManager::submit_cube_write(std::unique_ptr<IIOTask> task)
{
    return submit_task(std::move(task));
}

inline bool OMPTaskManager::submit_binary_write(std::unique_ptr<IIOTask> task)
{
    return submit_task(std::move(task));
}

inline bool OMPTaskManager::submit_rhog_read(std::unique_ptr<IIOTask> task)
{
    return submit_task(std::move(task));
}

inline bool OMPTaskManager::submit_task(std::unique_ptr<IIOTask> task)
{
    if (!task)
    {
        return false;
    }

    if (!running_.load(std::memory_order_acquire))
    {
        std::cerr << "OMPTaskManager: cannot submit, not initialized" << std::endl;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
        submitted_.fetch_add(1, std::memory_order_relaxed);
    }

    cv_.notify_one();
    return true;
}

// ======================== 读取结果获取 ========================

inline IOBuffer OMPTaskManager::pop_completed()
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (completed_queue_.empty())
    {
        return IOBuffer();
    }
    IOBuffer result = std::move(completed_queue_.front());
    completed_queue_.pop();
    return result;
}

inline IOBuffer OMPTaskManager::wait_next_completed()
{
    // 自旋等待 + taskyield (在 OpenMP 区域外调用时退化为普通自旋)
    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!completed_queue_.empty())
            {
                IOBuffer result = std::move(completed_queue_.front());
                completed_queue_.pop();
                return result;
            }
        }

        if (done_.load(std::memory_order_acquire) &&
            completed_queue_.empty())
        {
            return IOBuffer();  // 已停止, 无结果
        }

        #pragma omp taskyield
    }
}

// ======================== I/O 递归任务链 ========================

inline void OMPTaskManager::io_worker_chain()
{
    std::unique_ptr<IIOTask> task;

    // ---- Phase 1: 竞争出队 (加锁) ----
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        cv_.wait(lock, [this]() {
            return !task_queue_.empty() || done_.load(std::memory_order_acquire)
                                        || stop_.load(std::memory_order_acquire);
        });

        // 链终止条件
        if (stop_.load(std::memory_order_acquire))
        {
            return;
        }

        if (task_queue_.empty())
        {
            if (done_.load(std::memory_order_acquire))
            {
                // done_ 为 true 且队列空 → 正常退出
                return;
            }
            // 虚假唤醒: 生成下一个自己继续等待
            #pragma omp task untied
            {
                io_worker_chain();
            }
            return;
        }

        task = std::move(task_queue_.front());
        task_queue_.pop();
        in_flight_.fetch_add(1, std::memory_order_relaxed);
    }

    // ---- Phase 2: 通过 OMP task 执行 I/O (锁外) ----
    //
    // Task 所有权 → shared_ptr (must outlive the OMP task).
    // Member 变量在 depend 子句中需通过 local reference 访问
    // (OpenMP 要求 depend(var) 中的 var 是当前作用域内的简单变量名)。
    {
        std::shared_ptr<IIOTask> shared_task(std::move(task));
        OMPTaskManager* mgr = this;
        int& serial_dep = global_serial_dep_;  // local ref, usable in depend
        (void)serial_dep;  // used only by OpenMP depend clause

        TaskAffinity aff = shared_task->affinity();

        if (aff == TaskAffinity::SERIALIZE_ALL)
        {
            // depend(inout: serial_dep) 保证
            // 所有 SERIALIZE_ALL task 串行化
            #pragma omp task depend(inout: serial_dep) untied \
                shared(shared_task, mgr)
            {
                shared_task->execute();

                mgr->completed_.fetch_add(1, std::memory_order_relaxed);
                mgr->in_flight_.fetch_sub(1, std::memory_order_relaxed);

                {
                    std::lock_guard<std::mutex> lock(mgr->queue_mutex_);
                    mgr->completed_queue_.push(std::move(shared_task->buffer()));
                }
            }
        }
        else // INDEPENDENT
        {
            // 无 depend → 多个 worker 可并行执行
            #pragma omp task untied shared(shared_task, mgr)
            {
                shared_task->execute();

                mgr->completed_.fetch_add(1, std::memory_order_relaxed);
                mgr->in_flight_.fetch_sub(1, std::memory_order_relaxed);

                {
                    std::lock_guard<std::mutex> lock(mgr->queue_mutex_);
                    mgr->completed_queue_.push(std::move(shared_task->buffer()));
                }
            }
        }
    } // shared_task 在此处析构 (OMP task 已捕获自己的引用)

    // ---- Phase 3: 生成链的下一个环节 ----
    #pragma omp task untied
    {
        io_worker_chain();
    }
}

#endif // OMP_TASK_MANAGER_H
