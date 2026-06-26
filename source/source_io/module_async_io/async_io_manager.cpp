#include "async_io_manager.h"

#include <iostream>
#include <cassert>

// ============================================================================
// AsyncIOManager 实现
//
// 文件布局:
//   1. 生命周期管理 (start / stop / 析构)
//   2. 同步控制 (wait_all / is_idle / pending_count)
//   3. 任务提交 (submit_cube_write / submit_binary_write / submit_task)
//   4. 工作线程主循环 (io_loop)
//   5. 性能统计
// ============================================================================

// ==================================================================
// 1. 生命周期管理
// ==================================================================

void AsyncIOManager::start(size_t max_queue_size, size_t num_workers)
{
    // 防止重复启动
    if (running_.exchange(true))
    {
        std::cerr << "Warning: AsyncIOManager is already running." << std::endl;
        return;
    }

    max_queue_size_ = max_queue_size;
    num_workers_ = (num_workers > 0) ? num_workers : 1;

    // 创建 N 个工作线程，每个调用 io_loop(worker_id)
    io_workers_.reserve(num_workers_);
    for (size_t i = 0; i < num_workers_; ++i)
    {
        io_workers_.emplace_back(&AsyncIOManager::io_loop, this, i);
    }

    std::cout << "AsyncIOManager: " << num_workers_
              << " I/O worker thread(s) started (max_queue="
              << max_queue_size_ << ")" << std::endl;
}

void AsyncIOManager::wait_all()
{
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // 等待条件: 队列为空 且 没有正在执行的任务
    // task_queue_.empty() 确保队列中无待处理任务
    // in_flight_tasks_ == 0  确保工作线程已完成正在执行的任务
    cv_.wait(lock, [this]() {
        return (task_queue_.empty() && in_flight_tasks_.load() == 0) || !running_.load();
    });
}

size_t AsyncIOManager::pending_count() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.size();
}

bool AsyncIOManager::is_idle() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return task_queue_.empty();
}

bool AsyncIOManager::can_submit() const
{
    if (!running_.load())
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return (max_queue_size_ == 0) || (task_queue_.size() < max_queue_size_);
}

void AsyncIOManager::stop()
{
    if (!running_.load())
    {
        return;
    }

    // 1. 设置停止标志
    running_.store(false);

    // 2. 唤醒所有工作线程和独占等待者
    cv_.notify_all();
    cv_exclusive_.notify_all();

    // 3. 等待所有工作线程结束
    for (size_t i = 0; i < io_workers_.size(); ++i)
    {
        if (io_workers_[i].joinable())
        {
            io_workers_[i].join();
        }
    }
    io_workers_.clear();

    std::cout << "AsyncIOManager: " << num_workers_
              << " I/O worker thread(s) stopped."
              << " (submitted=" << stats_total_submitted_.load()
              << ", completed=" << stats_total_completed_.load()
              << ", rejected=" << stats_total_rejected_.load() << ")"
              << std::endl;
}

AsyncIOManager::~AsyncIOManager()
{
    // 如果用户忘记调用 stop(), 在析构时自动清理
    if (running_.load())
    {
        std::cerr << "AsyncIOManager: auto-stopping worker thread in destructor."
                  << " Pending tasks may be lost." << std::endl;
        stop();
    }
}

// ==================================================================
// 2. 任务提交
// ==================================================================

bool AsyncIOManager::submit_cube_write(IOBuffer&& buf)
{
    auto task = std::unique_ptr<IIOTask>(
        new CubeWriteTask(std::move(buf)));
    return submit_task(std::move(task));
}

bool AsyncIOManager::submit_binary_write(IOBuffer&& buf)
{
    auto task = std::unique_ptr<IIOTask>(
        new BinaryWriteTask(std::move(buf)));
    return submit_task(std::move(task));
}

bool AsyncIOManager::submit_rhog_read(IOBuffer&& buf)
{
    auto task = std::unique_ptr<IIOTask>(
        new RhogReadTask(std::move(buf)));
    return submit_task(std::move(task));
}

// ==================================================================
// 3. 读取结果获取
// ==================================================================

IOBuffer AsyncIOManager::pop_completed()
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (completed_queue_.empty())
    {
        return IOBuffer(); // 空缓冲区
    }
    IOBuffer result = std::move(completed_queue_.front());
    completed_queue_.pop();
    return result;
}

IOBuffer AsyncIOManager::wait_next_completed()
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_.wait(lock, [this]() {
        return !completed_queue_.empty() || !running_.load();
    });
    if (completed_queue_.empty())
    {
        return IOBuffer(); // 管理器已停止, 无结果
    }
    IOBuffer result = std::move(completed_queue_.front());
    completed_queue_.pop();
    return result;
}

size_t AsyncIOManager::completed_count() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return completed_queue_.size();
}

bool AsyncIOManager::submit_task(std::unique_ptr<IIOTask> task)
{
    if (!running_.load())
    {
        std::cerr << "AsyncIOManager: cannot submit task, manager not running."
                  << std::endl;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        // 检查队列是否已满
        if (max_queue_size_ > 0 && task_queue_.size() >= max_queue_size_)
        {
            // 队列满: 拒绝新任务
            stats_total_rejected_.fetch_add(1);
            return false;
        }

        // 将任务加入队列
        task_queue_.push(std::move(task));
        stats_total_submitted_.fetch_add(1);
    }

    // 通知所有工作线程有新任务到达 (多 worker 时需唤醒全部)
    cv_.notify_all();

    return true;
}

// ==================================================================
// 4. 工作线程主循环 (多 worker 竞争出队 + 亲和性检查)
// ==================================================================

void AsyncIOManager::io_loop(size_t worker_id)
{
    while (true)
    {
        std::unique_ptr<IIOTask> current_task;
        bool is_exclusive = false;

        // ---- Phase 1: 竞争出队（加锁） ----
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // 记录当前独占版本号，用于防止 ABA
            size_t my_epoch = exclusive_epoch_.load();

            // 等待条件: 队列非空 或 停止 或 独占模式下所有 INDEPENDENT 完成
            cv_.wait(lock, [this, my_epoch]() {
                return !task_queue_.empty() || !running_.load()
                    || (exclusive_pending_.load()
                        && in_flight_tasks_.load() == 0
                        && exclusive_epoch_.load() != my_epoch);
            });

            // 停止 + 队列空 → 退出
            if (!running_.load() && task_queue_.empty())
            {
                break;
            }

            // 队列空 (被 cv_exclusive_ 唤醒或虚假唤醒) → 继续等待
            if (task_queue_.empty())
            {
                continue;
            }

            // 查看队首任务的亲和性
            TaskAffinity aff = task_queue_.front()->affinity();

            if (aff == TaskAffinity::SERIALIZE_ALL)
            {
                // 独占任务: 等待所有进行中任务完成
                if (in_flight_tasks_.load() > 0)
                {
                    exclusive_pending_.store(true);
                    continue;  // 重新进入等待循环
                }

                // in_flight_tasks_ == 0: 可以执行独占任务
                exclusive_pending_.store(true);
                current_task = std::move(task_queue_.front());
                task_queue_.pop();
                is_exclusive = true;
            }
            else // INDEPENDENT
            {
                // 如果有独占任务在排队，INDEPENDENT 任务不能出队
                if (exclusive_pending_.load())
                {
                    continue;
                }

                current_task = std::move(task_queue_.front());
                task_queue_.pop();
            }

            if (current_task)
            {
                in_flight_tasks_.fetch_add(1);
            }
            // 锁在此处释放 (unique_lock 析构)
        }

        // ---- Phase 2: 执行 I/O（锁外） ----
        if (current_task)
        {
            bool success = current_task->execute();

            if (!success)
            {
                const IOBuffer& buf = current_task->buffer();
                std::cerr << "AsyncIOManager[worker " << worker_id << "]: "
                          << current_task->task_name()
                          << " failed for file " << buf.filename()
                          << ": " << buf.error_message() << std::endl;
            }

            // 推入完成队列
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                completed_queue_.push(std::move(current_task->buffer()));
            }

            stats_total_completed_.fetch_add(1);

            size_t remaining = in_flight_tasks_.fetch_sub(1) - 1;

            // 独占任务完成: 解除独占标记, 递增版本号防止 ABA
            if (is_exclusive)
            {
                exclusive_pending_.store(false);
                exclusive_epoch_.fetch_add(1);
            }
            // 最后一个 INDEPENDENT 任务完成: 通知独占等待者
            else if (remaining == 0 && exclusive_pending_.load())
            {
                exclusive_epoch_.fetch_add(1);
                cv_exclusive_.notify_all();
            }
        }

        // 唤醒可能在 wait_all / wait_next_completed / 其他 worker 中等待的线程
        cv_.notify_all();
    }
}

// ==================================================================
// 4. 性能统计
// ==================================================================

void AsyncIOManager::reset_stats()
{
    stats_total_submitted_.store(0);
    stats_total_completed_.store(0);
    stats_total_rejected_.store(0);
}
