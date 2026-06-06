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

void AsyncIOManager::start(size_t max_queue_size)
{
    // 防止重复启动
    if (running_.exchange(true))
    {
        std::cerr << "Warning: AsyncIOManager is already running." << std::endl;
        return;
    }

    max_queue_size_ = max_queue_size;

    // 创建工作线程: 调用 io_loop() 进入主循环
    io_worker_ = std::thread(&AsyncIOManager::io_loop, this);

    std::cout << "AsyncIOManager: I/O worker thread started (max_queue="
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

    // 2. 唤醒工作线程 (让其检查 running_ 并退出)
    cv_.notify_all();

    // 3. 等待工作线程结束
    if (io_worker_.joinable())
    {
        io_worker_.join();
    }

    std::cout << "AsyncIOManager: I/O worker thread stopped."
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

    // 通知工作线程有新任务到达
    cv_.notify_one();

    return true;
}

// ==================================================================
// 3. 工作线程主循环
// ==================================================================

void AsyncIOManager::io_loop()
{
    while (true)
    {
        std::unique_ptr<IIOTask> current_task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // 等待任务到达或停止信号
            cv_.wait(lock, [this]() {
                return !task_queue_.empty() || !running_.load();
            });

            // 如果被唤醒但已停止且队列为空，退出循环
            if (!running_.load() && task_queue_.empty())
            {
                break;
            }

            // 如果队列为空但仍在运行，继续等待
            if (task_queue_.empty())
            {
                continue;
            }

            // 取出一个任务
            current_task = std::move(task_queue_.front());
            task_queue_.pop();
            in_flight_tasks_.fetch_add(1);
        }

        // ---- 在锁外执行阻塞 I/O ----
        // 这样做的好处:
        //   1. 主线程可以在 I/O 执行期间继续提交新任务
        //   2. I/O 操作不会阻塞队列访问

        if (current_task)
        {
            bool success = current_task->execute();

            if (!success)
            {
                // 记录错误日志
                const IOBuffer& buf = current_task->buffer();
                std::cerr << "AsyncIOManager: " << current_task->task_name()
                          << " failed for file " << buf.filename()
                          << ": " << buf.error_message() << std::endl;
            }

            // 将已完成任务的缓冲区推入完成队列 (主线程通过 pop_completed 消费)
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                completed_queue_.push(std::move(current_task->buffer()));
            }

            // 更新统计
            stats_total_completed_.fetch_add(1);

            // 标记任务已完成 (通知可能在 wait_all 中的主线程)
            in_flight_tasks_.fetch_sub(1);
        }

        // 唤醒可能正在 wait_all 或 wait_next_completed 中的主线程
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
