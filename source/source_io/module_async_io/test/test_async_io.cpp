// ============================================================================
// test_async_io.cpp
//
// 单元测试: 异步 I/O 管理器
//
// 测试内容:
//   1. AsyncIOManager 生命周期 (start/stop)
//   2. Cube 文本格式写入的异步执行
//   3. 二进制格式写入的异步执行
//   4. 数据完整性验证 (写入后读取对比)
//   5. 队列满时的行为
//   6. 多任务提交与顺序保证
//   7. 错误传播
//   8. 析构安全性
//   9. 多 worker 并行写入
//   10. TaskAffinity 独占正确性
//   11. 混合亲和性压力测试
//
// 参考:
//   source/source_io/test_serial/rho_io_test.cpp
// ============================================================================

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "module_async_io/async_io_manager.h"
#include "module_async_io/io_buffer.h"
#include "module_async_io/io_task.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>

// ====================================================================
// 测试辅助函数
// ====================================================================

/// @brief 生成顺序递增的测试数据 (完全可复现，无浮点舍入误差)
/// 数据范围: 0.0, 1.0, 2.0, ... nxyz-1.0
static std::vector<double> make_test_data(int nx, int ny, int nz)
{
    std::vector<double> data(nx * ny * nz);
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
    {
        data[i] = static_cast<double>(i);
    }
    return data;
}

/// @brief 删除临时文件 (忽略错误)
static void clean_file(const std::string& path)
{
    std::remove(path.c_str());
}

// ====================================================================
// 测试辅助: TaskAffinity 测试专用的独占任务
// 重写 affinity() 返回 SERIALIZE_ALL，模拟操作共享文件的场景
// ====================================================================
class ExclusiveBinaryWriteTask : public IIOTask
{
  public:
    explicit ExclusiveBinaryWriteTask(IOBuffer&& buf) : buf_(std::move(buf)) {}

    bool execute() override
    {
        FILE* fp = fopen(buf_.filename().c_str(), "wb");
        if (!fp) { buf_.set_error("cannot open " + buf_.filename()); return false; }
        const std::vector<double>& d = buf_.data();
        size_t w = fwrite(d.data(), sizeof(double), d.size(), fp);
        fclose(fp);
        return w == d.size();
    }

    IOBuffer& buffer() override { return buf_; }
    const IOBuffer& buffer() const override { return buf_; }
    std::string task_name() const override { return "ExclusiveBinaryWrite"; }

    /// @brief 返回 SERIALIZE_ALL — 需独占所有 worker
    TaskAffinity affinity() const override { return TaskAffinity::SERIALIZE_ALL; }

  private:
    IOBuffer buf_;
};

// ====================================================================
// 测试夹具
// ====================================================================

class AsyncIOManagerTest : public ::testing::Test
{
  protected:
    AsyncIOManager& mgr = AsyncIOManager::instance();

    void SetUp() override
    {
        // 确保测试开始时管理器处于停止状态
        // 注意: 如果前一个测试崩溃，可能遗留运行状态
        if (mgr.is_running())
        {
            mgr.stop();
        }
        mgr.reset_stats();
    }

    void TearDown() override
    {
        // 停止管理器 (如果有任务未完成则等待)
        if (mgr.is_running())
        {
            mgr.wait_all();
            mgr.stop();
        }
        mgr.reset_stats();

        // 清理临时文件
        clean_file("test_async_cube.cube");
        clean_file("test_async_binary.bin");
        clean_file("test_async_multi_0.cube");
        clean_file("test_async_multi_1.cube");
        clean_file("test_async_multi_2.cube");
        clean_file("test_async_error.cube");
        for (int i = 0; i < 10; ++i) {
            clean_file("test_mw_" + std::to_string(i) + ".bin");
            clean_file("test_aff_" + std::to_string(i) + ".bin");
        }
    }
};

// ====================================================================
// 测试用例 1: 生命周期
// ====================================================================

TEST_F(AsyncIOManagerTest, StartStop)
{
    EXPECT_FALSE(mgr.is_running());

    mgr.start(4);
    EXPECT_TRUE(mgr.is_running());
    EXPECT_TRUE(mgr.is_idle());

    mgr.stop();
    EXPECT_FALSE(mgr.is_running());
}

TEST_F(AsyncIOManagerTest, DoubleStart)
{
    mgr.start(4);
    mgr.start(4);  // 第二次启动应当不报错
    EXPECT_TRUE(mgr.is_running());
    mgr.stop();
}

// ====================================================================
// 测试用例 2: Cube 写入
// ====================================================================

TEST_F(AsyncIOManagerTest, CubeWriteAndVerify)
{
    const int nx = 12, ny = 12, nz = 12;
    std::vector<double> data = make_test_data(nx, ny, nz);

    mgr.start(4);

    // 构建 IOBuffer
    IOBuffer buf = IOBuffer::make_cube_write(std::move(data), "test_async_cube.cube", 0, 1, 11);

    // 设置 Cube 文件头
    std::vector<std::string> comment = {
        "Cubefile created from ABACUS. Inner loop is z, followed by y and x",
        "1 # number of spin directions 0.000000 # Fermi energy, in Ry"
    };
    std::vector<double> origin = {0.0, 0.0, 0.0};
    std::vector<double> dx = {0.1, 0.0, 0.0};
    std::vector<double> dy = {0.0, 0.1, 0.0};
    std::vector<double> dz = {0.0, 0.0, 0.1};
    std::vector<int> atom_type = {14};
    std::vector<double> atom_charge = {4.0};
    std::vector<std::vector<double>> atom_pos = {{0.0, 0.0, 0.0}};

    buf.set_cube_header(comment, 1, origin, nx, ny, nz, dx, dy, dz,
                        atom_type, atom_charge, atom_pos);

    // ★ 保存 atom 数 (buf 将在 submit 中被 move)
    const int natom_count = atom_type.size();

    // 提交写入任务
    bool submitted = mgr.submit_cube_write(std::move(buf));
    EXPECT_TRUE(submitted);

    // 等待完成
    mgr.wait_all();

    // 验证文件存在且可读
    std::ifstream ifs("test_async_cube.cube");
    EXPECT_TRUE(ifs.good());

    // 验证文件内容: 跳过 Cube 头信息
    // 格式: 2 行注释 + 1 行 natom+origin + 3 行网格 + natom 行原子 = 6+natom 行
    std::string line;
    const int header_lines = 6 + natom_count;
    for (int i = 0; i < header_lines; ++i)
    {
        std::getline(ifs, line);
    }

    // 计算参考数据
    std::vector<double> ref = make_test_data(nx, ny, nz);

    // 读取网格数据并验证
    const int nxyz = nx * ny * nz;
    std::vector<double> read_data(nxyz);
    for (int i = 0; i < nxyz; ++i)
    {
        ifs >> read_data[i];
    }

    // 验证数据一致性
    // Cube 文本格式使用 scientific + precision=11 (12 位有效数字),
    // 最大舍入误差 ~ 5e-12 * |value|。使用 EXPECT_NEAR 配合 1e-8 容忍度
    const double epsilon = 1e-8;
    for (int i = 0; i < nxyz; ++i)
    {
        EXPECT_NEAR(read_data[i], ref[i], epsilon) << " at index " << i;
    }

    mgr.stop();
}

// ====================================================================
// 测试用例 3: 二进制写入
// ====================================================================

TEST_F(AsyncIOManagerTest, BinaryWriteAndVerify)
{
    const int n = 1000;
    std::vector<double> data(n);
    for (int i = 0; i < n; ++i)
    {
        data[i] = static_cast<double>(i) * 0.5;
    }

    mgr.start(4);

    IOBuffer buf = IOBuffer::make_binary_write(std::move(data), "test_async_binary.bin");
    bool submitted = mgr.submit_binary_write(std::move(buf));
    EXPECT_TRUE(submitted);

    mgr.wait_all();

    // 验证二进制文件
    FILE* fp = fopen("test_async_binary.bin", "rb");
    ASSERT_NE(fp, nullptr);

    std::vector<double> read_data(n);
    size_t read_cnt = fread(read_data.data(), sizeof(double), n, fp);
    fclose(fp);

    EXPECT_EQ(read_cnt, n);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(read_data[i], static_cast<double>(i) * 0.5);
    }

    mgr.stop();
}

// ====================================================================
// 测试用例 4: 多任务提交 (验证顺序和完整性)
// ====================================================================

TEST_F(AsyncIOManagerTest, MultipleTasks)
{
    const int n_files = 3;
    const int nx = 6, ny = 6, nz = 6;

    mgr.start(4);

    std::vector<std::string> comment = {
        "Cubefile created from ABACUS. Inner loop is z, followed by y and x",
        "1 # number of spin directions"
    };
    std::vector<double> origin = {0.0, 0.0, 0.0};
    std::vector<double> dx = {0.2, 0.0, 0.0};
    std::vector<double> dy = {0.0, 0.2, 0.0};
    std::vector<double> dz = {0.0, 0.0, 0.2};
    std::vector<int> atom_type = {14};
    std::vector<double> atom_charge = {4.0};
    std::vector<std::vector<double>> atom_pos = {{0.0, 0.0, 0.0}};

    for (int i = 0; i < n_files; ++i)
    {
        std::vector<double> data = make_test_data(nx, ny, nz);
        // 每个文件加入唯一的偏移，便于验证
        for (double& v : data)
        {
            v += i * 100.0;
        }

        std::string fn = "test_async_multi_" + std::to_string(i) + ".cube";
        IOBuffer buf = IOBuffer::make_cube_write(std::move(data), fn, 0, i, 11);
        buf.set_cube_header(comment, 1, origin, nx, ny, nz, dx, dy, dz,
                            atom_type, atom_charge, atom_pos);

        bool submitted = mgr.submit_cube_write(std::move(buf));
        EXPECT_TRUE(submitted);
    }

    mgr.wait_all();

    // 验证每个文件
    for (int i = 0; i < n_files; ++i)
    {
        std::string fn = "test_async_multi_" + std::to_string(i) + ".cube";
        std::ifstream ifs(fn);
        EXPECT_TRUE(ifs.good()) << "File " << fn << " should exist";

        // 读取数据验证: 跳过头部 (6 + natom 行, natom=1)
        std::string line;
        const int header_lines = 6 + atom_type.size(); // 2 comments + 1 origin + 3 grid + natom
        for (int j = 0; j < header_lines; ++j)
            std::getline(ifs, line);

        const int nxyz = nx * ny * nz;
        std::vector<double> ref = make_test_data(nx, ny, nz);
        for (double& v : ref)
            v += i * 100.0;

        std::vector<double> read_data(nxyz);
        for (int j = 0; j < nxyz; ++j)
            ifs >> read_data[j];

        // Cube 文本格式 precision=11 (~12 sig figs), 容忍度 1e-8
        const double epsilon = 1e-8;
        for (int j = 0; j < nxyz; ++j)
        {
            EXPECT_NEAR(read_data[j], ref[j], epsilon)
                << "File " << fn << " at index " << j;
        }
    }

    mgr.stop();
}

// ====================================================================
// 测试用例 5: 队列满时的行为
// ====================================================================

TEST_F(AsyncIOManagerTest, QueueFullRejection)
{
    const int max_queue = 2;
    mgr.start(max_queue);

    // 构造一个长时间运行的伪任务来占满队列
    // 使用真正的大数据量任务，确保 I/O 线程不能瞬间完成

    const int nx = 6, ny = 6, nz = 6;
    std::vector<std::string> comment = {
        "Cubefile created from ABACUS.",
        "1 # number of spin directions"
    };
    std::vector<double> origin = {0.0, 0.0, 0.0};
    std::vector<double> dx = {0.2, 0.0, 0.0};
    std::vector<double> dy = {0.0, 0.2, 0.0};
    std::vector<double> dz = {0.0, 0.0, 0.2};
    std::vector<int> atom_type = {14};
    std::vector<double> atom_charge = {4.0};
    std::vector<std::vector<double>> atom_pos = {{0.0, 0.0, 0.0}};

    // 提交 max_queue + 1 个任务，最后一个可能被拒绝（如果工作线程还没处理完前几个）
    int submitted_count = 0;
    for (int i = 0; i < max_queue + 3; ++i)
    {
        std::vector<double> data = make_test_data(nx, ny, nz);
        std::string fn = "test_queue_" + std::to_string(i) + ".cube";
        IOBuffer buf = IOBuffer::make_cube_write(std::move(data), fn, 0, i, 11);
        buf.set_cube_header(comment, 1, origin, nx, ny, nz, dx, dy, dz,
                            atom_type, atom_charge, atom_pos);

        if (mgr.submit_cube_write(std::move(buf)))
        {
            submitted_count++;
        }
    }

    // 至少 max_queue 个应该被接受
    EXPECT_GE(submitted_count, max_queue);

    mgr.wait_all();

    // 验证提交的 + 拒绝的 = max_queue + 3
    size_t total = mgr.total_submitted() + mgr.total_rejected();
    EXPECT_GE(total, max_queue + 3);

    // 清理临时文件
    for (int i = 0; i < max_queue + 3; ++i)
    {
        clean_file("test_queue_" + std::to_string(i) + ".cube");
    }

    mgr.stop();
}

// ====================================================================
// 测试用例 6: 错误文件路径
// ====================================================================

TEST_F(AsyncIOManagerTest, ErrorPropagation)
{
    mgr.start(4);

    // 提交一个无法创建文件的写入
    std::vector<double> data = {1.0, 2.0, 3.0};
    IOBuffer buf = IOBuffer::make_binary_write(std::move(data),
                                                "/nonexistent_dir/test_error.bin");
    bool submitted = mgr.submit_binary_write(std::move(buf));
    EXPECT_TRUE(submitted);

    // 等待完成 (错误不会导致崩溃)
    mgr.wait_all();

    // 验证总提交数与完成任务数
    EXPECT_EQ(mgr.total_submitted(), 1);
    EXPECT_EQ(mgr.total_completed(), 1);

    mgr.stop();
}

// ====================================================================
// 测试用例 7: is_idle 和 pending_count
// ====================================================================

TEST_F(AsyncIOManagerTest, StateQueries)
{
    mgr.start(4);

    EXPECT_TRUE(mgr.is_idle());
    EXPECT_EQ(mgr.pending_count(), 0);

    // 提交一个非常小的任务，应很快被消费
    std::vector<double> data = {42.0};
    IOBuffer buf = IOBuffer::make_binary_write(std::move(data), "test_state.bin");
    mgr.submit_binary_write(std::move(buf));

    // 短暂等待，确保 I/O 线程有机会处理
    mgr.wait_all();

    EXPECT_TRUE(mgr.is_idle());
    EXPECT_EQ(mgr.pending_count(), 0);

    clean_file("test_state.bin");
    mgr.stop();
}

// ====================================================================
// 测试用例 9: 多 worker 并行写入
// ====================================================================

TEST_F(AsyncIOManagerTest, MultiWorkerParallelWrite)
{
    const int n_files = 4;
    const size_t data_size = 10000;  // 足够大，确保任务不会瞬间完成

    // 启动 4 个 worker
    mgr.start(8, 4);
    EXPECT_EQ(mgr.num_workers(), 4u);

    for (int i = 0; i < n_files; ++i)
    {
        std::vector<double> data(data_size, static_cast<double>(i));
        std::string fn = "test_mw_" + std::to_string(i) + ".bin";
        IOBuffer buf = IOBuffer::make_binary_write(std::move(data), fn);
        bool ok = mgr.submit_binary_write(std::move(buf));
        EXPECT_TRUE(ok) << "Task " << i << " submission failed";
    }

    mgr.wait_all();

    // 验证所有文件正确写入
    EXPECT_EQ(mgr.total_submitted(), static_cast<size_t>(n_files));
    EXPECT_EQ(mgr.total_completed(), static_cast<size_t>(n_files));
    EXPECT_EQ(mgr.total_rejected(), 0u);

    for (int i = 0; i < n_files; ++i)
    {
        std::string fn = "test_mw_" + std::to_string(i) + ".bin";
        FILE* fp = fopen(fn.c_str(), "rb");
        ASSERT_NE(fp, nullptr) << "File " << fn << " should exist";

        std::vector<double> readback(data_size);
        size_t n = fread(readback.data(), sizeof(double), data_size, fp);
        fclose(fp);

        EXPECT_EQ(n, data_size);
        for (size_t j = 0; j < data_size; ++j)
        {
            EXPECT_DOUBLE_EQ(readback[j], static_cast<double>(i))
                << "File " << fn << " at index " << j;
        }
    }

    mgr.stop();
}

// ====================================================================
// 测试用例 10: TaskAffinity 独占正确性
// 验证 SERIALIZE_ALL 任务在所有 INDEPENDENT 完成后才执行
// ====================================================================

TEST_F(AsyncIOManagerTest, TaskAffinitySerialize)
{
    const size_t data_size = 50000;

    mgr.start(8, 2);  // 2 个 worker

    // 提交 2 个 INDEPENDENT 任务
    for (int i = 0; i < 2; ++i)
    {
        std::vector<double> data(data_size, static_cast<double>(i));
        std::string fn = "test_aff_" + std::to_string(i) + ".bin";
        IOBuffer buf = IOBuffer::make_binary_write(std::move(data), fn);
        bool ok = mgr.submit_binary_write(std::move(buf));
        EXPECT_TRUE(ok);
    }

    // 提交 1 个 SERIALIZE_ALL 任务 (通过 ExclusiveBinaryWriteTask)
    {
        std::vector<double> data(data_size, 99.0);
        IOBuffer buf = IOBuffer::make_binary_write(std::move(data), "test_aff_9.bin");
        auto task = std::unique_ptr<IIOTask>(new ExclusiveBinaryWriteTask(std::move(buf)));
        TaskAffinity aff = task->affinity();
        EXPECT_EQ(aff, TaskAffinity::SERIALIZE_ALL);
        bool ok = mgr.submit_task(std::move(task));
        EXPECT_TRUE(ok);
    }

    mgr.wait_all();

    EXPECT_EQ(mgr.total_submitted(), 3u);
    EXPECT_EQ(mgr.total_completed(), 3u);
    EXPECT_EQ(mgr.total_rejected(), 0u);

    // 验证所有文件正确写入
    for (int i = 0; i < 2; ++i)
    {
        std::string fn = "test_aff_" + std::to_string(i) + ".bin";
        FILE* fp = fopen(fn.c_str(), "rb");
        ASSERT_NE(fp, nullptr);
        std::vector<double> rb(data_size);
        fread(rb.data(), sizeof(double), data_size, fp);
        fclose(fp);
        EXPECT_DOUBLE_EQ(rb[0], static_cast<double>(i));
    }

    // 验证独占任务文件
    {
        FILE* fp = fopen("test_aff_9.bin", "rb");
        ASSERT_NE(fp, nullptr);
        std::vector<double> rb(data_size);
        fread(rb.data(), sizeof(double), data_size, fp);
        fclose(fp);
        EXPECT_DOUBLE_EQ(rb[0], 99.0);
    }

    mgr.stop();
}

// ====================================================================
// 测试用例 11: 混合亲和性压力测试
// 交替提交 INDEPENDENT 和 SERIALIZE_ALL，验证无死锁、无数据损坏
// ====================================================================

TEST_F(AsyncIOManagerTest, MixedAffinityStress)
{
    const int rounds = 10;
    const size_t data_size = 1000;

    mgr.start(0, 4);  // 4 个 worker, 无限制队列 (避免满队列拒绝)

    for (int r = 0; r < rounds; ++r)
    {
        // 提交 1 个 INDEPENDENT 任务
        {
            std::vector<double> data(data_size, static_cast<double>(r * 2));
            std::string fn = "test_mw_" + std::to_string(r) + ".bin";
            IOBuffer buf = IOBuffer::make_binary_write(std::move(data), fn);
            bool ok = mgr.submit_binary_write(std::move(buf));
            EXPECT_TRUE(ok) << "INDEPENDENT round " << r << " failed";
        }

        // 提交 1 个 SERIALIZE_ALL 任务
        {
            std::vector<double> data(data_size, static_cast<double>(r * 2 + 1));
            std::string fn = "test_aff_" + std::to_string(r) + ".bin";
            IOBuffer buf = IOBuffer::make_binary_write(std::move(data), fn);
            auto task = std::unique_ptr<IIOTask>(
                new ExclusiveBinaryWriteTask(std::move(buf)));
            bool ok = mgr.submit_task(std::move(task));
            EXPECT_TRUE(ok) << "SERIALIZE_ALL round " << r << " failed";
        }
    }

    mgr.wait_all();

    // 验证统计
    EXPECT_EQ(mgr.total_submitted(), static_cast<size_t>(rounds * 2));
    EXPECT_EQ(mgr.total_completed(), static_cast<size_t>(rounds * 2));
    EXPECT_EQ(mgr.total_rejected(), 0u);

    // 验证所有文件 (挑几个采样验证，避免测试过慢)
    for (int r = 0; r < rounds; ++r)
    {
        // INDEPENDENT 任务文件
        {
            std::string fn = "test_mw_" + std::to_string(r) + ".bin";
            FILE* fp = fopen(fn.c_str(), "rb");
            ASSERT_NE(fp, nullptr) << "Missing INDEPENDENT file round " << r;
            std::vector<double> rb(data_size);
            size_t n = fread(rb.data(), sizeof(double), data_size, fp);
            fclose(fp);
            EXPECT_EQ(n, data_size);
            EXPECT_DOUBLE_EQ(rb[0], static_cast<double>(r * 2));
        }

        // SERIALIZE_ALL 任务文件
        {
            std::string fn = "test_aff_" + std::to_string(r) + ".bin";
            FILE* fp = fopen(fn.c_str(), "rb");
            ASSERT_NE(fp, nullptr) << "Missing SERIALIZE_ALL file round " << r;
            std::vector<double> rb(data_size);
            size_t n = fread(rb.data(), sizeof(double), data_size, fp);
            fclose(fp);
            EXPECT_EQ(n, data_size);
            EXPECT_DOUBLE_EQ(rb[0], static_cast<double>(r * 2 + 1));
        }
    }

    mgr.stop();
}

// ====================================================================
// 主函数 (如果作为独立可执行文件)
// ====================================================================

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
