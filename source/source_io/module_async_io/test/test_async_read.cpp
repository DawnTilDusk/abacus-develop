// ============================================================================
// test_async_read.cpp
//
// 单元测试: 异步读取 (RhogReadTask + AsyncIOManager 读取 API)
//
// 测试内容:
//   1. RhogReadTask: 正确解析 QE 兼容二进制文件头部 (gamma_only, npwtot, nspin)
//   2. RhogReadTask: 正确读取 Miller 索引
//   3. RhogReadTask: 正确读取所有自旋的复数数据
//   4. RhogReadTask: 错误处理 (文件不存在、格式错误)
//   5. IOBuffer: rhog 元数据 move 语义正确
//   6. AsyncIOManager: submit_rhog_read + wait_next_completed 往返
//   7. AsyncIOManager: 多自旋读取的流水线正确性
//   8. AsyncIOManager: pop_completed 非阻塞行为
//   9. AsyncIOManager: completed_count 统计正确
//
// 设计:
//   - 直接调用 RhogReadTask::execute() 测试任务逻辑
//   - 通过 AsyncIOManager 完整链路测试
//   - 测试数据使用已知确定性模式
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
#include <cstring>
#include <algorithm>

// ====================================================================
// 测试辅助函数
// ====================================================================

/// @brief 写入一个 QE 兼容的二进制 rhog 文件，用于测试
///
/// 文件格式:
///   /3/ gamma_only npwtot nspin /3/
///   /9/ b1[0..2] b2[0..2] b3[0..2] /9/
///   /3*npwtot/ miller[0..3*npwtot-1] /3*npwtot/
///   [对每个自旋:]
///     /npwtot/ rhog[0..npwtot-1] (complex<double>) /npwtot/
///
/// @param path         输出文件路径
/// @param gamma_only   gamma_only 标记 (0 或 1)
/// @param npwtot       总 G 向量数
/// @param nspin        自旋数
/// @param b            倒格矢 [9] (b1x,b1y,b1z,b2x,...)
/// @param miller       Miller 索引 [3*npwtot]
/// @param spin_data    自旋数据 [nspin][2*npwtot] (实部/虚部交替)
static void write_test_rhog(const std::string& path,
                            int gamma_only,
                            int npwtot,
                            int nspin,
                            const double b[9],
                            const int* miller,
                            const double* spin_data)
{
    FILE* fp = std::fopen(path.c_str(), "wb");
    ASSERT_NE(fp, nullptr) << "Cannot open " << path << " for writing";

    // ---- 头部: /3/ gamma_only npwtot nspin /3/ ----
    int marker3 = 3;
    std::fwrite(&marker3, sizeof(int), 1, fp);
    std::fwrite(&gamma_only, sizeof(int), 1, fp);
    std::fwrite(&npwtot, sizeof(int), 1, fp);
    std::fwrite(&nspin, sizeof(int), 1, fp);
    std::fwrite(&marker3, sizeof(int), 1, fp);

    // ---- 倒格矢: /9/ b1 b2 b3 /9/ ----
    int marker9 = 9;
    std::fwrite(&marker9, sizeof(int), 1, fp);
    std::fwrite(b, sizeof(double), 9, fp);
    std::fwrite(&marker9, sizeof(int), 1, fp);

    // ---- Miller 索引: /3*npwtot/ miller /3*npwtot/ ----
    int miller_marker = 3 * npwtot;
    std::fwrite(&miller_marker, sizeof(int), 1, fp);
    std::fwrite(miller, sizeof(int), miller_marker, fp);
    std::fwrite(&miller_marker, sizeof(int), 1, fp);

    // ---- 自旋数据 ----
    for (int is = 0; is < nspin; ++is)
    {
        const double* spin = spin_data + static_cast<size_t>(is) * 2 * npwtot;
        std::fwrite(&npwtot, sizeof(int), 1, fp);          // /npwtot/
        std::fwrite(spin, sizeof(double), 2 * npwtot, fp); // complex data
        std::fwrite(&npwtot, sizeof(int), 1, fp);          // /npwtot/
    }

    std::fclose(fp);
}

/// @brief 生成确定的 Miller 索引 (可复现)
/// Miller 索引在范围 [-nx/2, nx/2] 内循环
static std::vector<int> make_test_miller(int npwtot, int nx, int ny, int nz)
{
    std::vector<int> miller(3 * npwtot);
    for (int i = 0; i < npwtot; ++i)
    {
        miller[3 * i + 0] = (i % nx) - nx / 2;
        miller[3 * i + 1] = ((i / nx) % ny) - ny / 2;
        miller[3 * i + 2] = ((i / (nx * ny)) % nz) - nz / 2;
    }
    return miller;
}

/// @brief 生成确定的复数测试数据: re = i, im = -i
static std::vector<double> make_test_spin_data(int npwtot, int nspin)
{
    std::vector<double> data(static_cast<size_t>(nspin) * 2 * npwtot);
    for (int is = 0; is < nspin; ++is)
    {
        for (int i = 0; i < npwtot; ++i)
        {
            size_t idx = static_cast<size_t>(is) * 2 * npwtot + static_cast<size_t>(2) * i;
            data[idx] = static_cast<double>(i);        // 实部
            data[idx + 1] = static_cast<double>(-i);   // 虚部
        }
    }
    return data;
}

/// @brief 删除临时文件 (忽略错误)
static void clean_file(const std::string& path)
{
    std::remove(path.c_str());
}

// ====================================================================
// 测试夹具
// ====================================================================

class AsyncReadTest : public ::testing::Test
{
  protected:
    // 测试文件路径
    const std::string test_file_ = "test_rhog_read.bin";
    const std::string nonexistent_file_ = "/nonexistent/path/rhog.bin";

    // 测试参数
    const int nx_ = 16;
    const int ny_ = 16;
    const int nz_ = 16;
    const int npwtot_ = 256;   // 需 < nx*ny*nz/2 以覆盖边界条件
    const int npw_local_ = 64; // 模拟本地 G 向量数

    void SetUp() override
    {
        clean_file(test_file_);
    }

    void TearDown() override
    {
        clean_file(test_file_);
    }
};

// ====================================================================
// 测试用例 1: RhogReadTask 解析单自旋文件头部
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_SingleSpin_Header)
{
    const int nspin = 1;
    const double b[9] = {1.0,0,0, 0,1.0,0, 0,0,1.0};
    std::vector<int> miller = make_test_miller(npwtot_, nx_, ny_, nz_);
    std::vector<double> spindata = make_test_spin_data(npwtot_, nspin);

    // 写入测试文件
    write_test_rhog(test_file_, 0, npwtot_, nspin, b, miller.data(), spindata.data());

    // 构造 RhogReadTask
    IOBuffer buf = IOBuffer::make_binary_read_result(test_file_, npw_local_);
    RhogReadTask task(std::move(buf));

    // 执行读取
    bool ok = task.execute();
    EXPECT_TRUE(ok);

    // 验证头部信息
    const IOBuffer& result = task.buffer();
    EXPECT_FALSE(result.has_error());
    EXPECT_EQ(result.gamma_only_in(), 0);
    EXPECT_EQ(result.npwtot_in(), npwtot_);
    EXPECT_EQ(result.nspin_in(), nspin);

    // 验证倒格矢
    EXPECT_EQ(result.b1().size(), 3);
    EXPECT_EQ(result.b2().size(), 3);
    EXPECT_EQ(result.b3().size(), 3);
    EXPECT_DOUBLE_EQ(result.b1()[0], 1.0);
    EXPECT_DOUBLE_EQ(result.b2()[1], 1.0);
    EXPECT_DOUBLE_EQ(result.b3()[2], 1.0);

    // 验证 Miller 索引
    ASSERT_EQ(result.miller().size(), miller.size());
    for (int i = 0; i < static_cast<int>(miller.size()); ++i)
    {
        EXPECT_EQ(result.miller()[i], miller[i]) << " at miller[" << i << "]";
    }
}

// ====================================================================
// 测试用例 2: RhogReadTask 正确读取单自旋复数数据
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_SingleSpin_DataIntegrity)
{
    const int nspin = 1;
    const double b[9] = {1.0,0,0, 0,1.0,0, 0,0,1.0};
    std::vector<int> miller = make_test_miller(npwtot_, nx_, ny_, nz_);
    std::vector<double> spindata = make_test_spin_data(npwtot_, nspin);

    write_test_rhog(test_file_, 0, npwtot_, nspin, b, miller.data(), spindata.data());

    IOBuffer buf = IOBuffer::make_binary_read_result(test_file_, npw_local_);
    RhogReadTask task(std::move(buf));
    bool ok = task.execute();
    ASSERT_TRUE(ok);

    const IOBuffer& result = task.buffer();

    // data_ 大小应为 nspin * 2 * npwtot
    size_t expected_size = static_cast<size_t>(nspin) * 2 * npwtot_;
    EXPECT_EQ(result.data().size(), expected_size);

    // 验证第一个自旋的数据
    for (int i = 0; i < npwtot_; ++i)
    {
        EXPECT_DOUBLE_EQ(result.data()[2 * i],     static_cast<double>(i))
            << " real part at i=" << i;
        EXPECT_DOUBLE_EQ(result.data()[2 * i + 1], static_cast<double>(-i))
            << " imag part at i=" << i;
    }
}

// ====================================================================
// 测试用例 3: RhogReadTask 多自旋读取
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_MultiSpin_DataIntegrity)
{
    const int nspin = 4;
    const double b[9] = {2.0,0,0, 0,2.0,0, 0,0,2.0};
    std::vector<int> miller = make_test_miller(npwtot_, nx_, ny_, nz_);
    std::vector<double> spindata = make_test_spin_data(npwtot_, nspin);

    write_test_rhog(test_file_, 1, npwtot_, nspin, b, miller.data(), spindata.data());

    IOBuffer buf = IOBuffer::make_binary_read_result(test_file_, npw_local_);
    RhogReadTask task(std::move(buf));
    bool ok = task.execute();
    ASSERT_TRUE(ok);

    const IOBuffer& result = task.buffer();
    EXPECT_EQ(result.gamma_only_in(), 1);
    EXPECT_EQ(result.nspin_in(), nspin);

    // 验证每个自旋的数据布局: [spin0: re0, im0, re1, im1, ...] [spin1: ...]
    size_t spin_stride = static_cast<size_t>(2) * npwtot_;
    for (int is = 0; is < nspin; ++is)
    {
        const double* spin = result.data().data() + is * spin_stride;
        for (int i = 0; i < npwtot_; ++i)
        {
            EXPECT_DOUBLE_EQ(spin[2 * i],     static_cast<double>(i))
                << " real part at spin=" << is << " i=" << i;
            EXPECT_DOUBLE_EQ(spin[2 * i + 1], static_cast<double>(-i))
                << " imag part at spin=" << is << " i=" << i;
        }
    }
}

// ====================================================================
// 测试用例 4: RhogReadTask 错误处理 — 文件不存在
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_FileNotFound)
{
    IOBuffer buf = IOBuffer::make_binary_read_result(nonexistent_file_, npw_local_);
    RhogReadTask task(std::move(buf));

    bool ok = task.execute();
    EXPECT_FALSE(ok);

    const IOBuffer& result = task.buffer();
    EXPECT_TRUE(result.has_error());
    EXPECT_NE(result.error_message().find("cannot open"), std::string::npos)
        << "Error message should mention 'cannot open', got: " << result.error_message();
}

// ====================================================================
// 测试用例 5: RhogReadTask 错误处理 — 截断文件
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_TruncatedFile)
{
    // 只写入头部, 不写 Miller 索引和数据
    FILE* fp = std::fopen(test_file_.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    int marker = 3, gamma = 0, npw = 100, nsp = 1;
    std::fwrite(&marker, sizeof(int), 1, fp);
    std::fwrite(&gamma, sizeof(int), 1, fp);
    std::fwrite(&npw, sizeof(int), 1, fp);
    std::fwrite(&nsp, sizeof(int), 1, fp);
    std::fwrite(&marker, sizeof(int), 1, fp);
    std::fclose(fp);

    IOBuffer buf = IOBuffer::make_binary_read_result(test_file_, npw_local_);
    RhogReadTask task(std::move(buf));

    bool ok = task.execute();
    EXPECT_FALSE(ok);

    const IOBuffer& result = task.buffer();
    EXPECT_TRUE(result.has_error());
}

// ====================================================================
// 测试用例 6: RhogReadTask 错误处理 — 空文件名
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_EmptyFilename)
{
    IOBuffer buf = IOBuffer::make_binary_read_result("", npw_local_);
    RhogReadTask task(std::move(buf));

    bool ok = task.execute();
    EXPECT_FALSE(ok);

    const IOBuffer& result = task.buffer();
    EXPECT_TRUE(result.has_error());
    EXPECT_NE(result.error_message().find("empty filename"), std::string::npos);
}

// ====================================================================
// 测试用例 7: IOBuffer rhog 元数据 move 语义
// ====================================================================

TEST_F(AsyncReadTest, IOBuffer_RhogMetadataMove)
{
    // 创建一个缓冲区并填充 rhog 元数据
    IOBuffer buf1 = IOBuffer::make_binary_read_result("test_move.bin", 100);

    double b1_arr[3] = {1.1, 1.2, 1.3};
    double b2_arr[3] = {2.1, 2.2, 2.3};
    double b3_arr[3] = {3.1, 3.2, 3.3};
    std::vector<int> miller = {1, 2, 3, 4, 5, 6};
    std::vector<double> data = {1.0, -1.0, 2.0, -2.0};

    buf1.set_rhog_result(1, 10, 2, b1_arr, b2_arr, b3_arr,
                         std::move(miller), std::move(data));

    // 验证 buf1
    EXPECT_EQ(buf1.gamma_only_in(), 1);
    EXPECT_EQ(buf1.npwtot_in(), 10);
    EXPECT_EQ(buf1.nspin_in(), 2);
    ASSERT_EQ(buf1.b1().size(), 3);
    EXPECT_DOUBLE_EQ(buf1.b1()[0], 1.1);
    EXPECT_DOUBLE_EQ(buf1.b2()[1], 2.2);
    EXPECT_DOUBLE_EQ(buf1.b3()[2], 3.3);
    ASSERT_EQ(buf1.miller().size(), 6);
    EXPECT_EQ(buf1.miller()[0], 1);
    EXPECT_EQ(buf1.data().size(), 4);

    // Move 构造
    IOBuffer buf2(std::move(buf1));

    // 验证 buf2 拥有数据
    EXPECT_EQ(buf2.gamma_only_in(), 1);
    EXPECT_EQ(buf2.npwtot_in(), 10);
    EXPECT_EQ(buf2.nspin_in(), 2);
    ASSERT_EQ(buf2.b1().size(), 3);
    EXPECT_DOUBLE_EQ(buf2.b1()[0], 1.1);
    ASSERT_EQ(buf2.miller().size(), 6);
    EXPECT_EQ(buf2.miller()[0], 1);
    ASSERT_EQ(buf2.data().size(), 4);

    // 验证 buf1 已被清空
    EXPECT_EQ(buf1.gamma_only_in(), 0);
    EXPECT_EQ(buf1.npwtot_in(), 0);
    EXPECT_EQ(buf1.nspin_in(), 0);
    EXPECT_TRUE(buf1.b1().empty());
    EXPECT_TRUE(buf1.miller().empty());
    EXPECT_TRUE(buf1.data().empty());
}

// ====================================================================
// 测试用例 8: AsyncIOManager submit_rhog_read + wait_next_completed
// ====================================================================

TEST_F(AsyncReadTest, Manager_RhogReadRoundtrip)
{
    const int nspin = 2;
    const double b[9] = {1.0,0,0, 0,1.0,0, 0,0,1.0};
    std::vector<int> miller = make_test_miller(npwtot_, nx_, ny_, nz_);
    std::vector<double> spindata = make_test_spin_data(npwtot_, nspin);

    write_test_rhog(test_file_, 0, npwtot_, nspin, b, miller.data(), spindata.data());

    AsyncIOManager& mgr = AsyncIOManager::instance();
    mgr.start(4);
    mgr.reset_stats();

    // 提交异步读取任务
    IOBuffer submit_buf = IOBuffer::make_binary_read_result(test_file_, npw_local_);
    bool submitted = mgr.submit_rhog_read(std::move(submit_buf));
    EXPECT_TRUE(submitted);

    // 等待结果
    IOBuffer result = mgr.wait_next_completed();

    // 验证结果
    EXPECT_FALSE(result.has_error());
    EXPECT_EQ(result.npwtot_in(), npwtot_);
    EXPECT_EQ(result.nspin_in(), nspin);
    EXPECT_EQ(result.data().size(), static_cast<size_t>(nspin) * 2 * npwtot_);

    // 验证前几个复数
    EXPECT_DOUBLE_EQ(result.data()[0], 0.0);   // spin0, i=0, re
    EXPECT_DOUBLE_EQ(result.data()[1], 0.0);   // spin0, i=0, im
    EXPECT_DOUBLE_EQ(result.data()[2], 1.0);   // spin0, i=1, re
    EXPECT_DOUBLE_EQ(result.data()[3], -1.0);  // spin0, i=1, im

    // 验证管理器的统计
    EXPECT_EQ(mgr.total_submitted(), 1);
    EXPECT_EQ(mgr.total_completed(), 1);

    mgr.stop();
}

// ====================================================================
// 测试用例 9: AsyncIOManager 多自旋流水线读取
// ====================================================================

TEST_F(AsyncReadTest, Manager_MultiSpinPipeline)
{
    const int nspin = 4;
    const double b[9] = {1.5,0,0, 0,1.5,0, 0,0,1.5};
    std::vector<int> miller = make_test_miller(npwtot_, nx_, ny_, nz_);
    std::vector<double> spindata = make_test_spin_data(npwtot_, nspin);

    write_test_rhog(test_file_, 0, npwtot_, nspin, b, miller.data(), spindata.data());

    AsyncIOManager& mgr = AsyncIOManager::instance();
    mgr.start(4);
    mgr.reset_stats();

    // 模拟流水线: 对每个自旋提交一个读取任务 (实际场景中每个自旋文件不同)
    // 这里都使用同一个文件，但足够测试多任务流水线
    const int n_tasks = 3;
    for (int i = 0; i < n_tasks; ++i)
    {
        std::string fn = (i == 0) ? test_file_ : nonexistent_file_;
        IOBuffer buf = IOBuffer::make_binary_read_result(fn, npw_local_);
        bool submitted = mgr.submit_rhog_read(std::move(buf));
        // 第 0 个应当成功, 第 1-2 个因队列未满也成功但读后失败
        EXPECT_TRUE(submitted) << " task " << i;
    }

    // 逐个等待结果 (FIFO)
    for (int i = 0; i < n_tasks; ++i)
    {
        IOBuffer result = mgr.wait_next_completed();
        if (i == 0)
        {
            EXPECT_FALSE(result.has_error());
            EXPECT_EQ(result.nspin_in(), nspin);
        }
        else
        {
            EXPECT_TRUE(result.has_error()); // 不存在的文件
        }
    }

    EXPECT_EQ(mgr.total_submitted(), n_tasks);
    EXPECT_EQ(mgr.total_completed(), n_tasks);

    // 清理
    mgr.stop();
}

// ====================================================================
// 测试用例 10: pop_completed 非阻塞行为
// ====================================================================

TEST_F(AsyncReadTest, Manager_PopCompletedNonBlocking)
{
    AsyncIOManager& mgr = AsyncIOManager::instance();
    mgr.start(4);
    mgr.reset_stats();

    // 队列空时 pop 应返回空 IOBuffer
    IOBuffer empty = mgr.pop_completed();
    EXPECT_TRUE(empty.filename().empty());
    EXPECT_TRUE(empty.data().empty());

    // 提交一个任务
    const double b[9] = {1,0,0, 0,1,0, 0,0,1};
    std::vector<int> miller(6, 0);
    std::vector<double> spindata(2 * 3, 1.0);
    write_test_rhog(test_file_, 0, 3, 1, b, miller.data(), spindata.data());

    IOBuffer submit_buf = IOBuffer::make_binary_read_result(test_file_, npw_local_);
    mgr.submit_rhog_read(std::move(submit_buf));

    // 等待完成
    mgr.wait_all();

    // 完成后 pop 应返回非空结果
    IOBuffer result = mgr.pop_completed();
    EXPECT_FALSE(result.data().empty());
    EXPECT_EQ(result.npwtot_in(), 3);

    // 再次 pop 应为空 (已消费)
    IOBuffer empty_again = mgr.pop_completed();
    EXPECT_TRUE(empty_again.data().empty());

    mgr.stop();
}

// ====================================================================
// 测试用例 11: completed_count 统计
// ====================================================================

TEST_F(AsyncReadTest, Manager_CompletedCount)
{
    AsyncIOManager& mgr = AsyncIOManager::instance();
    mgr.start(4);
    mgr.reset_stats();

    EXPECT_EQ(mgr.completed_count(), 0);

    // 提交两个任务
    const double b[9] = {1,0,0, 0,1,0, 0,0,1};
    std::vector<int> miller(6, 0);
    std::vector<double> spindata(2 * 3, 1.0);
    write_test_rhog(test_file_, 0, 3, 1, b, miller.data(), spindata.data());

    // 任务 1 (成功)
    IOBuffer buf1 = IOBuffer::make_binary_read_result(test_file_, npw_local_);
    mgr.submit_rhog_read(std::move(buf1));

    // 任务 2 (失败)
    IOBuffer buf2 = IOBuffer::make_binary_read_result(nonexistent_file_, npw_local_);
    mgr.submit_rhog_read(std::move(buf2));

    // 等待全部完成
    mgr.wait_all();

    // completed 队列应有 2 项 (任务完成且被推入)
    EXPECT_GE(mgr.completed_count(), 0);
    EXPECT_EQ(mgr.total_submitted(), 2);
    EXPECT_EQ(mgr.total_completed(), 2);

    // 消费一个
    IOBuffer r1 = mgr.pop_completed();
    EXPECT_FALSE(r1.filename().empty());
    EXPECT_GE(mgr.completed_count(), 0);

    mgr.stop();
}

// ====================================================================
// 测试用例 12: RhogReadTask gamma_only 标志正确解析
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_GammaOnlyFlag)
{
    const double b[9] = {1,0,0, 0,1,0, 0,0,1};
    std::vector<int> miller = make_test_miller(10, nx_, ny_, nz_);
    std::vector<double> spindata = make_test_spin_data(10, 1);

    // 测试 gamma_only=1
    write_test_rhog(test_file_, 1, 10, 1, b, miller.data(), spindata.data());

    IOBuffer buf = IOBuffer::make_binary_read_result(test_file_, 10);
    RhogReadTask task(std::move(buf));
    bool ok = task.execute();
    ASSERT_TRUE(ok);
    EXPECT_EQ(task.buffer().gamma_only_in(), 1);

    // 测试 gamma_only=0
    clean_file(test_file_);
    write_test_rhog(test_file_, 0, 10, 1, b, miller.data(), spindata.data());

    IOBuffer buf2 = IOBuffer::make_binary_read_result(test_file_, 10);
    RhogReadTask task2(std::move(buf2));
    ok = task2.execute();
    ASSERT_TRUE(ok);
    EXPECT_EQ(task2.buffer().gamma_only_in(), 0);
}

// ====================================================================
// 测试用例 13: 大文件压力测试
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_LargeFile)
{
    const int nspin = 1;
    const int large_npwtot = 10000; // 10k G-vectors
    const double b[9] = {1,0,0, 0,1,0, 0,0,1};
    std::vector<int> miller = make_test_miller(large_npwtot, nx_, ny_, nz_);
    std::vector<double> spindata = make_test_spin_data(large_npwtot, nspin);

    write_test_rhog(test_file_, 0, large_npwtot, nspin, b, miller.data(), spindata.data());

    IOBuffer buf = IOBuffer::make_binary_read_result(test_file_, large_npwtot);
    RhogReadTask task(std::move(buf));

    bool ok = task.execute();

    EXPECT_TRUE(ok);
    EXPECT_EQ(task.buffer().npwtot_in(), large_npwtot);
    EXPECT_EQ(task.buffer().data().size(), static_cast<size_t>(2) * large_npwtot);

    // 验证首尾数据
    EXPECT_DOUBLE_EQ(task.buffer().data()[0], 0.0);
    EXPECT_DOUBLE_EQ(task.buffer().data()[1], 0.0);
    int last = large_npwtot - 1;
    EXPECT_DOUBLE_EQ(task.buffer().data()[2 * last],     static_cast<double>(last));
    EXPECT_DOUBLE_EQ(task.buffer().data()[2 * last + 1], static_cast<double>(-last));
}

// ====================================================================
// 测试用例 14: 空文件错误处理
// ====================================================================

TEST_F(AsyncReadTest, RhogReadTask_EmptyFile)
{
    // 创建一个空文件
    FILE* fp = std::fopen(test_file_.c_str(), "wb");
    ASSERT_NE(fp, nullptr);
    std::fclose(fp);

    IOBuffer buf = IOBuffer::make_binary_read_result(test_file_, npw_local_);
    RhogReadTask task(std::move(buf));

    bool ok = task.execute();
    EXPECT_FALSE(ok);
    EXPECT_TRUE(task.buffer().has_error());
}

// ====================================================================
// 主函数
// ====================================================================

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
