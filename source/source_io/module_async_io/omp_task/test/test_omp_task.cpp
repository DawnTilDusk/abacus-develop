// test_omp_task.cpp
// Unit tests: OMPTaskManager synchronous path
#include "gtest/gtest.h"
#include "omp_task_manager.h"
#include "io_task.h"
#include "io_buffer.h"
#include <vector>
#include <string>
#include <cstdio>

static void clean_file(const std::string& path) { std::remove(path.c_str()); }

static std::vector<double> make_test_data(int n) {
    std::vector<double> d(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) d[static_cast<size_t>(i)] = static_cast<double>(i);
    return d;
}

class OMPTaskManagerTest : public ::testing::Test {
protected:
    OMPTaskManager& mgr = OMPTaskManager::instance();
    void SetUp() override   { mgr.reset_for_testing(); }
    void TearDown() override{ mgr.reset_for_testing(); }
};

TEST_F(OMPTaskManagerTest, InitAndReset) {
    EXPECT_FALSE(mgr.is_running());
    mgr.init(4);
    EXPECT_TRUE(mgr.is_running());
    EXPECT_EQ(mgr.total_submitted(), 0u);
    mgr.reset_for_testing();
    EXPECT_FALSE(mgr.is_running());
}

TEST_F(OMPTaskManagerTest, SubmitBinaryWrite) {
    mgr.init(2);
    std::vector<double> data = make_test_data(200);
    IOBuffer buf = IOBuffer::make_binary_write(std::move(data), "test_omp_bin.bin");
    auto task = std::unique_ptr<IIOTask>(new BinaryWriteTask(std::move(buf)));
    EXPECT_TRUE(mgr.submit_binary_write(std::move(task)));
    EXPECT_EQ(mgr.total_submitted(), 1u);
    EXPECT_EQ(mgr.total_completed(), 0u);
    mgr.signal_done();
    clean_file("test_omp_bin.bin");
}

TEST_F(OMPTaskManagerTest, RejectWithoutInit) {
    EXPECT_FALSE(mgr.is_running());
    std::vector<double> data(64, 1.0);
    IOBuffer buf = IOBuffer::make_binary_write(std::move(data), "test_noinit.bin");
    auto task = std::unique_ptr<IIOTask>(new BinaryWriteTask(std::move(buf)));
    EXPECT_FALSE(mgr.submit_binary_write(std::move(task)));
}

TEST_F(OMPTaskManagerTest, PopCompletedEmpty) {
    mgr.init(2);
    IOBuffer r = mgr.pop_completed();
    EXPECT_TRUE(r.data().empty());
}

TEST_F(OMPTaskManagerTest, WaitNextCompletedAfterDone) {
    mgr.init(2);
    mgr.signal_done();
    IOBuffer r = mgr.wait_next_completed();
    EXPECT_TRUE(r.data().empty());
}

TEST_F(OMPTaskManagerTest, SignalStop) {
    mgr.init(2);
    mgr.signal_stop();
    SUCCEED();
}

TEST_F(OMPTaskManagerTest, BinaryTaskExecution) {
    const int n = 200;
    std::vector<double> ref = make_test_data(n);
    IOBuffer buf = IOBuffer::make_binary_write(std::move(ref), "test_exec.bin");
    BinaryWriteTask task(std::move(buf));
    EXPECT_TRUE(task.execute());
    FILE* fp = std::fopen("test_exec.bin", "rb");
    ASSERT_NE(fp, nullptr);
    std::vector<double> rb(static_cast<size_t>(n));
    fread(rb.data(), sizeof(double), static_cast<size_t>(n), fp);
    fclose(fp);
    EXPECT_DOUBLE_EQ(rb[0], 0.0);
    EXPECT_DOUBLE_EQ(rb[static_cast<size_t>(n)-1], static_cast<double>(n-1));
    clean_file("test_exec.bin");
}

TEST_F(OMPTaskManagerTest, MultiSubmit) {
    mgr.init(2);
    for (int i = 0; i < 5; ++i) {
        std::vector<double> d(64, static_cast<double>(i));
        IOBuffer buf = IOBuffer::make_binary_write(std::move(d),
            "test_multi_" + std::to_string(i) + ".bin");
        auto task = std::unique_ptr<IIOTask>(new BinaryWriteTask(std::move(buf)));
        EXPECT_TRUE(mgr.submit_binary_write(std::move(task)));
    }
    EXPECT_EQ(mgr.total_submitted(), 5u);
    EXPECT_EQ(mgr.total_completed(), 0u);
    for (int i = 0; i < 5; ++i)
        clean_file("test_multi_" + std::to_string(i) + ".bin");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
