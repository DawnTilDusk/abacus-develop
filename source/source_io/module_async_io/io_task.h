#ifndef IO_TASK_H
#define IO_TASK_H

#include "io_buffer.h"
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>

// ============================================================================
// IIOTask: 异步 I/O 任务的抽象基类
//
// 设计模式: 策略模式 + 命令模式
//   - 每种 I/O 操作 (cube写入、二进制写入、cube读取等) 封装为独立的 Task 子类
//   - I/O 工作线程只调用 execute() 接口，无需关心具体操作细节
//
// 职责:
//   1. 持有 IOBuffer (数据 + 元信息)
//   2. 实现 execute() 方法以执行阻塞的 I/O 操作
//   3. 执行结果通过 IOBuffer::error_message 传播给主线程
// ============================================================================

/// @brief 异步 I/O 任务的抽象基类
class IIOTask
{
  public:
    IIOTask() = default;
    virtual ~IIOTask() = default;

    // 禁止拷贝
    IIOTask(const IIOTask&) = delete;
    IIOTask& operator=(const IIOTask&) = delete;

    // 允许 move 构造/赋值
    IIOTask(IIOTask&&) = default;
    IIOTask& operator=(IIOTask&&) = default;

    /// @brief 执行 I/O 操作 (在工作线程中调用)
    /// @return true 表示成功，false 表示失败
    virtual bool execute() = 0;

    /// @brief 获取任务持有的数据缓冲区
    virtual IOBuffer& buffer() = 0;
    virtual const IOBuffer& buffer() const = 0;

    /// @brief 获取任务名称 (用于日志输出)
    virtual std::string task_name() const = 0;
};

// ============================================================================
// CubeWriteTask: 将电荷密度数据写入 Cube 格式文件
//
// 写入逻辑:
//   1. 从 IOBuffer 中读取头信息和网格数据
//   2. 调用标准 write_cube() 函数或等效逻辑写入文件
//   3. 更新 IOBuffer 中的错误状态
// ============================================================================

/// @brief Cube 文本格式写入任务
class CubeWriteTask : public IIOTask
{
  public:
    /// @brief 构造 Cube 写入任务
    /// @param buf 包含所有网格数据和头信息的缓冲区
    explicit CubeWriteTask(IOBuffer&& buf)
        : buf_(std::move(buf))
    {
    }

    bool execute() override
    {
        // 参数校验
        if (buf_.data().empty())
        {
            buf_.set_error("CubeWriteTask: empty data buffer");
            return false;
        }
        if (buf_.filename().empty())
        {
            buf_.set_error("CubeWriteTask: empty filename");
            return false;
        }
        if (buf_.comment().size() < 2)
        {
            buf_.set_error("CubeWriteTask: need at least 2 comment lines");
            return false;
        }

        std::ofstream ofs(buf_.filename());
        if (!ofs)
        {
            buf_.set_error("CubeWriteTask: cannot open file " + buf_.filename());
            return false;
        }

        // ---- 写入头信息 ----
        for (int i = 0; i < 2; ++i)
        {
            ofs << buf_.comment()[i] << "\n";
        }

        ofs << std::fixed;
        ofs << std::setprecision(1);

        // 原子数 + 原点
        ofs << buf_.natom() << " "
            << buf_.origin()[0] << " "
            << buf_.origin()[1] << " "
            << buf_.origin()[2] << " \n";

        // 网格维度和格点向量
        ofs << std::setprecision(6);
        ofs << buf_.nx() << " "
            << buf_.dx()[0] << " " << buf_.dx()[1] << " " << buf_.dx()[2] << "\n";
        ofs << buf_.ny() << " "
            << buf_.dy()[0] << " " << buf_.dy()[1] << " " << buf_.dy()[2] << "\n";
        ofs << buf_.nz() << " "
            << buf_.dz()[0] << " " << buf_.dz()[1] << " " << buf_.dz()[2] << "\n";

        // 原子信息
        for (int i = 0; i < buf_.natom(); ++i)
        {
            ofs << " " << buf_.atom_type()[i]
                << " " << buf_.atom_charge()[i]
                << " " << buf_.atom_pos()[i][0]
                << " " << buf_.atom_pos()[i][1]
                << " " << buf_.atom_pos()[i][2] << "\n";
        }

        // ---- 写入网格数据 ----
        ofs.unsetf(std::ofstream::fixed);
        ofs << std::setprecision(buf_.precision());
        ofs << std::scientific;

        const int nz = buf_.nz();
        const int nxy = buf_.nx() * buf_.ny();
        const int ndata_line = 6; // 每行6个数据，与原有实现一致

        for (int ixy = 0; ixy < nxy; ++ixy)
        {
            for (int iz = 0; iz < nz; ++iz)
            {
                ofs << " " << buf_.data()[ixy * nz + iz];
                if ((iz + 1) % ndata_line == 0 && iz != nz - 1)
                {
                    ofs << "\n";
                }
            }
            ofs << "\n";
        }

        ofs.close();
        return true;
    }

    IOBuffer& buffer() override { return buf_; }
    const IOBuffer& buffer() const override { return buf_; }
    std::string task_name() const override { return "CubeWrite"; }

  private:
    IOBuffer buf_;  ///< 持有数据所有权
};

// ============================================================================
// BinaryWriteTask: 将数据以二进制格式写入文件
//
// 用于写入重启文件 (*-CHARGE-DENSITY.restart)
// 使用 Binstream 风格的二进制写入 (fwrite)
// ============================================================================

/// @brief 二进制格式写入任务
class BinaryWriteTask : public IIOTask
{
  public:
    explicit BinaryWriteTask(IOBuffer&& buf)
        : buf_(std::move(buf))
    {
    }

    bool execute() override
    {
        if (buf_.data().empty())
        {
            buf_.set_error("BinaryWriteTask: empty data buffer");
            return false;
        }

        FILE* fp = fopen(buf_.filename().c_str(), "wb");
        if (!fp)
        {
            buf_.set_error("BinaryWriteTask: cannot open file " + buf_.filename());
            return false;
        }

        size_t written = fwrite(buf_.data().data(), sizeof(double),
                                buf_.data().size(), fp);
        fclose(fp);

        if (written != buf_.data().size())
        {
            buf_.set_error("BinaryWriteTask: short write (" + std::to_string(written)
                           + " vs " + std::to_string(buf_.data().size()) + " elements)");
            return false;
        }

        return true;
    }

    IOBuffer& buffer() override { return buf_; }
    const IOBuffer& buffer() const override { return buf_; }
    std::string task_name() const override { return "BinaryWrite"; }

  private:
    IOBuffer buf_;
};

// ============================================================================
// CubeReadTask: 从 Cube 文件读取数据到 IOBuffer
//
// 注意: 此操作在工作线程中执行文件读取，
//       但 Miller 索引映射和 MPI 通信仍在主线程执行。
// ============================================================================

/// @brief Cube 文件读取任务
class CubeReadTask : public IIOTask
{
  public:
    explicit CubeReadTask(IOBuffer&& buf)
        : buf_(std::move(buf))
    {
    }

    bool execute() override
    {
        std::ifstream ifs(buf_.filename());
        if (!ifs)
        {
            buf_.set_error("CubeReadTask: cannot open file " + buf_.filename());
            return false;
        }

        // 跳过 2 行注释和 6 行头信息 (共 8 行)
        std::string line;
        for (int i = 0; i < 8; ++i)
        {
            std::getline(ifs, line);
        }

        // 读取网格数据
        std::vector<double>& data = buf_.data();
        const size_t nxyz = buf_.nx() * buf_.ny() * buf_.nz();
        data.resize(nxyz);

        for (size_t i = 0; i < nxyz; ++i)
        {
            ifs >> data[i];
        }

        ifs.close();
        return true;
    }

    IOBuffer& buffer() override { return buf_; }
    const IOBuffer& buffer() const override { return buf_; }
    std::string task_name() const override { return "CubeRead"; }

  private:
    IOBuffer buf_;
};

// ============================================================================
// RhogReadTask: 从 QE 兼容二进制重启文件读取电荷密度数据
//
// 文件格式 (与 Quantum ESPRESSO 的 write_rhog 兼容):
//
//   /3/            -- int: 3 (大小标记)
//   gamma_only     -- int: 0 或 1
//   ngm_g          -- int: 总 G 向量数
//   nspin          -- int: 自旋数
//   /3/            -- int: 3 (大小标记)
//   /9/            -- int: 9 (大小标记)
//   b1[0..2]       -- 3 doubles: 倒格矢 1
//   b2[0..2]       -- 3 doubles: 倒格矢 2
//   b3[0..2]       -- 3 doubles: 倒格矢 3
//   /9/            -- int: 9 (大小标记)
//   /3*ngm_g/      -- int: 3*ngm_g (大小标记)
//   miller[0..3*ngm_g-1] -- 3*ngm_g ints: Miller 指数
//   /3*ngm_g/      -- int: 3*ngm_g (大小标记)
//   [对每个自旋:]
//     /ngm_g/      -- int: ngm_g (大小标记)
//     rhog[0..ngm_g-1] -- ngm_g complex<double>: 电荷密度
//     /ngm_g/      -- int: ngm_g (大小标记)
//
// 工作流程:
//   1. I/O 工作线程读取文件的全部原始二进制数据
//   2. 解析头 (gamma_only, npwtot, nspin, b1/b2/b3)
//   3. 读取 Miller 索引
//   4. 将所有自旋的复数数据按自旋顺序读入 data_ (交错实部/虚部)
//   5. 主线程通过 buffer() 获取结果并进行 G 向量映射
// ============================================================================

/// @brief 二进制 rhog 读取任务 (在 I/O 工作线程中执行)
class RhogReadTask : public IIOTask
{
  public:
    explicit RhogReadTask(IOBuffer&& buf)
        : buf_(std::move(buf))
    {
    }

    bool execute() override
    {
        if (buf_.filename().empty())
        {
            buf_.set_error("RhogReadTask: empty filename");
            return false;
        }

        // 以二进制模式打开文件
        FILE* fp = std::fopen(buf_.filename().c_str(), "rb");
        if (!fp)
        {
            buf_.set_error("RhogReadTask: cannot open file " + buf_.filename());
            return false;
        }

        // ---- 读取头部: /3/ gamma_only npwtot nspin /3/ ----
        int marker = 0;
        int gamma_only = 0, npwtot = 0, nspin_file = 0;

        if (std::fread(&marker, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "header marker 1"); }
        if (std::fread(&gamma_only, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "gamma_only"); }
        if (std::fread(&npwtot, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "npwtot"); }
        if (std::fread(&nspin_file, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "nspin"); }
        if (std::fread(&marker, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "header marker 2"); }

        // ---- 读取倒格矢: /9/ b1 b2 b3 /9/ ----
        double b1[3], b2[3], b3[3];
        if (std::fread(&marker, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "b marker 1"); }
        if (std::fread(b1, sizeof(double), 3, fp) != 3) { return read_fail_close(fp, "b1"); }
        if (std::fread(b2, sizeof(double), 3, fp) != 3) { return read_fail_close(fp, "b2"); }
        if (std::fread(b3, sizeof(double), 3, fp) != 3) { return read_fail_close(fp, "b3"); }
        if (std::fread(&marker, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "b marker 2"); }

        // ---- 读取 Miller 指数 ----
        int miller_count = 0;
        if (std::fread(&miller_count, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "miller count marker"); }

        int expected_miller = 3 * npwtot;
        if (miller_count != expected_miller)
        {
            std::string err = "RhogReadTask: miller count mismatch (expected "
                              + std::to_string(expected_miller) + ", got "
                              + std::to_string(miller_count) + ")";
            buf_.set_error(err);
            std::fclose(fp);
            return false;
        }

        std::vector<int> miller(miller_count);
        if (std::fread(miller.data(), sizeof(int), miller_count, fp) != static_cast<size_t>(miller_count))
        {
            return read_fail_close(fp, "miller data");
        }

        // 读取尾部标记
        int end_marker = 0;
        if (std::fread(&end_marker, sizeof(int), 1, fp) != 1) { return read_fail_close(fp, "miller end marker"); }

        // ---- 读取所有自旋的 rhog 数据 ----
        // data_ 中存储: [spin0 的复数据交替实部/虚部] [spin1 ...]
        // 每个自旋: npwtot 个 complex<double> = 2 * npwtot 个 double
        std::vector<double> all_data;
        all_data.reserve(static_cast<size_t>(nspin_file) * 2 * npwtot);

        for (int is = 0; is < nspin_file; ++is)
        {
            int spin_marker = 0;
            if (std::fread(&spin_marker, sizeof(int), 1, fp) != 1)
            {
                return read_fail_close(fp, "spin marker " + std::to_string(is));
            }

            // 读取 npwtot 个复数 = 2*npwtot 个 double
            std::vector<double> spin_data(static_cast<size_t>(2) * npwtot);
            size_t items_read = std::fread(spin_data.data(), sizeof(double),
                                           static_cast<size_t>(2) * npwtot, fp);
            if (items_read != static_cast<size_t>(2) * npwtot)
            {
                std::string err = "RhogReadTask: short read at spin "
                                  + std::to_string(is) + " (expected "
                                  + std::to_string(2 * npwtot) + " doubles, got "
                                  + std::to_string(items_read) + ")";
                buf_.set_error(err);
                std::fclose(fp);
                return false;
            }

            all_data.insert(all_data.end(), spin_data.begin(), spin_data.end());

            // 读取尾部标记
            if (std::fread(&spin_marker, sizeof(int), 1, fp) != 1)
            {
                return read_fail_close(fp, "spin end marker " + std::to_string(is));
            }
        }

        std::fclose(fp);

        // ---- 将结果存入 IOBuffer ----
        buf_.set_rhog_result(gamma_only, npwtot, nspin_file,
                             b1, b2, b3, std::move(miller), std::move(all_data));

        return true;
    }

    IOBuffer& buffer() override { return buf_; }
    const IOBuffer& buffer() const override { return buf_; }
    std::string task_name() const override { return "RhogRead"; }

  private:
    IOBuffer buf_;

    /// @brief 读取失败时关闭文件并设置错误消息
    bool read_fail_close(FILE* fp, const std::string& what)
    {
        std::string err = "RhogReadTask: failed to read " + what
                          + " from " + buf_.filename();
        buf_.set_error(err);
        std::fclose(fp);
        return false;
    }
};

#endif // IO_TASK_H
