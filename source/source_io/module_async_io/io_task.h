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

#endif // IO_TASK_H
