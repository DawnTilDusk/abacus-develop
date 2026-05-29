#ifndef IO_BUFFER_H
#define IO_BUFFER_H

#include <string>
#include <vector>
#include <complex>
#include <cstddef>

// ============================================================================
// IOBuffer: 异步 I/O 的数据容器
//
// 职责:
//   将待写入/读取的数据以及相关元信息打包成一个独立单元，
//   便于在生产线程(主计算线程)和消费线程(I/O工作线程)之间安全传递。
//
// 设计要点:
//   1. 使用 move 语义避免深拷贝，降低数据传递开销。
//   2. 每个 IOBuffer 包含文件路径、自旋索引、步号等完整上下文。
//   3. 同时支持写入(write)和读取(read)两种场景。
//   4. 内部使用 std::vector<double> 管理数据，RAII 自动释放。
// ============================================================================

/// @brief 异步 I/O 缓冲区 —— 封装一次 I/O 操作所需的所有数据
class IOBuffer
{
  public:
    // ---- 构造/析构 ----
    IOBuffer() = default;

    /// @brief 禁止拷贝 (数据应通过 move 传递)
    IOBuffer(const IOBuffer&) = delete;
    IOBuffer& operator=(const IOBuffer&) = delete;

    /// @brief 允许 move 构造
    IOBuffer(IOBuffer&& other) noexcept
        : data_(std::move(other.data_)),
          filename_(std::move(other.filename_)),
          spin_index_(other.spin_index_),
          istep_(other.istep_),
          precision_(other.precision_),
          is_binary_(other.is_binary_),
          error_message_(std::move(other.error_message_)),
          // ---- rhog 读取元数据 ----
          npwtot_in_(other.npwtot_in_),
          gamma_only_in_(other.gamma_only_in_),
          nspin_in_(other.nspin_in_),
          b1_(std::move(other.b1_)),
          b2_(std::move(other.b2_)),
          b3_(std::move(other.b3_)),
          miller_(std::move(other.miller_)),
          // ---- Cube 文件头信息 ----
          comment_(std::move(other.comment_)),
          natom_(other.natom_),
          origin_(std::move(other.origin_)),
          nx_(other.nx_),
          ny_(other.ny_),
          nz_(other.nz_),
          dx_(std::move(other.dx_)),
          dy_(std::move(other.dy_)),
          dz_(std::move(other.dz_)),
          atom_type_(std::move(other.atom_type_)),
          atom_charge_(std::move(other.atom_charge_)),
          atom_pos_(std::move(other.atom_pos_))
    {
        // 置空源对象，确保析构安全
        other.spin_index_ = 0;
        other.istep_ = 0;
        other.precision_ = 11;
        other.is_binary_ = false;
        other.natom_ = 0;
        other.nx_ = other.ny_ = other.nz_ = 0;
        other.npwtot_in_ = 0;
        other.gamma_only_in_ = 0;
        other.nspin_in_ = 0;
    }

    /// @brief 允许 move 赋值
    IOBuffer& operator=(IOBuffer&& other) noexcept
    {
        if (this != &other)
        {
            data_ = std::move(other.data_);
            filename_ = std::move(other.filename_);
            spin_index_ = other.spin_index_;
            istep_ = other.istep_;
            precision_ = other.precision_;
            is_binary_ = other.is_binary_;
            error_message_ = std::move(other.error_message_);
            // ---- rhog 读取元数据 ----
            npwtot_in_ = other.npwtot_in_;
            gamma_only_in_ = other.gamma_only_in_;
            nspin_in_ = other.nspin_in_;
            b1_ = std::move(other.b1_);
            b2_ = std::move(other.b2_);
            b3_ = std::move(other.b3_);
            miller_ = std::move(other.miller_);
            // ---- Cube 文件头信息 ----
            comment_ = std::move(other.comment_);
            natom_ = other.natom_;
            origin_ = std::move(other.origin_);
            nx_ = other.nx_;
            ny_ = other.ny_;
            nz_ = other.nz_;
            dx_ = std::move(other.dx_);
            dy_ = std::move(other.dy_);
            dz_ = std::move(other.dz_);
            atom_type_ = std::move(other.atom_type_);
            atom_charge_ = std::move(other.atom_charge_);
            atom_pos_ = std::move(other.atom_pos_);

            other.spin_index_ = 0;
            other.istep_ = 0;
            other.precision_ = 11;
            other.is_binary_ = false;
            other.natom_ = 0;
            other.nx_ = other.ny_ = other.nz_ = 0;
            other.npwtot_in_ = 0;
            other.gamma_only_in_ = 0;
            other.nspin_in_ = 0;
        }
        return *this;
    }

    ~IOBuffer() = default;

    // ======================== 写入场景接口 ========================

    /// @brief 创建一个 Cube 文件写入缓冲区
    /// @param data    待写入的完整网格数据 (nx*ny*nz)
    /// @param fn      输出文件路径
    /// @param is      自旋通道索引
    /// @param istep   离子步编号
    /// @param prec    输出精度 (小数位数)
    static IOBuffer make_cube_write(std::vector<double>&& data,
                                    const std::string& fn,
                                    int is,
                                    int istep,
                                    int prec = 11)
    {
        IOBuffer buf;
        buf.data_ = std::move(data);
        buf.filename_ = fn;
        buf.spin_index_ = is;
        buf.istep_ = istep;
        buf.precision_ = prec;
        buf.is_binary_ = false;
        return buf;
    }

    /// @brief 创建一个二进制文件写入缓冲区
    static IOBuffer make_binary_write(std::vector<double>&& data,
                                      const std::string& fn)
    {
        IOBuffer buf;
        buf.data_ = std::move(data);
        buf.filename_ = fn;
        buf.is_binary_ = true;
        return buf;
    }

    // ======================== 读取场景接口 ========================

    /// @brief 创建一个读取结果缓冲区 (由 I/O 工作线程填充)
    static IOBuffer make_read_result(const std::string& fn, size_t expected_size)
    {
        IOBuffer buf;
        buf.filename_ = fn;
        buf.data_.reserve(expected_size);
        buf.is_binary_ = false;
        return buf;
    }

    /// @brief 创建一个 rhog 二进制读取缓冲区
    /// @param fn  二进制重启文件路径
    /// @param npw 进程本地 G 向量数 (用于留出 data_ 空间)
    static IOBuffer make_binary_read_result(const std::string& fn, int npw)
    {
        IOBuffer buf;
        buf.filename_ = fn;
        buf.is_binary_ = true;
        // 预分配空间: 每个 G 向量一个复数 (2 doubles)
        buf.data_.reserve(2 * npw);
        return buf;
    }

    // ======================== Cube 头信息设置 ========================

    /// @brief 设置 Cube 文件头信息 (在 submit 前由主线程填充)
    void set_cube_header(const std::vector<std::string>& comment,
                         int natom,
                         const std::vector<double>& origin,
                         int nx,
                         int ny,
                         int nz,
                         const std::vector<double>& dx,
                         const std::vector<double>& dy,
                         const std::vector<double>& dz,
                         const std::vector<int>& atom_type,
                         const std::vector<double>& atom_charge,
                         const std::vector<std::vector<double>>& atom_pos)
    {
        comment_ = comment;
        natom_ = natom;
        origin_ = origin;
        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
        dx_ = dx;
        dy_ = dy;
        dz_ = dz;
        atom_type_ = atom_type;
        atom_charge_ = atom_charge;
        atom_pos_ = atom_pos;
    }

    // ======================== 只读访问器 ========================

    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }
    const std::string& filename() const { return filename_; }
    int spin_index() const { return spin_index_; }
    int istep() const { return istep_; }
    int precision() const { return precision_; }
    bool is_binary() const { return is_binary_; }

    const std::string& error_message() const { return error_message_; }
    void set_error(const std::string& msg) { error_message_ = msg; }
    bool has_error() const { return !error_message_.empty(); }

    // ======================== Rhog 二进制读取访问器 ========================

    /// @brief 设置 rhog 读取结果 (由 RhogReadTask 在工作线程中调用)
    void set_rhog_result(int gamma_only,
                         int npwtot,
                         int nspin_file,
                         const double b1[3],
                         const double b2[3],
                         const double b3[3],
                         std::vector<int>&& miller,
                         std::vector<double>&& data)
    {
        gamma_only_in_ = gamma_only;
        npwtot_in_ = npwtot;
        nspin_in_ = nspin_file;
        b1_.assign(b1, b1 + 3);
        b2_.assign(b2, b2 + 3);
        b3_.assign(b3, b3 + 3);
        miller_ = std::move(miller);
        data_ = std::move(data);
    }

    int gamma_only_in() const { return gamma_only_in_; }
    int npwtot_in() const { return npwtot_in_; }
    int nspin_in() const { return nspin_in_; }
    const std::vector<double>& b1() const { return b1_; }
    const std::vector<double>& b2() const { return b2_; }
    const std::vector<double>& b3() const { return b3_; }
    const std::vector<int>& miller() const { return miller_; }

    // Cube 头访问器
    const std::vector<std::string>& comment() const { return comment_; }
    int natom() const { return natom_; }
    const std::vector<double>& origin() const { return origin_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    const std::vector<double>& dx() const { return dx_; }
    const std::vector<double>& dy() const { return dy_; }
    const std::vector<double>& dz() const { return dz_; }
    const std::vector<int>& atom_type() const { return atom_type_; }
    const std::vector<double>& atom_charge() const { return atom_charge_; }
    const std::vector<std::vector<double>>& atom_pos() const { return atom_pos_; }

    /// @brief 数据总字节数
    size_t total_bytes() const { return data_.size() * sizeof(double); }

    /// @brief 释放内部数据，归还内存
    void clear() { data_.clear(); data_.shrink_to_fit(); }

  private:
    // ---- 核心数据 ----
    std::vector<double> data_;          ///< 存储的浮点数据 (电荷密度网格)
    std::string filename_;              ///< 目标文件路径
    int spin_index_ = 0;                ///< 自旋通道索引 (0-based)
    int istep_ = 0;                     ///< 离子步编号
    int precision_ = 11;                ///< 文本格式精度
    bool is_binary_ = false;            ///< true=二进制格式, false=文本格式

    // ---- 错误状态 ----
    std::string error_message_;         ///< 如果非空，表示 I/O 操作失败了

    // ---- Rhog 二进制读取元数据 (在 RhogReadTask 中由 I/O 线程填充) ----
    int npwtot_in_ = 0;                  ///< 文件中的总 G 向量数
    int gamma_only_in_ = 0;              ///< 文件是否仅 Gamma 点
    int nspin_in_ = 0;                   ///< 文件中的自旋数
    std::vector<double> b1_, b2_, b3_;   ///< 倒格矢 (各 3 个 double)
    std::vector<int> miller_;            ///< Miller 指数 [3 * npwtot_in_]

    // ---- Cube 文件头信息 (仅在 write_cube 场景使用) ----
    std::vector<std::string> comment_;           ///< 文件注释行 (2行)
    int natom_ = 0;                               ///< 原子数
    std::vector<double> origin_;                  ///< 原点坐标
    int nx_ = 0, ny_ = 0, nz_ = 0;               ///< 网格维度
    std::vector<double> dx_, dy_, dz_;            ///< 格点向量
    std::vector<int> atom_type_;                  ///< 原子类型 (原子序数)
    std::vector<double> atom_charge_;             ///< 原子电荷
    std::vector<std::vector<double>> atom_pos_;   ///< 原子位置
};

#endif // IO_BUFFER_H
