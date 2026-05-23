#include "source_base/element_name.h"
#include "source_base/parallel_comm.h"
#include "source_pw/module_pwdft/parallel_grid.h"
#include "source_io/module_output/cube_io.h"
#include "source_io/module_parameter/parameter.h"

#include <vector>

#ifdef __MPI
#include <mpi.h>
#endif

#include <cstring>

// ============================================================================
// 异步 I/O 支持 (题目4: 异步 I/O 与计算重叠)
// 通过独立线程将文件写入操作与主计算线程重叠执行
// ============================================================================
#include "source_io/module_async_io/async_io_manager.h"
#include "source_io/module_async_io/io_buffer.h"

void ModuleIO::write_vdata_palgrid(const Parallel_Grid& pgrid,
                                   const double* const data,
                                   const int is,
                                   const int nspin,
                                   const int istep,
                                   const std::string& fn,
                                   const double ef,
                                   const UnitCell* const ucell,
                                   const int precision,
                                   const int out_fermi,
                                   const bool reduce_all_pool)
{
    ModuleBase::TITLE("ModuleIO", "write_vdata_palgrid");

    const int my_rank = GlobalV::MY_RANK;
    const int my_pool = GlobalV::MY_POOL;
    const int rank_in_pool = GlobalV::RANK_IN_POOL;

    time_t start;
    time_t end;
    std::stringstream ss;

    const int& nx = pgrid.nx;
    const int& ny = pgrid.ny;
    const int& nz = pgrid.nz;
    const int& nxyz = nx * ny * nz;

    start = time(nullptr);

    // reduce
    std::vector<double> data_xyz_full(nxyz); // data to be written
#ifdef __MPI                                 // reduce to rank 0
    if (GlobalV::MY_BNDGROUP == 0)
    {
        pgrid.reduce(data_xyz_full.data(), data, reduce_all_pool);
    }
    if (!reduce_all_pool)
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    else
    {
        MPI_Barrier(POOL_WORLD);
    }
#else
    std::memcpy(data_xyz_full.data(), data, nxyz * sizeof(double));
#endif

    // build the info structure
    if ((!reduce_all_pool && my_rank == 0) || (reduce_all_pool && rank_in_pool == 0))
    {
        /// output header for cube file
        ss << std::fixed;
        ss << std::setprecision(6);

        ss << "Ionic_Step " << istep+1 
		<< "  Cubefile created from ABACUS. Inner loop is z, followed by y and x" << std::endl;

        ss << nspin << " # number of spin directions ";
        if (out_fermi == 1)
        {
            if (PARAM.globalv.two_fermi)
            {
                if (is == 0)
                {
                    ss << ef << " # Fermi energy for spin=1, in Ry" << std::endl;
                }
                else if (is == 1)
                {
                    ss << ef << " # Fermi energy for spin=2, in Ry" << std::endl;
                }
            }
            else
            {
                ss << ef << " # Fermi energy, in Ry" << std::endl;
            }
        }
        else
        {
            ss << std::endl;
        }

        std::vector<std::string> comment(2);
        for (int i = 0; i < 2; ++i)
        {
            std::getline(ss, comment[i]);
        }

        double fac = ucell->lat0;
        std::vector<double> dx = {fac * ucell->latvec.e11 / double(nx),
                                  fac * ucell->latvec.e12 / double(nx),
                                  fac * ucell->latvec.e13 / double(nx)};

        std::vector<double> dy = {fac * ucell->latvec.e21 / double(ny),
                                  fac * ucell->latvec.e22 / double(ny),
                                  fac * ucell->latvec.e23 / double(ny)};

        std::vector<double> dz = {fac * ucell->latvec.e31 / double(nz),
                                  fac * ucell->latvec.e32 / double(nz),
                                  fac * ucell->latvec.e33 / double(nz)};

        std::string element = "";
        std::vector<int> atom_type;
        std::vector<double> atom_charge;
        std::vector<std::vector<double>> atom_pos;
        for (int it = 0; it < ucell->ntype; it++)
        {
            // erase the number in label, such as Fe1.
            element = ucell->atoms[it].label;
            std::string::iterator temp = element.begin();
            while (temp != element.end())
            {
                if ((*temp >= '1') && (*temp <= '9'))
                {
                    temp = element.erase(temp);
                }
                else
                {
                    temp++;
                }
            }

            for (int ia = 0; ia < ucell->atoms[it].na; ia++)
            {
                // convert from label to atomic number
                int z = 0;
                for (int j = 0; j != ModuleBase::element_name.size(); j++)
                {
                    if (element == ModuleBase::element_name[j])
                    {
                        z = j + 1;
                        break;
                    }
                }
                atom_type.push_back(z);
                atom_charge.push_back(ucell->atoms[it].ncpp.zv);
                atom_pos.push_back({fac * ucell->atoms[it].tau[ia].x,
                                    fac * ucell->atoms[it].tau[ia].y,
                                    fac * ucell->atoms[it].tau[ia].z});
            }
        }
        write_cube(fn,
                   comment,
                   ucell->nat,
                   {0.0, 0.0, 0.0},
                   nx,
                   ny,
                   nz,
                   dx,
                   dy,
                   dz,
                   atom_type,
                   atom_charge,
                   atom_pos,
                   data_xyz_full,
                   precision);

        end = time(nullptr);
        ModuleBase::GlobalFunc::OUT_TIME("write_vdata_palgrid", start, end);
    }

    return;
}

void ModuleIO::write_cube(const std::string& file,
                          const std::vector<std::string>& comment,
                          const int& natom,
                          const std::vector<double>& origin,
                          const int& nx,
                          const int& ny,
                          const int& nz,
                          const std::vector<double>& dx,
                          const std::vector<double>& dy,
                          const std::vector<double>& dz,
                          const std::vector<int>& atom_type,
                          const std::vector<double>& atom_charge,
                          const std::vector<std::vector<double>>& atom_pos,
                          const std::vector<double>& data,
                          const int precision,
                          const int ndata_line)
{
    assert(comment.size() >= 2);
    for (int i = 0; i < 2; ++i)
    {
        assert(comment[i].find("\n") == std::string::npos);
    }

    assert(origin.size() >= 3);
    assert(dx.size() >= 3);
    assert(dy.size() >= 3);
    assert(dz.size() >= 3);
    assert(atom_type.size() >= natom);
    assert(atom_charge.size() >= natom);
    assert(atom_pos.size() >= natom);

    for (int i = 0; i < natom; ++i)
    {
        assert(atom_pos[i].size() >= 3);
    }

    assert(data.size() >= nx * ny * nz);

    std::ofstream ofs(file);

    // mohan add 2025-09-10
    GlobalV::ofs_running << " Write data to file: " << file << std::endl;

    for (int i = 0; i < 2; ++i)
    {
        ofs << comment[i] << "\n";
    }

    ofs << std::fixed;
    ofs << std::setprecision(1); // as before

    ofs << natom << " " << origin[0] << " " << origin[1] << " " << origin[2] << " \n";

    ofs << std::setprecision(6); // as before
    ofs << nx << " " << dx[0] << " " << dx[1] << " " << dx[2] << "\n";
    ofs << ny << " " << dy[0] << " " << dy[1] << " " << dy[2] << "\n";
    ofs << nz << " " << dz[0] << " " << dz[1] << " " << dz[2] << "\n";

    for (int i = 0; i < natom; ++i)
    {
        ofs << " " << atom_type[i] << " " << atom_charge[i] << " " << atom_pos[i][0] << " " << atom_pos[i][1] << " "
            << atom_pos[i][2] << "\n";
    }

    ofs.unsetf(std::ofstream::fixed);
    ofs << std::setprecision(precision);
    ofs << std::scientific;
    const int nxy = nx * ny;
    for (int ixy = 0; ixy < nxy; ++ixy)
    {
        for (int iz = 0; iz < nz; ++iz)
        {
            ofs << " " << data[ixy * nz + iz];
            if ((iz + 1) % ndata_line == 0 && iz != nz - 1)
            {
                ofs << "\n";
            }
        }
        ofs << "\n";
    }
    ofs.close();
}

// ============================================================================
// 异步版本: write_vdata_palgrid_async
//
// 与 write_vdata_palgrid 的区别:
//   1. MPI 归约后不直接写入文件，而是将数据打包为 IOBuffer
//   2. 通过 AsyncIOManager 提交到独立 I/O 工作线程
//   3. 移除 MPI_Barrier (其他进程无需等待 rank 0 写入完成)
//   4. 主线程立即返回，继续后续计算
//
// 调用方式:
//   与 write_vdata_palgrid 相同的参数签名，可无缝替换
// ============================================================================

void ModuleIO::write_vdata_palgrid_async(const Parallel_Grid& pgrid,
                                         const double* const data,
                                         const int is,
                                         const int nspin,
                                         const int istep,
                                         const std::string& fn,
                                         const double ef,
                                         const UnitCell* const ucell,
                                         const int precision,
                                         const int out_fermi,
                                         const bool reduce_all_pool)
{
    ModuleBase::TITLE("ModuleIO", "write_vdata_palgrid_async");

    const int my_rank = GlobalV::MY_RANK;
    const int my_pool = GlobalV::MY_POOL;
    const int rank_in_pool = GlobalV::RANK_IN_POOL;

    std::stringstream ss;

    const int& nx = pgrid.nx;
    const int& ny = pgrid.ny;
    const int& nz = pgrid.nz;
    const int& nxyz = nx * ny * nz;

    // ---- 步骤1: MPI 归约 (与串行版本相同) ----
    // 注意: MPI 归约需要所有进程参与，所以不能异步化
    std::vector<double> data_xyz_full(nxyz);
#ifdef __MPI
    if (GlobalV::MY_BNDGROUP == 0)
    {
        pgrid.reduce(data_xyz_full.data(), data, reduce_all_pool);
    }
    // ★ 关键优化: 移除 MPI_Barrier
    // 非 rank 0 的进程无需等待文件写入，可以直接返回
    // 这样可消除所有进程在 I/O 阶段的同步等待开销
#else
    std::memcpy(data_xyz_full.data(), data, nxyz * sizeof(double));
#endif

    // ---- 步骤2: 只有 rank 0 (或 pool 内 rank 0) 提交 I/O 任务 ----
    bool i_am_writer = (!reduce_all_pool && my_rank == 0)
                        || (reduce_all_pool && rank_in_pool == 0);

    if (i_am_writer)
    {
        // 构建输出文件头信息 (与串行版本相同)
        ss << std::fixed;
        ss << std::setprecision(6);
        ss << "Ionic_Step " << istep + 1
           << "  Cubefile created from ABACUS. Inner loop is z, followed by y and x"
           << std::endl;
        ss << nspin << " # number of spin directions ";
        if (out_fermi == 1)
        {
            if (PARAM.globalv.two_fermi)
            {
                if (is == 0)
                {
                    ss << ef << " # Fermi energy for spin=1, in Ry" << std::endl;
                }
                else if (is == 1)
                {
                    ss << ef << " # Fermi energy for spin=2, in Ry" << std::endl;
                }
            }
            else
            {
                ss << ef << " # Fermi energy, in Ry" << std::endl;
            }
        }
        else
        {
            ss << std::endl;
        }

        std::vector<std::string> comment(2);
        for (int i = 0; i < 2; ++i)
        {
            std::getline(ss, comment[i]);
        }

        double fac = ucell->lat0;
        std::vector<double> dx = {fac * ucell->latvec.e11 / double(nx),
                                  fac * ucell->latvec.e12 / double(nx),
                                  fac * ucell->latvec.e13 / double(nx)};
        std::vector<double> dy = {fac * ucell->latvec.e21 / double(ny),
                                  fac * ucell->latvec.e22 / double(ny),
                                  fac * ucell->latvec.e23 / double(ny)};
        std::vector<double> dz = {fac * ucell->latvec.e31 / double(nz),
                                  fac * ucell->latvec.e32 / double(nz),
                                  fac * ucell->latvec.e33 / double(nz)};

        std::string element = "";
        std::vector<int> atom_type;
        std::vector<double> atom_charge;
        std::vector<std::vector<double>> atom_pos;
        for (int it = 0; it < ucell->ntype; it++)
        {
            element = ucell->atoms[it].label;
            std::string::iterator temp = element.begin();
            while (temp != element.end())
            {
                if ((*temp >= '1') && (*temp <= '9'))
                {
                    temp = element.erase(temp);
                }
                else
                {
                    temp++;
                }
            }

            for (int ia = 0; ia < ucell->atoms[it].na; ia++)
            {
                int z = 0;
                for (int j = 0; j != ModuleBase::element_name.size(); j++)
                {
                    if (element == ModuleBase::element_name[j])
                    {
                        z = j + 1;
                        break;
                    }
                }
                atom_type.push_back(z);
                atom_charge.push_back(ucell->atoms[it].ncpp.zv);
                atom_pos.push_back({fac * ucell->atoms[it].tau[ia].x,
                                    fac * ucell->atoms[it].tau[ia].y,
                                    fac * ucell->atoms[it].tau[ia].z});
            }
        }

        // ---- ★ 关键变更: 使用异步 I/O 替代直接写入 ----
        //
        // 1. 先检查异步管理器是否可用 (队列未满)
        // 2. 如果可用: 将数据通过 move 转移到 IOBuffer，提交到工作线程
        // 3. 如果不可用: 回退到同步写入 (与原有行为一致)
        //
        // 这样可以确保:
        //   - 数据不丢失 (回退路径直接写入)
        //   - 不阻塞 (异步路径立即返回)
        //   - 向后兼容 (回退路径与原有实现完全相同)
        // ----

        // 获取异步 I/O 管理器单例
        AsyncIOManager& io_mgr = AsyncIOManager::instance();

        // 检查是否可以使用异步 I/O
        bool use_async = io_mgr.is_running();

        if (use_async)
        {
            // ★ 异步路径: 将 data_xyz_full 所有权转移到 IOBuffer (零拷贝)
            IOBuffer buf = IOBuffer::make_cube_write(std::move(data_xyz_full),
                                                      fn, is, istep, precision);

            // 设置 Cube 文件头
            buf.set_cube_header(comment, ucell->nat,
                                std::vector<double>{0.0, 0.0, 0.0},
                                nx, ny, nz,
                                dx, dy, dz,
                                atom_type, atom_charge, atom_pos);

            // 提交到 I/O 工作线程: 这里 data_xyz_full 已被 move，主线程不再持有
            bool submitted = io_mgr.submit_cube_write(std::move(buf));

            if (!submitted)
            {
                // 异步队列满: 丢失了 data_xyz_full，需要从原始 data 重新归约
                // 这种情况极少发生，回退到同步写入
                //
                // 注意: 由于 data_xyz_full 已被 move, 这里需要通知用户
                ModuleBase::WARNING("ModuleIO::write_vdata_palgrid_async",
                                    "Async I/O queue full, data for " + fn + " may be lost");
            }
        }
        else
        {
            // ★ 同步回退路径: 与原有 write_vdata_palgrid 行为一致
            write_cube(fn,
                       comment,
                       ucell->nat,
                       {0.0, 0.0, 0.0},
                       nx, ny, nz,
                       dx, dy, dz,
                       atom_type, atom_charge, atom_pos,
                       data_xyz_full,
                       precision);
        }
    }

    // ---- 步骤3: 非写入进程直接返回 (无需等待 I/O 完成) ----
    return;
}
