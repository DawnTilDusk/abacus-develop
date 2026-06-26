#include "source_base/element_name.h"
#include "source_base/parallel_comm.h"
#include "source_pw/module_pwdft/parallel_grid.h"
#include "source_io/module_output/cube_io.h"
#include "source_io/module_output/charge_compress.h"
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

// ============================================================================
// OMP task 异步 I/O (OpenMP 4.0 task+depend, 无 std::thread)
// 通过 USE_OMP_TASK_ASYNC 宏在 CMake 中切换启用
// ============================================================================
#ifdef USE_OMP_TASK_ASYNC
#include "source_io/module_async_io/omp_task/omp_task_manager.h"
#endif

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
                                   const bool two_fermi,
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
            if (two_fermi)
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
        // ================================================================
        // 异步 I/O 集成点 (题目4)
        // ================================================================
#ifdef USE_OMP_TASK_ASYNC
        OMPTaskManager& omp_mgr = OMPTaskManager::instance();
        if (omp_mgr.is_running())
        {
            // ── OpenMP 4.0 task 路径 ──
            IOBuffer buf = IOBuffer::make_cube_write(std::move(data_xyz_full),
                                                      fn, is, istep, precision);
            buf.set_cube_header(comment, ucell->nat,
                                std::vector<double>{0.0, 0.0, 0.0},
                                nx, ny, nz,
                                dx, dy, dz,
                                atom_type, atom_charge, atom_pos);

            auto task = std::unique_ptr<IIOTask>(
                new CubeWriteTask(std::move(buf)));
            omp_mgr.submit_cube_write(std::move(task));
        }
        else
        {
            write_cube(fn,
                       comment,
                       ucell->nat,
                       {0.0, 0.0, 0.0},
                       nx, ny, nz,
                       dx, dy, dz,
                       atom_type, atom_charge, atom_pos,
                       data_xyz_full, precision);

            end = time(nullptr);
            ModuleBase::GlobalFunc::OUT_TIME("write_vdata_palgrid", start, end);
        }
#else
        AsyncIOManager& io_mgr = AsyncIOManager::instance();
        if (io_mgr.is_running() && io_mgr.can_submit())
        {
            // ★ 异步路径: have checked can_submit() first → submit now
            //    `data_xyz_full` will be moved into IOBuffer, then worker thread writes it.
            //    If queue full, falls through to sync path below (data_xyz_full untouched).
            IOBuffer buf = IOBuffer::make_cube_write(std::move(data_xyz_full),
                                                      fn, is, istep, precision);
            buf.set_cube_header(comment, ucell->nat,
                                std::vector<double>{0.0, 0.0, 0.0},
                                nx, ny, nz,
                                dx, dy, dz,
                                atom_type, atom_charge, atom_pos);
            io_mgr.submit_cube_write(std::move(buf));
        }
        else if (io_mgr.is_running())
        {
            // ★ 异步 I/O 队列已满: 回退到同步写入
            //    data_xyz_full 未被移动，可以安全使用
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
        else
        {
            // ★ 同步回退路径: 与原有行为完全一致
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
#endif  // USE_OMP_TASK_ASYNC / else (original std::thread path)
    }  // closes: if ((!reduce_all_pool && my_rank == 0) || ...)

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
                          const int ndata_line,
                          const bool compress,
                          const int compress_nthreads)
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

    if (compress)
    {
        // Write header as text, then append compressed binary data section.
        // Close the text stream first so we can reopen in binary append mode.
        ofs.close();

        // Compress the data using v1 format (auto-fallback to raw if not beneficial)
        size_t nxyz = static_cast<size_t>(nx) * ny * nz;
        std::vector<uint8_t> cbuf;
        bool ok = false;
        if (compress_nthreads > 1)
        {
            ok = compress_charge_data_omp_v1(data.data(), nxyz, cbuf, compress_nthreads);
        }
        else
        {
            ok = compress_charge_data_v1(data.data(), nxyz, cbuf);
        }

        if (!ok)
        {
            ModuleBase::WARNING_QUIT("ModuleIO::write_cube",
                                     "Failed to compress charge data");
            return;
        }

        // Append compressed binary blob to the file
        std::ofstream ofs_bin(file, std::ios::binary | std::ios::app);
        ofs_bin.write(reinterpret_cast<const char*>(cbuf.data()), cbuf.size());
        ofs_bin.close();
    }
    else
    {
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
}

#ifdef __MPI

// Binary marker to identify MPI-parallel binary cube files
// "CIPM" = Cube I/O Parallel Marker (little-endian)
static constexpr uint32_t CUBE_MPI_MARKER = 0x4D504943;

void ModuleIO::write_cube_mpi(const std::string& file,
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
                               const MPI_Comm& comm)
{
    assert(comment.size() >= 2);
    assert(origin.size() >= 3);
    assert(data.size() >= static_cast<size_t>(nx) * ny * nz);

    int nprocs = 1;
    int my_rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &my_rank);

    const int nxy = nx * ny;

    // Compute z-slice distribution across ranks
    int nz_local = nz / nprocs;
    int remainder = nz % nprocs;
    int z_start = 0;
    for (int r = 0; r < my_rank; ++r)
    {
        z_start += nz_local + (r < remainder ? 1 : 0);
    }
    if (my_rank < remainder)
        nz_local += 1;

    const int my_count = nz_local * nxy;

    // Build the text header string (rank 0 only) — identical to standard cube header
    std::string header_str;
    size_t header_bytes = 0;

    if (my_rank == 0)
    {
        std::ostringstream hdr;
        hdr << std::fixed;

        for (int i = 0; i < 2; ++i)
            hdr << comment[i] << "\n";

        hdr << std::setprecision(1);
        hdr << natom << " " << origin[0] << " " << origin[1] << " " << origin[2] << " \n";

        hdr << std::setprecision(6);
        hdr << nx << " " << dx[0] << " " << dx[1] << " " << dx[2] << "\n";
        hdr << ny << " " << dy[0] << " " << dy[1] << " " << dy[2] << "\n";
        hdr << nz << " " << dz[0] << " " << dz[1] << " " << dz[2] << "\n";

        for (int i = 0; i < natom; ++i)
        {
            hdr << " " << atom_type[i] << " " << atom_charge[i] << " "
                << atom_pos[i][0] << " " << atom_pos[i][1] << " " << atom_pos[i][2] << "\n";
        }

        header_str = hdr.str();
        header_bytes = header_str.size();
    }

    MPI_Bcast(&header_bytes, 1, MPI_UNSIGNED_LONG, 0, comm);

    // Open file with MPI-IO
    MPI_File fh;
    MPI_File_open(comm, file.c_str(),
                  MPI_MODE_WRONLY | MPI_MODE_CREATE,
                  MPI_INFO_NULL, &fh);

    // Rank 0 writes the text header + binary marker
    if (my_rank == 0)
    {
        MPI_File_write_at(fh, 0, header_str.data(), static_cast<int>(header_bytes),
                          MPI_CHAR, MPI_STATUS_IGNORE);
        // Write binary marker so readers can detect MPI-parallel binary format
        MPI_File_write_at(fh, static_cast<MPI_Offset>(header_bytes),
                          &CUBE_MPI_MARKER, 1, MPI_UINT32_T, MPI_STATUS_IGNORE);
    }

    // Barrier to ensure header is written before data
    MPI_Barrier(comm);

    // Each rank writes its z-slice data using MPI file views for correct
    // z-fastest ordering. File layout: for each ixy, all nz values
    // consecutively. Each rank writes its nz_local values within each nz block.
    MPI_Offset data_offset = static_cast<MPI_Offset>(header_bytes) + sizeof(uint32_t);

    // Create strided file view: for each ixy row, this rank writes nz_local
    // values with a stride of nz (skipping data written by other ranks).
    MPI_Datatype filetype;
    MPI_Type_vector(nxy, nz_local, nz, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);

    // Set file view: the view starts at this rank's z offset within each block
    MPI_File_set_view(fh, data_offset + z_start * sizeof(double),
                      MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

    // Pack local buffer in row-major order matching the file view
    std::vector<double> buf(my_count);
    for (int ixy = 0; ixy < nxy; ++ixy)
    {
        for (int iz = 0; iz < nz_local; ++iz)
        {
            buf[ixy * nz_local + iz] = data[ixy * nz + (z_start + iz)];
        }
    }

    MPI_File_write_all(fh, buf.data(), my_count, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_Type_free(&filetype);

    MPI_File_close(&fh);
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
        // 策略: 先检查队列是否可提交，再决定是否移动数据
        //   - 队列有空位: 将 data_xyz_full 通过 move 转移到 IOBuffer，提交到工作线程
        //   - 队列满: 直接同步写入 (data_xyz_full 未被移动，可以安全使用)
        //
        // 这样可以确保:
        //   - 数据不丢失 (满时 safe fallback 到同步写入)
        //   - 不阻塞 (异步路径立即返回)
        //   - 向后兼容 (同步写入与原有实现完全相同)
        // ----

        // 获取异步 I/O 管理器单例
#ifdef USE_OMP_TASK_ASYNC
        OMPTaskManager& omp_mgr = OMPTaskManager::instance();
        if (omp_mgr.is_running())
        {
            IOBuffer buf = IOBuffer::make_cube_write(std::move(data_xyz_full),
                                                      fn, is, istep, precision);
            buf.set_cube_header(comment, ucell->nat,
                                std::vector<double>{0.0, 0.0, 0.0},
                                nx, ny, nz,
                                dx, dy, dz,
                                atom_type, atom_charge, atom_pos);
            auto task = std::unique_ptr<IIOTask>(
                new CubeWriteTask(std::move(buf)));
            omp_mgr.submit_cube_write(std::move(task));
        }
        else
        {
            write_cube(fn, comment, ucell->nat, {0.0, 0.0, 0.0},
                       nx, ny, nz, dx, dy, dz,
                       atom_type, atom_charge, atom_pos,
                       data_xyz_full, precision);
        }
#else
        AsyncIOManager& io_mgr = AsyncIOManager::instance();

        if (io_mgr.is_running() && io_mgr.can_submit())
        {
            // ★ 异步路径: 队列有空位，将 data_xyz_full 所有权转移到 IOBuffer (零拷贝)
            IOBuffer buf = IOBuffer::make_cube_write(std::move(data_xyz_full),
                                                      fn, is, istep, precision);

            // 设置 Cube 文件头
            buf.set_cube_header(comment, ucell->nat,
                                std::vector<double>{0.0, 0.0, 0.0},
                                nx, ny, nz,
                                dx, dy, dz,
                                atom_type, atom_charge, atom_pos);

            // 提交到 I/O 工作线程: 此时队列有空位，必定成功
            io_mgr.submit_cube_write(std::move(buf));
        }
        else
        {
            // ★ 同步回退路径: 队列满或异步 I/O 不可用，直接写入
            // data_xyz_full 未被移动，可以安全使用
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
#endif  // USE_OMP_TASK_ASYNC / else (original std::thread path)
    }  // closes: if (i_am_writer)

    // ---- 步骤3: 非写入进程直接返回 (无需等待 I/O 完成) ----
    return;
}
#endif  // __MPI
