#ifndef CUBE_IO_H
#define CUBE_IO_H
#include "source_cell/unitcell.h"

#include <string>
class Parallel_Grid;

#ifdef __MPI
#include <mpi.h>
#endif

namespace ModuleIO
{
/// read volumetric data from .cube file into the parallel distributed grid.
bool read_vdata_palgrid(const Parallel_Grid& pgrid,
                        const int my_rank,
                        std::ofstream& ofs_running,
                        const std::string& fn,
                        double* const data,
                        const int nat);

/// write volumetric data on the parallized grid into a .cube file
void write_vdata_palgrid(const Parallel_Grid& pgrid,
                         const double* const data,
                         const int is,
                         const int nspin,
                         const int iter,
                         const std::string& fn,
                         const double ef,
                         const UnitCell* const ucell,
                         const int precision = 11,
                         const int out_fermi = 1,
                         const bool reduce_all_pool = false); // only reduce in the main pool as default

/// read the full data from a cube file
bool read_cube(const std::string& file,
               std::vector<std::string>& comment,
               int& natom,
               std::vector<double>& origin,
               int& nx,
               int& ny,
               int& nz,
               std::vector<double>& dx,
               std::vector<double>& dy,
               std::vector<double>& dz,
               std::vector<int>& atom_type,
               std::vector<double>& atom_charge,
               std::vector<std::vector<double>>& atom_pos,
               std::vector<double>& data);

/// write a cube file
void write_cube(const std::string& file,
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
                const int ndata_line = 6,
                const bool compress = false,
                const int compress_nthreads = 0);

/// MPI-IO parallel cube file write. All ranks must have the full data array.
/// Each rank writes its z-slice range via collective MPI-IO.
/// The MPI communicator must be provided.
#ifdef __MPI
void write_cube_mpi(const std::string& file,
                    const std::vector<std::string>& comment,
                    const int& natom,
                    const std::vector<double>& origin,
                    const int& nx, const int& ny, const int& nz,
                    const std::vector<double>& dx,
                    const std::vector<double>& dy,
                    const std::vector<double>& dz,
                    const std::vector<int>& atom_type,
                    const std::vector<double>& atom_charge,
                    const std::vector<std::vector<double>>& atom_pos,
                    const std::vector<double>& data,
                    const int precision,
                    const MPI_Comm& comm);
#endif

// ============================================================================
// 异步 I/O 接口 (题目4: 异步 I/O 与计算重叠)
//
// 提供异步版本的 write_vdata_palgrid，将文件 I/O 操作提交到独立的
// I/O 工作线程执行，使主计算线程无需等待文件写入完成即可继续计算。
//
// 使用方法:
//   1. 程序启动时调用 AsyncIOManager::instance().start()
//   2. 在需要写入 Cube 文件的位置，调用 write_vdata_palgrid_async()
//      替代 write_vdata_palgrid()，传入相同的参数
//   3. 在需要确保所有文件已写入完成的位置(例如程序退出前)，
//      调用 AsyncIOManager::instance().wait_all()
//   4. 程序退出前调用 AsyncIOManager::instance().stop()
//
// 性能收益:
//   - I/O 与计算重叠: 主线程提交任务后立即返回，继续后续计算
//   - 减少 MPI_Barrier 等待: 移除与 I/O 相关的同步屏障
//   - 双缓冲机制: I/O 线程写一个缓冲区时，主线程可填充另一个缓冲区
// ============================================================================

/// @brief 异步版本的 write_vdata_palgrid
/// 将 MPI 归约后的数据提交到异步 I/O 管理器，由独立工作线程写入文件
void write_vdata_palgrid_async(const Parallel_Grid& pgrid,
                               const double* const data,
                               const int is,
                               const int nspin,
                               const int iter,
                               const std::string& fn,
                               const double ef,
                               const UnitCell* const ucell,
                               const int precision = 11,
                               const int out_fermi = 1,
                               const bool reduce_all_pool = false);

/**
 * @brief The trilinear interpolation method
 *
 * Trilinear interpolation is a method for interpolating grid data in 3D space.
 * It estimates the value at a given position by interpolating the data along the three adjacent points.
 *
 * Specifically, for 3D grid data, trilinear interpolation requires determining the eight data points that are
 * closest to the point where the estimation is required. These data points form a cube, with vertices at
 * (x0,y0,z0), (x0+1,y0,z0), (x0,y0+1,z0), (x0+1,y0+1,z0), (x0,y0,z0+1), (x0+1,y0,z0+1), (x0,y0+1,z0+1) and
 * (x0+1,y0+1,z0+1). Here, (x0,y0,z0) is the data point closest to the estimation point and has coordinate
 * values less than those of the estimation point.
 *
 * For the estimation location (x,y,z), its estimated value in the grid can be calculated using the following
 * formula: f(x,y,z) = f000(1-dx)(1-dy)(1-dz) + f100dx(1-dy)(1-dz) + f010(1-dx)dy(1-dz) +
 *             f001(1-dx)(1-dy)dz + f101dx(1-dy)dz + f011(1-dx)dydz +
 *             f110dxdy(1-dz) + f111dxdydz
 * where fijk represents the data value at vertex i,j,k of the cube, and dx = x - x0, dy = y - y0, dz = z - z0
 * represent the distance between the estimation point and the closest data point in each of the three
 * directions, divided by the grid spacing. Here, it is assumed that the grid spacing is equal and can be
 * omitted during computation.
 *
 * @param data_in the input data of size nxyz_read
 * @param nx_read nx read from file
 * @param ny_read ny read from file
 * @param nz_read nz read from file
 * @param nx the dimension of grids along x
 * @param ny the dimension of grids along y
 * @param nz the dimension of grids along z
 * @param data_out the interpolated results of size nxyz
 */
void trilinear_interpolate(const double* const data_in,
                           const int& nx_read,
                           const int& ny_read,
                           const int& nz_read,
                           const int& nx,
                           const int& ny,
                           const int& nz,
                           double* data_out);
} // namespace ModuleIO

#endif
