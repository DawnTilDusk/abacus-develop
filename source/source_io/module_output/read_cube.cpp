#include "source_io/module_output/cube_io.h"
#include "source_io/module_output/charge_compress.h"
#include <limits>
#include "source_pw/module_pwdft/parallel_grid.h"
#include <cstring>  // use std::memcpy
#include <cmath>
#include <vector>

namespace
{
struct AxisInterpolationMap
{
    int low = 0;
    int high = 0;
    double w_low = 1.0;
    double w_high = 0.0;
};

std::vector<AxisInterpolationMap> build_axis_interpolation_map(const int src_size, const int dst_size)
{
    std::vector<AxisInterpolationMap> axis_map(dst_size);
    if (src_size <= 1)
    {
        for (auto& item : axis_map)
        {
            item.low = 0;
            item.high = 0;
            item.w_low = 1.0;
            item.w_high = 0.0;
        }
        return axis_map;
    }

    const double scale = static_cast<double>(src_size) / static_cast<double>(dst_size);
    const double period = static_cast<double>(src_size);
    for (int i = 0; i < dst_size; ++i)
    {
        double frac = 0.5 * (scale * (1.0 + 2.0 * i) - 1.0);
        frac -= std::floor(frac / period) * period;

        const int low = static_cast<int>(frac);
        const double delta = frac - static_cast<double>(low);

        axis_map[i].low = low;
        axis_map[i].high = (low + 1 == src_size) ? 0 : (low + 1);
        axis_map[i].w_low = 1.0 - delta;
        axis_map[i].w_high = delta;
    }
    return axis_map;
}
} // namespace

bool ModuleIO::read_vdata_palgrid(
    const Parallel_Grid& pgrid,
    const int my_rank,
    std::ofstream& ofs_running,
    const std::string& fn,
    double* const data,
    const int natom)
{
    ModuleBase::TITLE("ModuleIO", "read_vdata_palgrid");

    // check if the file exists
    std::ifstream ifs(fn.c_str());
    if (!ifs)
    {
        std::string tmp_warning_info = "!!! Couldn't find the file: " + fn;
        ofs_running << tmp_warning_info << std::endl;
        return false;
    }
    else
    {
        ofs_running << " Find the file " << fn << " , try to read it." << std::endl;
    }

    // read the full grid data
    const int& nx = pgrid.nx;
    const int& ny = pgrid.ny;
    const int& nz = pgrid.nz;
    const int& nxyz = nx * ny * nz;
    std::vector<double> data_xyz_full(nxyz, 0.0);
    if (my_rank == 0)
    {
        std::vector<std::string> comment;
        int natom = 0;
        std::vector<double> origin;
        std::vector<int> nvoxel;
        int nx_read = 0;
        int ny_read = 0;
        int nz_read = 0;
        std::vector<double> dx(3);
        std::vector<double> dy(3);
        std::vector<double> dz(3);
        std::vector<std::vector<double>> axis_vecs;
        std::vector<int> atom_type;
        std::vector<double> atom_charge;
        std::vector<std::vector<double>> atom_pos;
        std::vector<double> data_read;

        // we've already checked the file existence, so we don't need the returned value here
        ModuleIO::read_cube(fn, comment, natom, origin, nx_read, ny_read, nz_read, 
			dx, dy, dz, atom_type, atom_charge, atom_pos, data_read);

        // if mismatch, trilinear interpolate
        if (nx == nx_read && ny == ny_read && nz == nz_read)
        {
            std::memcpy(data_xyz_full.data(), data_read.data(), nxyz * sizeof(double));
        }
        else
        {
            trilinear_interpolate(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, data_xyz_full.data());
        }
    }

    // distribute
#ifdef __MPI 
    pgrid.bcast(data_xyz_full.data(), data, my_rank);
#else
    std::memcpy(data, data_xyz_full.data(), nxyz * sizeof(double));
#endif
    return true;
}

void ModuleIO::trilinear_interpolate(
    const double* const data_in,
    const int& nx_read,
    const int& ny_read,
    const int& nz_read,
    const int& nx,
    const int& ny,
    const int& nz,
    double* data_out)
{
    ModuleBase::TITLE("ModuleIO", "trilinear_interpolate");

    const std::vector<AxisInterpolationMap> x_map = build_axis_interpolation_map(nx_read, nx);
    const std::vector<AxisInterpolationMap> y_map = build_axis_interpolation_map(ny_read, ny);
    const std::vector<AxisInterpolationMap> z_map = build_axis_interpolation_map(nz_read, nz);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            const AxisInterpolationMap& x_item = x_map[ix];
            const AxisInterpolationMap& y_item = y_map[iy];

            const int idx_x0y0 = (x_item.low * ny_read + y_item.low) * nz_read;
            const int idx_x1y0 = (x_item.high * ny_read + y_item.low) * nz_read;
            const int idx_x0y1 = (x_item.low * ny_read + y_item.high) * nz_read;
            const int idx_x1y1 = (x_item.high * ny_read + y_item.high) * nz_read;

            const double w00 = x_item.w_low * y_item.w_low;
            const double w10 = x_item.w_high * y_item.w_low;
            const double w01 = x_item.w_low * y_item.w_high;
            const double w11 = x_item.w_high * y_item.w_high;

            double* const out_row = data_out + (ix * ny + iy) * nz;

#ifdef _OPENMP
#pragma omp simd
#endif
            for (int iz = 0; iz < nz; iz++)
            {
                const AxisInterpolationMap& z_item = z_map[iz];
                const int lowz = z_item.low;
                const int highz = z_item.high;

                const double low_plane = data_in[idx_x0y0 + lowz] * w00
                                         + data_in[idx_x1y0 + lowz] * w10
                                         + data_in[idx_x0y1 + lowz] * w01
                                         + data_in[idx_x1y1 + lowz] * w11;
                const double high_plane = data_in[idx_x0y0 + highz] * w00
                                          + data_in[idx_x1y0 + highz] * w10
                                          + data_in[idx_x0y1 + highz] * w01
                                          + data_in[idx_x1y1 + highz] * w11;

                out_row[iz] = low_plane * z_item.w_low + high_plane * z_item.w_high;
            }
        }
    }
}

bool ModuleIO::read_cube(const std::string& file,
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
    std::vector<double>& data)
{
    std::ifstream ifs(file);

    if (!ifs) 
    { 
	    return false; 
    }

    comment.resize(2);
    for (auto& c : comment) 
    { 
	    std::getline(ifs, c); 
    }

    ifs >> natom;

    origin.resize(3);
    for (auto& cp : origin) 
    { 
	    ifs >> cp; 
    }

    dx.resize(3);
    dy.resize(3);
    dz.resize(3);
    ifs >> nx >> dx[0] >> dx[1] >> dx[2];
    ifs >> ny >> dy[0] >> dy[1] >> dy[2];
    ifs >> nz >> dz[0] >> dz[1] >> dz[2];

    atom_type.resize(natom);
    atom_charge.resize(natom);
    atom_pos.resize(natom, std::vector<double>(3));
    for (int i = 0;i < natom;++i)
    {
        ifs >> atom_type[i] >> atom_charge[i] >> atom_pos[i][0] >> atom_pos[i][1] >> atom_pos[i][2];
    }

    const int nxyz = nx * ny * nz;
    data.resize(nxyz);

    // Record position after header to detect compression / binary format.
    // Text data starts with digit/minus/dot.
    // Compressed data starts with 'Z' (first byte of "ZCMP"/"ZCM2" magic in LE).
    // MPI binary data starts with 'C' (first byte of "CIPM" marker in LE).
    ifs >> std::ws;
    std::streampos data_start = ifs.tellg();
    int next_char = ifs.peek();

    // Attempt compressed read ('Z' = ZCMP v0 or ZCM2 v1 magic)
    if (next_char == 'Z' || next_char == 'C')
    {
        ifs.close();

        std::ifstream ifs_bin(file, std::ios::binary | std::ios::ate);
        size_t file_size = ifs_bin.tellg();
        size_t header_size = static_cast<size_t>(data_start);
        size_t raw_len = (file_size > header_size) ? (file_size - header_size) : 0;

        if (raw_len >= 4)
        {
            std::vector<uint8_t> raw_buf(raw_len);
            ifs_bin.seekg(data_start);
            ifs_bin.read(reinterpret_cast<char*>(raw_buf.data()), raw_len);
            ifs_bin.close();

            uint32_t magic = 0;
            std::memcpy(&magic, raw_buf.data(), 4);

            // Check for zlib-compressed data (ZCMP v0 or ZCM2 v1 magic)
            if (magic == CHARGE_COMPRESS_MAGIC || magic == CHARGE_COMPRESS_MAGIC_V1)
            {
                // Universal decompress auto-detects v0/v1 format
                if (decompress_charge_data_any(raw_buf.data(), raw_len, data.data(), nxyz))
                    return true;
            }

            // Check for MPI-parallel binary data (CIPM marker)
            static constexpr uint32_t CUBE_MPI_MARKER = 0x4D504943;
            if (magic == CUBE_MPI_MARKER)
            {
                // MPI binary format: 4B marker + nxyz * sizeof(double)
                size_t expected_size = 4 + static_cast<size_t>(nxyz) * sizeof(double);
                if (raw_len >= expected_size)
                {
                    std::memcpy(data.data(), raw_buf.data() + 4, nxyz * sizeof(double));
                    return true;
                }
            }
        }

        // Fallback: reopen as text and parse
        ifs.open(file);
        ifs.seekg(data_start);
    }

    for (int i = 0;i < nxyz;++i)
    {
	    ifs >> data[i];
    }

    ifs.close();
    return true;
}
