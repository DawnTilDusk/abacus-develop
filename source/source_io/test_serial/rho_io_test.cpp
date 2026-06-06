#include "source_io/module_output/cube_io.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <vector>
#include "source_base/global_variable.h"
#include "source_io/module_output/cube_io.h"
#include "prepare_unitcell.h"
#include "source_pw/module_pwdft/parallel_grid.h"

#ifdef __LCAO
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}
LCAO_Orbitals::LCAO_Orbitals()
{
}
LCAO_Orbitals::~LCAO_Orbitals()
{
}
#endif


Magnetism::Magnetism()
{
    this->tot_mag = 0.0;
    this->abs_mag = 0.0;
    this->start_mag = nullptr;
}


Magnetism::~Magnetism()
{
    delete[] this->start_mag;
}
Parallel_Grid::~Parallel_Grid() {}


#define private public
#include "source_io/module_parameter/parameter.h"
#undef private

/***************************************************************
 *  unit test of read_rho, write_rho and trilinear_interpolate
 ***************************************************************/

namespace
{
struct RefAxisMap
{
    int low = 0;
    int high = 0;
    double w_low = 1.0;
    double w_high = 0.0;
};

RefAxisMap build_ref_axis_map(const int src_size, const int dst_size, const int dst_index)
{
    if (src_size <= 1)
    {
        return {0, 0, 1.0, 0.0};
    }

    double frac = 0.5 * (static_cast<double>(src_size) / static_cast<double>(dst_size) * (1.0 + 2.0 * dst_index) - 1.0);
    frac -= std::floor(frac / static_cast<double>(src_size)) * static_cast<double>(src_size);

    const int low = static_cast<int>(std::floor(frac));
    const double delta = frac - static_cast<double>(low);
    return {low, (low + 1 == src_size) ? 0 : (low + 1), 1.0 - delta, delta};
}

double trilinear_reference_value(const double* const data_in,
                                 const int nx_read,
                                 const int ny_read,
                                 const int nz_read,
                                 const int nx,
                                 const int ny,
                                 const int nz,
                                 const int ix,
                                 const int iy,
                                 const int iz)
{
    const RefAxisMap x_map = build_ref_axis_map(nx_read, nx, ix);
    const RefAxisMap y_map = build_ref_axis_map(ny_read, ny, iy);
    const RefAxisMap z_map = build_ref_axis_map(nz_read, nz, iz);

    const auto index = [=](const int x, const int y, const int z) {
        return (x * ny_read + y) * nz_read + z;
    };

    return data_in[index(x_map.low, y_map.low, z_map.low)] * x_map.w_low * y_map.w_low * z_map.w_low
         + data_in[index(x_map.high, y_map.low, z_map.low)] * x_map.w_high * y_map.w_low * z_map.w_low
         + data_in[index(x_map.low, y_map.high, z_map.low)] * x_map.w_low * y_map.w_high * z_map.w_low
         + data_in[index(x_map.low, y_map.low, z_map.high)] * x_map.w_low * y_map.w_low * z_map.w_high
         + data_in[index(x_map.high, y_map.high, z_map.low)] * x_map.w_high * y_map.w_high * z_map.w_low
         + data_in[index(x_map.high, y_map.low, z_map.high)] * x_map.w_high * y_map.w_low * z_map.w_high
         + data_in[index(x_map.low, y_map.high, z_map.high)] * x_map.w_low * y_map.w_high * z_map.w_high
         + data_in[index(x_map.high, y_map.high, z_map.high)] * x_map.w_high * y_map.w_high * z_map.w_high;
}

void trilinear_reference(const double* const data_in,
                         const int nx_read,
                         const int ny_read,
                         const int nz_read,
                         const int nx,
                         const int ny,
                         const int nz,
                         double* const data_out)
{
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int iz = 0; iz < nz; ++iz)
            {
                data_out[(ix * ny + iy) * nz + iz] =
                    trilinear_reference_value(data_in, nx_read, ny_read, nz_read, nx, ny, nz, ix, iy, iz);
            }
        }
    }
}

std::vector<double> make_interp_input(const int nx, const int ny, const int nz)
{
    std::vector<double> data(nx * ny * nz, 0.0);
    for (int ix = 0; ix < nx; ++ix)
    {
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int iz = 0; iz < nz; ++iz)
            {
                const double x = static_cast<double>(ix + 1);
                const double y = static_cast<double>(iy + 2);
                const double z = static_cast<double>(iz + 3);
                data[(ix * ny + iy) * nz + iz] =
                    std::sin(0.17 * x) + std::cos(0.23 * y) + 0.5 * std::sin(0.31 * z) + 0.01 * x * y;
            }
        }
    }
    return data;
}

double max_abs_error(const std::vector<double>& lhs, const std::vector<double>& rhs)
{
    double max_err = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        max_err = std::max(max_err, std::abs(lhs[i] - rhs[i]));
    }
    return max_err;
}
} // namespace

/**
 * - Tested Functions:
 *   - read_rho()
 *     - the function to read_rho from file
 *     - the serial version without MPI
 *   - trilinear_interpolate()
 *     - the trilinear interpolation method
 *     - the serial version without MPI
 */

class RhoIOTest : public ::testing::Test
{
  protected:
    int nspin = 1;
    int nrxx = 36 * 36 * 36;
    int prenspin = 1;
    double** rho;
    UnitCell* ucell;

    int my_rank = 0;
    std::ofstream ofs_running = std::ofstream("unittest.log");

    void SetUp()
    {
        rho = new double*[nspin];
        ucell = new UnitCell;
        for (int is = 0; is < nspin; ++is)
        {
            rho[is] = new double[nrxx];
        }
    }
    void TearDown()
    {
        for (int is = 0; is < nspin; ++is)
        {
            delete[] rho[is];
        }
        delete[] rho;
        delete ucell;
    }
};

TEST_F(RhoIOTest, Read)
{
    int is = 0;
    std::string fn = "./support/chg.cube";
    int nx = 36;
    int ny = 36;
    int nz = 36;
    double ef;
    UcellTestPrepare utp = UcellTestLib["Si"];
    ucell = utp.SetUcellInfo();
    Parallel_Grid pgrid(nx, ny, nz, nz, nrxx, nz, 1);
    ModuleIO::read_vdata_palgrid(pgrid, my_rank, ofs_running, fn, rho[is], ucell->nat);
    EXPECT_DOUBLE_EQ(rho[0][0], 1.27020863940e-03);
    EXPECT_DOUBLE_EQ(rho[0][46655], 1.33581335706e-02);
}

TEST_F(RhoIOTest, Write)
{
    int nx = 36;
    int ny = 36;
    int nz = 36;
    UcellTestPrepare utp = UcellTestLib["Si"];
    ucell = utp.SetUcellInfo();
    ucell->lat0 = 10.2;
    ucell->latvec = { -0.5,0,0.5,0,0.5,0.5,-0.5,0.5,0 };
    ucell->atoms[0].tau[0] = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    ucell->atoms[0].tau[1] = ModuleBase::Vector3<double>(-0.75, 0.75, 0.75);
    ucell->atoms[0].ncpp.zv = 4;
    Parallel_Grid pgrid(nx, ny, nz, nz, nrxx, nz, 1);
    ModuleIO::read_vdata_palgrid(pgrid, my_rank, ofs_running, "support/chg.cube", rho[0], ucell->nat);
    ModuleIO::write_vdata_palgrid(pgrid, rho[0], 0, nspin, 0, "test_write_vdata_palgrid.cube", 0.461002, ucell, 11, 1, false, false);
    std::ifstream ifs1("test_write_vdata_palgrid.cube", std::ifstream::binary | std::ifstream::ate);
    std::ifstream ifs2("support/chg.cube", std::ifstream::binary | std::ifstream::ate);
    EXPECT_EQ(ifs1.tellg(), ifs2.tellg());
    ifs1.close();
    ifs2.close();
}

TEST_F(RhoIOTest, TrilinearInterpolate)
{
    int nx = 36;
    int ny = 40;
    int nz = 44;
    int nx_read = 36;
    int ny_read = 36;
    int nz_read = 36;
    std::ifstream ifs("./support/chg.cube");
    for (int i = 0; i < 8; ++i)
    {
        ifs.ignore(300, '\n');
    }
    std::vector<double> data_read(nx_read * ny_read * nz_read);
    for (int ix = 0; ix < nx_read; ix++)
    {
        for (int iy = 0; iy < ny_read; iy++)
        {
            for (int iz = 0; iz < nz_read; iz++)
            {
                ifs >> data_read[(ix * ny_read + iy) * nz_read + iz];
            }
        }
    }

    const int nxyz = nx * ny * nz;
    std::vector<double> actual(nxyz, 0.0);
    std::vector<double> reference(nxyz, 0.0);
    ModuleIO::trilinear_interpolate(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, actual.data());
    trilinear_reference(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, reference.data());

    const double max_err = max_abs_error(actual, reference);
    EXPECT_LT(max_err, 1e-6);
}

TEST_F(RhoIOTest, TrilinearInterpolateConstantFieldBoundary)
{
    const int nx_read = 1;
    const int ny_read = 2;
    const int nz_read = 3;
    const int nx = 7;
    const int ny = 5;
    const int nz = 4;
    const double constant = 3.141592653589793;
    std::vector<double> data_read(nx_read * ny_read * nz_read, constant);
    std::vector<double> actual(nx * ny * nz, 0.0);

    ModuleIO::trilinear_interpolate(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, actual.data());

    for (double value : actual)
    {
        EXPECT_NEAR(value, constant, 1e-12);
    }
}

TEST_F(RhoIOTest, TrilinearInterpolateDifferentGridSizes)
{
    const std::vector<std::tuple<int, int, int, int, int, int>> cases = {
        {2, 2, 2, 5, 4, 3},
        {3, 4, 5, 6, 7, 8},
        {4, 5, 6, 3, 4, 5},
        {7, 5, 3, 9, 6, 4}
    };

    for (const auto& dims : cases)
    {
        const int nx_read = std::get<0>(dims);
        const int ny_read = std::get<1>(dims);
        const int nz_read = std::get<2>(dims);
        const int nx = std::get<3>(dims);
        const int ny = std::get<4>(dims);
        const int nz = std::get<5>(dims);

        const std::vector<double> data_read = make_interp_input(nx_read, ny_read, nz_read);
        std::vector<double> actual(nx * ny * nz, 0.0);
        std::vector<double> reference(nx * ny * nz, 0.0);

        ModuleIO::trilinear_interpolate(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, actual.data());
        trilinear_reference(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, reference.data());

        SCOPED_TRACE(::testing::Message()
                     << "src=(" << nx_read << "," << ny_read << "," << nz_read
                     << "), dst=(" << nx << "," << ny << "," << nz << ")");
        EXPECT_LT(max_abs_error(actual, reference), 1e-6);
    }
}

TEST_F(RhoIOTest, TrilinearInterpolateBoundarySamples)
{
    const int nx_read = 2;
    const int ny_read = 3;
    const int nz_read = 4;
    const int nx = 5;
    const int ny = 4;
    const int nz = 7;

    std::vector<double> data_read(nx_read * ny_read * nz_read, 0.0);
    for (int ix = 0; ix < nx_read; ++ix)
    {
        for (int iy = 0; iy < ny_read; ++iy)
        {
            for (int iz = 0; iz < nz_read; ++iz)
            {
                data_read[(ix * ny_read + iy) * nz_read + iz] = 100.0 * ix + 10.0 * iy + iz;
            }
        }
    }

    std::vector<double> actual(nx * ny * nz, 0.0);
    ModuleIO::trilinear_interpolate(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, actual.data());

    const std::array<std::array<int, 3>, 5> samples = {{
        {{0, 0, 0}},
        {{nx - 1, 0, 0}},
        {{0, ny - 1, 0}},
        {{0, 0, nz - 1}},
        {{nx - 1, ny - 1, nz - 1}}
    }};

    for (const auto& sample : samples)
    {
        const int ix = sample[0];
        const int iy = sample[1];
        const int iz = sample[2];
        const double expected = trilinear_reference_value(data_read.data(), nx_read, ny_read, nz_read, nx, ny, nz, ix, iy, iz);
        const double actual_value = actual[(ix * ny + iy) * nz + iz];
        EXPECT_LT(std::abs(actual_value - expected), 1e-6);
    }
}


struct CubeIOTest : public ::testing::Test
{
    std::vector<std::string> comment;
    int natom = 0;
    std::vector<double> origin;
    std::vector<int> nvoxel;
    int nx_read = 0;
    int ny_read = 0;
    int nz_read = 0;
    std::vector<double> dx;
    std::vector<double> dy;
    std::vector<double> dz;
    std::vector<std::vector<double>> axis_vecs;
    std::vector<int> atom_type;
    std::vector<double> atom_charge;
    std::vector<std::vector<double>> atom_pos;
    std::vector<double> data_read;
    const std::string fn = "./support/chg.cube";
};


TEST_F(CubeIOTest, ReadCube)
{
    ModuleIO::read_cube(fn, comment, natom, origin, 
     nx_read, ny_read, nz_read, 
     dx, dy, dz, 
     atom_type, atom_charge, atom_pos, data_read);
    EXPECT_EQ(comment[0], "Ionic_Step 1  Cubefile created from ABACUS. Inner loop is z, followed by y and x");
    EXPECT_EQ(comment[1], "1 # number of spin directions 0.461002 # Fermi energy, in Ry");
    EXPECT_EQ(natom, 2);
    for (auto& o : origin) { EXPECT_EQ(o, 0.0); }
    EXPECT_EQ(nx_read, 36);
    EXPECT_EQ(ny_read, 36);
    EXPECT_EQ(nz_read, 36);
    EXPECT_DOUBLE_EQ(dx[0], -0.141667);
    EXPECT_DOUBLE_EQ(dy[2], 0.141667);
    EXPECT_DOUBLE_EQ(dz[1], 0.141667);
    EXPECT_EQ(atom_type.size(), natom);
    EXPECT_EQ(atom_charge.size(), natom);
    EXPECT_EQ(atom_pos.size(), natom);
    for (auto& t : atom_type) { EXPECT_EQ(t, 14); }
    for (auto& c : atom_charge) { EXPECT_DOUBLE_EQ(c, 4.0); }
    EXPECT_DOUBLE_EQ(atom_pos[1][1], 7.65);
    const int nxyz = nx_read * ny_read * nz_read;
    EXPECT_EQ(data_read.size(), nxyz);
    EXPECT_EQ(data_read[1], 2.64004483879e-03);
    EXPECT_EQ(data_read[nxyz - 1], 1.33581335706e-02);
}


TEST_F(CubeIOTest, WriteCube)
{
	ModuleIO::read_cube(fn, comment, natom, origin, 
			nx_read, ny_read, nz_read, 
			dx, dy, dz, 
			atom_type, atom_charge, atom_pos, data_read);

	ModuleIO::write_cube("test_write.cube", 
			comment, natom, origin, 
			nx_read, ny_read, nz_read, 
			dx, dy, dz, atom_type, 
			atom_charge, atom_pos, data_read, 11);
    std::ifstream ifs1("test_write.cube", std::ifstream::binary | std::ifstream::ate);
    std::ifstream ifs2("./support/chg.cube", std::ifstream::binary | std::ifstream::ate);
    EXPECT_EQ(ifs1.tellg(), ifs2.tellg());
    ifs1.close();
    ifs2.close();
}
