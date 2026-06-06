#include "gtest/gtest.h"

#define private public
#include "source_io/module_parameter/parameter.h"
#undef private

#include "source_base/global_variable.h"
#include "source_io/module_output/charge_compress.h"
#include "source_io/module_output/cube_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <zlib.h>

#ifdef __MPI
#include "source_basis/module_pw/test/test_tool.h"
#include "mpi.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Tested: charge density compression via zlib.
 *
 * The compression API (to be integrated into write_cube / read_rhog):
 *   - ModuleIO::compress_charge_data(const double* src, size_t n, std::vector<uint8_t>& dst)
 *   - ModuleIO::decompress_charge_data(const uint8_t* src, size_t src_len,
 *                             double* dst, size_t n)
 *
 * Wire format: [magic:4B "ZCMP"] [original_count:8B] [compressed payload]
 *
 * Correctness tests:
 *   - Roundtrip fidelity (error < 1e-6)
 *   - Boundary: all-zero, constant, random, smooth-gaussian
 *   - Wire format validation
 *
 * Performance tests:
 *   - Compression ratio vs data pattern
 *   - Time measurements for 64^3, 128^3, 256^3 grids
 *
 * Extension points (after OpenMP compression implemented):
 *   - Bench_Compress_OpenMP_nthreads{N}
 */

// ===========================================================================
// Compression / decompression now provided by ModuleIO::charge_compress
// ===========================================================================

// -------------------------------------------------------------------
// Data generators
// -------------------------------------------------------------------

static std::vector<double> gen_all_zeros(size_t n)
{
    return std::vector<double>(n, 0.0);
}

static std::vector<double> gen_constant(size_t n, double val)
{
    return std::vector<double>(n, val);
}

static std::vector<double> gen_random(size_t n, unsigned seed = 42)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = dist(rng);
    return v;
}

static std::vector<double> gen_smooth_gaussian(size_t nx, size_t ny, size_t nz)
{
    // simulates realistic charge density: Gaussian bumps around "atom" positions
    size_t nxyz = nx * ny * nz;
    std::vector<double> v(nxyz, 0.0);
    double centers[][3] = {{0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}};
    double sigma = 0.08;

    size_t nxy = nx * ny;
    for (size_t ix = 0; ix < nx; ++ix)
    {
        double x = static_cast<double>(ix) / nx;
        for (size_t iy = 0; iy < ny; ++iy)
        {
            double y = static_cast<double>(iy) / ny;
            for (size_t iz = 0; iz < nz; ++iz)
            {
                double z = static_cast<double>(iz) / nz;
                double val = 0.0;
                for (auto& c : centers)
                {
                    double r2 = (x - c[0]) * (x - c[0])
                              + (y - c[1]) * (y - c[1])
                              + (z - c[2]) * (z - c[2]);
                    val += std::exp(-r2 / (2.0 * sigma * sigma));
                }
                v[ix * nxy + iy * nz + iz] = val;
            }
        }
    }
    return v;
}

// -------------------------------------------------------------------
// Test fixture
// -------------------------------------------------------------------

class ChargeCompressionTest : public ::testing::Test
{
  protected:
    static constexpr double kTolerance = 1e-6;
};

// ===================================================================
// Correctness tests
// ===================================================================

TEST_F(ChargeCompressionTest, CompressDecompressRoundtrip_Small)
{
    size_t n = 10000;
    auto original = gen_random(n, 123);

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), n, compressed));
    EXPECT_GT(compressed.size(), ModuleIO::CHARGE_COMPRESS_HEADER_SIZE);

    std::vector<double> restored(n, -999.0);
    ASSERT_TRUE(ModuleIO::decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), n));

    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(original[i], restored[i], kTolerance) << "mismatch at " << i;
    }
}

TEST_F(ChargeCompressionTest, AllZerosCompressesWell)
{
    size_t n = 100000;
    auto original = gen_all_zeros(n);

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), n, compressed));

    // all-zero should compress to much less than original size
    size_t raw_size = n * sizeof(double);
    EXPECT_LT(compressed.size(), raw_size / 10)
        << "all-zero data should achieve high compression";

    std::vector<double> restored(n, -1.0);
    ASSERT_TRUE(ModuleIO::decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), n));
    for (size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(0.0, restored[i]);
}

TEST_F(ChargeCompressionTest, ConstantDataCompressesWell)
{
    size_t n = 100000;
    auto original = gen_constant(n, 3.14159);

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), n, compressed));

    size_t raw_size = n * sizeof(double);
    EXPECT_LT(compressed.size(), raw_size / 10)
        << "constant data should achieve high compression";

    std::vector<double> restored(n);
    ASSERT_TRUE(ModuleIO::decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), n));
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(original[i], restored[i], kTolerance);
}

TEST_F(ChargeCompressionTest, RandomDataRoundtrip)
{
    size_t n = 50000;
    auto original = gen_random(n, 77);

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), n, compressed));

    // random data may not compress well, but should still be valid
    std::vector<double> restored(n);
    ASSERT_TRUE(ModuleIO::decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), n));
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(original[i], restored[i], kTolerance);
}

TEST_F(ChargeCompressionTest, SmoothGaussianDataRoundtrip)
{
    auto original = gen_smooth_gaussian(20, 20, 20); // n=8000

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), original.size(), compressed));

    size_t raw_size = original.size() * sizeof(double);
    EXPECT_LT(compressed.size(), raw_size * 2 / 3)
        << "smooth data should achieve moderate compression";

    std::vector<double> restored(original.size());
    ASSERT_TRUE(ModuleIO::decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), original.size()));
    for (size_t i = 0; i < original.size(); ++i)
        EXPECT_NEAR(original[i], restored[i], kTolerance);
}

TEST_F(ChargeCompressionTest, WireFormatValidatesMagic)
{
    size_t n = 100;
    auto original = gen_random(n, 1);
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), n, compressed));

    // check magic
    uint32_t magic = 0;
    std::memcpy(&magic, compressed.data(), 4);
    EXPECT_EQ(magic, ModuleIO::CHARGE_COMPRESS_MAGIC);

    // check count field
    uint64_t count = 0;
    std::memcpy(&count, compressed.data() + 4, 8);
    EXPECT_EQ(count, static_cast<uint64_t>(n));
}

TEST_F(ChargeCompressionTest, DecompressRejectsBadMagic)
{
    std::vector<uint8_t> bad(ModuleIO::CHARGE_COMPRESS_HEADER_SIZE + 10, 0xFF);
    double dst[10];
    EXPECT_FALSE(ModuleIO::decompress_charge_data(bad.data(), bad.size(), dst, 10));
}

TEST_F(ChargeCompressionTest, DecompressRejectsWrongCount)
{
    size_t n = 100;
    auto original = gen_random(n, 1);
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), n, compressed));

    // try to decompress with wrong element count
    std::vector<double> restored(200);
    EXPECT_FALSE(ModuleIO::decompress_charge_data(compressed.data(), compressed.size(),
                                        restored.data(), 200));
}

TEST_F(ChargeCompressionTest, RejectsTooSmallBuffer)
{
    size_t n = 100;
    auto original = gen_random(n, 1);
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), n, compressed));

    // buffer smaller than header
    double small_buf[1];
    EXPECT_FALSE(ModuleIO::decompress_charge_data(compressed.data(), 4, small_buf, n));
}

// ===================================================================
// Performance benchmarks
// ===================================================================

static void bench_report(const std::string& name, long long time_us, int repeat,
                         double data_mb, double compressed_mb)
{
    double ms = static_cast<double>(time_us) / 1000.0 / repeat;
    double ratio = data_mb > 0.0 ? compressed_mb / data_mb : 1.0;
    double mbps = data_mb / (ms / 1000.0);
    printf("[BENCH] %-38s  compress=%8.2f ms  ratio=%5.1f%%  size=%6.1f MB -> %6.1f MB  throughput=%8.2f MB/s\n",
           name.c_str(), ms, ratio * 100.0, data_mb, compressed_mb, mbps);
}

TEST_F(ChargeCompressionTest, Bench_Compress_Small_64)
{
    auto data = gen_smooth_gaussian(64, 64, 64);
    double data_mb = static_cast<double>(data.size() * sizeof(double)) / 1048576.0;
    int repeat = 10;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> compressed;
    for (int r = 0; r < repeat; ++r)
        ModuleIO::compress_charge_data(data.data(), data.size(), compressed);
    auto t1 = std::chrono::high_resolution_clock::now();
    long long t = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    double cmb = static_cast<double>(compressed.size()) / 1048576.0;
    bench_report("Compress_Smooth_64", t, repeat, data_mb, cmb);
}

TEST_F(ChargeCompressionTest, Bench_Compress_Medium_128)
{
    auto data = gen_smooth_gaussian(128, 128, 128);
    double data_mb = static_cast<double>(data.size() * sizeof(double)) / 1048576.0;
    int repeat = 3;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> compressed;
    for (int r = 0; r < repeat; ++r)
        ModuleIO::compress_charge_data(data.data(), data.size(), compressed);
    auto t1 = std::chrono::high_resolution_clock::now();
    long long t = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    double cmb = static_cast<double>(compressed.size()) / 1048576.0;
    bench_report("Compress_Smooth_128", t, repeat, data_mb, cmb);
}

TEST_F(ChargeCompressionTest, Bench_Compress_Large_256)
{
    auto data = gen_smooth_gaussian(256, 256, 256);
    double data_mb = static_cast<double>(data.size() * sizeof(double)) / 1048576.0;
    int repeat = 1;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> compressed;
    for (int r = 0; r < repeat; ++r)
        ModuleIO::compress_charge_data(data.data(), data.size(), compressed);
    auto t1 = std::chrono::high_resolution_clock::now();
    long long t = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    double cmb = static_cast<double>(compressed.size()) / 1048576.0;
    bench_report("Compress_Smooth_256", t, repeat, data_mb, cmb);
}

TEST_F(ChargeCompressionTest, Bench_CompressRatioByPattern)
{
    // compare compression ratios across different data patterns
    size_t n = 64 * 64 * 64;
    double data_mb = static_cast<double>(n * sizeof(double)) / 1048576.0;

    struct Case {
        const char* name;
        std::vector<double> data;
    };
    std::vector<Case> cases = {
        {"zeros", gen_all_zeros(n)},
        {"constant", gen_constant(n, 1.0)},
        {"smooth", gen_smooth_gaussian(64, 64, 64)},
        {"random", gen_random(n, 99)},
    };

    for (auto& c : cases)
    {
        std::vector<uint8_t> compressed;
        ModuleIO::compress_charge_data(c.data.data(), n, compressed);
        double cmb = static_cast<double>(compressed.size()) / 1048576.0;
        double ratio = cmb / data_mb;
        printf("[BENCH] CompressRatio_%-12s  size=%6.1f MB -> %6.2f MB  ratio=%5.1f%%\n",
               c.name, data_mb, cmb, ratio * 100.0);
    }
}

// ===================================================================
// OpenMP parallel compression tests
// ===================================================================

TEST_F(ChargeCompressionTest, OmpCompressDecompressRoundtrip)
{
    size_t n = 100000;
    auto original = gen_random(n, 456);

    int nthreads = 0;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if (nthreads < 2)
        nthreads = 2;
#else
    nthreads = 1;
#endif

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data_omp(original.data(), n, compressed, nthreads));
    EXPECT_GT(compressed.size(), 0u);

    std::vector<double> restored(n, -999.0);
    ASSERT_TRUE(ModuleIO::decompress_charge_data_omp(compressed.data(), compressed.size(),
                                                     restored.data(), n));

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(original[i], restored[i], kTolerance) << "mismatch at " << i;
}

TEST_F(ChargeCompressionTest, OmpCompressSameAsSerial)
{
    size_t n = 50000;
    auto original = gen_random(n, 789);

    std::vector<uint8_t> c_serial;
    ASSERT_TRUE(ModuleIO::compress_charge_data(original.data(), n, c_serial));

    int nthreads = 0;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
    if (nthreads < 2)
        nthreads = 2;
#else
    nthreads = 1;
#endif

    std::vector<uint8_t> c_omp;
    ASSERT_TRUE(ModuleIO::compress_charge_data_omp(original.data(), n, c_omp, nthreads));

    // Both should decompress to the same data, even if wire formats differ
    std::vector<double> r_serial(n), r_omp(n);
    ASSERT_TRUE(ModuleIO::decompress_charge_data(c_serial.data(), c_serial.size(),
                                                  r_serial.data(), n));
    ASSERT_TRUE(ModuleIO::decompress_charge_data_omp(c_omp.data(), c_omp.size(),
                                                      r_omp.data(), n));

    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(r_serial[i], r_omp[i], kTolerance) << "serial vs OMP mismatch at " << i;
}

TEST_F(ChargeCompressionTest, OmpAllZerosCompressesWell)
{
    size_t n = 100000;
    auto original = gen_all_zeros(n);

    int nthreads = 4;
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(ModuleIO::compress_charge_data_omp(original.data(), n, compressed, nthreads));

    size_t raw_size = n * sizeof(double);
    EXPECT_LT(compressed.size(), raw_size / 10);

    std::vector<double> restored(n, -1.0);
    ASSERT_TRUE(ModuleIO::decompress_charge_data_omp(compressed.data(), compressed.size(),
                                                      restored.data(), n));
    for (size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(0.0, restored[i]);
}

// ===================================================================
// Integration tests with write_cube / read_cube
// ===================================================================

TEST_F(ChargeCompressionTest, WriteCubeWithCompressionRoundtrip)
{
    // Create a small cube file with compression, then read it back
    std::vector<std::string> comment = {"COMPRESSED_ZLIB TestHeader", "Inner loop is z"};
    int natom = 1;
    std::vector<double> origin = {0.0, 0.0, 0.0};
    int nx = 10, ny = 10, nz = 10;
    std::vector<double> dx_v = {0.1, 0.0, 0.0};
    std::vector<double> dy_v = {0.0, 0.1, 0.0};
    std::vector<double> dz_v = {0.0, 0.0, 0.1};
    std::vector<int> atom_type = {1};
    std::vector<double> atom_charge = {1.0};
    std::vector<std::vector<double>> atom_pos = {{0.0, 0.0, 0.0}};

    int nxyz = nx * ny * nz;
    std::vector<double> data(nxyz);
    for (int i = 0; i < nxyz; ++i)
        data[i] = std::sin(static_cast<double>(i) * 0.01);

    std::string fn = "test_compress_writecube.cube";

    // Write with compression
    ModuleIO::write_cube(fn, comment, natom, origin, nx, ny, nz,
                         dx_v, dy_v, dz_v, atom_type, atom_charge, atom_pos,
                         data, 6, 6, true, 0);

    // Read back
    std::vector<std::string> cmt;
    int natom_r = 0;
    std::vector<double> org_r;
    int nx_r = 0, ny_r = 0, nz_r = 0;
    std::vector<double> dx_r(3), dy_r(3), dz_r(3);
    std::vector<int> atype;
    std::vector<double> ac;
    std::vector<std::vector<double>> ap;
    std::vector<double> rdata;

    bool ok = ModuleIO::read_cube(fn, cmt, natom_r, org_r,
                                  nx_r, ny_r, nz_r, dx_r, dy_r, dz_r,
                                  atype, ac, ap, rdata);
    ASSERT_TRUE(ok);

    EXPECT_EQ(natom, natom_r);
    EXPECT_EQ(nx, nx_r);
    EXPECT_EQ(ny, ny_r);
    EXPECT_EQ(nz, nz_r);
    ASSERT_EQ(data.size(), rdata.size());

    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_NEAR(data[i], rdata[i], kTolerance) << "mismatch at " << i;

    std::remove(fn.c_str());
}

TEST_F(ChargeCompressionTest, AutoDetectCompressedFile)
{
    // Write one compressed and one uncompressed file with the same data.
    // read_cube should correctly handle both.
    std::vector<std::string> comment = {"Test", "z is fastest"};
    int natom = 1;
    std::vector<double> origin = {0.0, 0.0, 0.0};
    int nx = 8, ny = 8, nz = 8;
    std::vector<double> dx_v = {1.0/8, 0.0, 0.0};
    std::vector<double> dy_v = {0.0, 1.0/8, 0.0};
    std::vector<double> dz_v = {0.0, 0.0, 1.0/8};
    std::vector<int> atom_type = {8};
    std::vector<double> atom_charge = {6.0};
    std::vector<std::vector<double>> atom_pos = {{0.5, 0.5, 0.5}};

    int nxyz = nx * ny * nz;
    std::vector<double> data(nxyz);
    for (int i = 0; i < nxyz; ++i)
        data[i] = std::exp(-static_cast<double>(i) * 0.001);

    std::string fn_cmp = "test_autodetect_comp.cube";
    std::string fn_txt = "test_autodetect_text.cube";

    // Write compressed
    ModuleIO::write_cube(fn_cmp, comment, natom, origin, nx, ny, nz,
                         dx_v, dy_v, dz_v, atom_type, atom_charge, atom_pos,
                         data, 6, 6, true, 0);

    // Write uncompressed
    ModuleIO::write_cube(fn_txt, comment, natom, origin, nx, ny, nz,
                         dx_v, dy_v, dz_v, atom_type, atom_charge, atom_pos,
                         data, 6, 6, false, 0);

    // Read compressed
    std::vector<double> rdata_cmp;
    {
        std::vector<std::string> cmt;
        int nr = 0, nxr = 0, nyr = 0, nzr = 0;
        std::vector<double> org_r, dxr(3), dyr(3), dzr(3);
        std::vector<int> at;
        std::vector<double> ac;
        std::vector<std::vector<double>> ap;
        ASSERT_TRUE(ModuleIO::read_cube(fn_cmp, cmt, nr, org_r, nxr, nyr, nzr,
                                         dxr, dyr, dzr, at, ac, ap, rdata_cmp));
    }

    // Read uncompressed
    std::vector<double> rdata_txt;
    {
        std::vector<std::string> cmt;
        int nr = 0, nxr = 0, nyr = 0, nzr = 0;
        std::vector<double> org_r, dxr(3), dyr(3), dzr(3);
        std::vector<int> at;
        std::vector<double> ac;
        std::vector<std::vector<double>> ap;
        ASSERT_TRUE(ModuleIO::read_cube(fn_txt, cmt, nr, org_r, nxr, nyr, nzr,
                                         dxr, dyr, dzr, at, ac, ap, rdata_txt));
    }

    ASSERT_EQ(data.size(), rdata_cmp.size());
    ASSERT_EQ(data.size(), rdata_txt.size());

    for (size_t i = 0; i < data.size(); ++i)
    {
        EXPECT_NEAR(data[i], rdata_cmp[i], kTolerance);
        EXPECT_NEAR(data[i], rdata_txt[i], kTolerance);
    }

    std::remove(fn_cmp.c_str());
    std::remove(fn_txt.c_str());
}

// ===================================================================
// OpenMP parallel compression benchmarks
// ===================================================================

TEST_F(ChargeCompressionTest, Bench_Compress_OMP_nthreads4_256)
{
#ifdef _OPENMP
    auto data = gen_smooth_gaussian(256, 256, 256);
    double data_mb = static_cast<double>(data.size() * sizeof(double)) / 1048576.0;
    int repeat = 1;
    int nthreads = 4;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> compressed;
    for (int r = 0; r < repeat; ++r)
        ModuleIO::compress_charge_data_omp(data.data(), data.size(), compressed, nthreads);
    auto t1 = std::chrono::high_resolution_clock::now();
    long long t = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    double cmb = static_cast<double>(compressed.size()) / 1048576.0;
    double ms = static_cast<double>(t) / 1000.0 / repeat;
    double mbps = data_mb / (ms / 1000.0);
    printf("[BENCH] Compress_OMP_256_nthreads%-2d   time=%8.2f ms  throughput=%8.2f MB/s  size=%6.1f MB -> %6.1f MB\n",
           nthreads, ms, mbps, data_mb, cmb);
#else
    GTEST_SKIP() << "OpenMP not available";
#endif
}

TEST_F(ChargeCompressionTest, Bench_Compress_OMP_Scaling_128)
{
#ifdef _OPENMP
    auto data = gen_smooth_gaussian(128, 128, 128);
    double data_mb = static_cast<double>(data.size() * sizeof(double)) / 1048576.0;
    int repeat = 3;

    for (int nthreads : {1, 2, 4, 8})
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> compressed;
        for (int r = 0; r < repeat; ++r)
            ModuleIO::compress_charge_data_omp(data.data(), data.size(), compressed, nthreads);
        auto t1 = std::chrono::high_resolution_clock::now();
        long long t = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        double ms = static_cast<double>(t) / 1000.0 / repeat;
        double mbps = data_mb / (ms / 1000.0);
        printf("[BENCH] Compress_OMP_128_nthreads%-2d  time=%8.2f ms  throughput=%8.2f MB/s\n",
               nthreads, ms, mbps);
    }
#else
    GTEST_SKIP() << "OpenMP not available";
#endif
}

int main(int argc, char** argv)
{
#ifdef __MPI
    setupmpi(argc, argv, GlobalV::NPROC, GlobalV::MY_RANK);
    divide_pools(GlobalV::NPROC, GlobalV::MY_RANK, GlobalV::NPROC_IN_POOL,
                 GlobalV::KPAR, GlobalV::MY_POOL, GlobalV::RANK_IN_POOL);
#endif

    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

#ifdef __MPI
    finishmpi();
#endif
    return result;
}
