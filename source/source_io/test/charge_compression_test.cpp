#include "gtest/gtest.h"

#define private public
#include "source_io/module_parameter/parameter.h"
#undef private

#include "source_base/global_variable.h"

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

/**
 * Tested: charge density compression via zlib.
 *
 * The compression API (to be integrated into write_cube / read_rhog):
 *   - compress_charge_data(const double* src, size_t n, std::vector<uint8_t>& dst)
 *   - decompress_charge_data(const uint8_t* src, size_t src_len,
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

// -------------------------------------------------------------------
// Compression / decompression reference implementation (zlib)
// -------------------------------------------------------------------

constexpr uint32_t COMPRESS_MAGIC = 0x504D435A; // "ZCMP" little-endian
constexpr size_t HEADER_SIZE = 12;               // 4B magic + 8B original_count

static bool compress_charge_data(const double* src, size_t n, std::vector<uint8_t>& dst)
{
    dst.clear();
    dst.resize(HEADER_SIZE);

    // write magic
    std::memcpy(dst.data(), &COMPRESS_MAGIC, 4);

    // write original element count
    uint64_t count = static_cast<uint64_t>(n);
    std::memcpy(dst.data() + 4, &count, 8);

    uLongf src_len = static_cast<uLongf>(n * sizeof(double));
    uLongf bound = compressBound(src_len);
    std::vector<uint8_t> cbuf(bound);

    int ret = compress2(cbuf.data(), &bound,
                        reinterpret_cast<const Bytef*>(src), src_len,
                        Z_BEST_COMPRESSION);
    if (ret != Z_OK)
        return false;

    dst.insert(dst.end(), cbuf.begin(), cbuf.begin() + bound);
    return true;
}

static bool decompress_charge_data(const uint8_t* src, size_t src_len,
                                   double* dst, size_t n)
{
    if (src_len < HEADER_SIZE)
        return false;

    uint32_t magic = 0;
    std::memcpy(&magic, src, 4);
    if (magic != COMPRESS_MAGIC)
        return false;

    uint64_t count = 0;
    std::memcpy(&count, src + 4, 8);
    if (count != static_cast<uint64_t>(n))
        return false;

    uLongf dst_len = static_cast<uLongf>(n * sizeof(double));
    int ret = uncompress(reinterpret_cast<Bytef*>(dst), &dst_len,
                         src + HEADER_SIZE, src_len - HEADER_SIZE);
    return ret == Z_OK;
}

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
    ASSERT_TRUE(compress_charge_data(original.data(), n, compressed));
    EXPECT_GT(compressed.size(), HEADER_SIZE);

    std::vector<double> restored(n, -999.0);
    ASSERT_TRUE(decompress_charge_data(compressed.data(), compressed.size(),
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
    ASSERT_TRUE(compress_charge_data(original.data(), n, compressed));

    // all-zero should compress to much less than original size
    size_t raw_size = n * sizeof(double);
    EXPECT_LT(compressed.size(), raw_size / 10)
        << "all-zero data should achieve high compression";

    std::vector<double> restored(n, -1.0);
    ASSERT_TRUE(decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), n));
    for (size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(0.0, restored[i]);
}

TEST_F(ChargeCompressionTest, ConstantDataCompressesWell)
{
    size_t n = 100000;
    auto original = gen_constant(n, 3.14159);

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(compress_charge_data(original.data(), n, compressed));

    size_t raw_size = n * sizeof(double);
    EXPECT_LT(compressed.size(), raw_size / 10)
        << "constant data should achieve high compression";

    std::vector<double> restored(n);
    ASSERT_TRUE(decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), n));
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(original[i], restored[i], kTolerance);
}

TEST_F(ChargeCompressionTest, RandomDataRoundtrip)
{
    size_t n = 50000;
    auto original = gen_random(n, 77);

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(compress_charge_data(original.data(), n, compressed));

    // random data may not compress well, but should still be valid
    std::vector<double> restored(n);
    ASSERT_TRUE(decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), n));
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(original[i], restored[i], kTolerance);
}

TEST_F(ChargeCompressionTest, SmoothGaussianDataRoundtrip)
{
    auto original = gen_smooth_gaussian(20, 20, 20); // n=8000

    std::vector<uint8_t> compressed;
    ASSERT_TRUE(compress_charge_data(original.data(), original.size(), compressed));

    size_t raw_size = original.size() * sizeof(double);
    EXPECT_LT(compressed.size(), raw_size * 2 / 3)
        << "smooth data should achieve moderate compression";

    std::vector<double> restored(original.size());
    ASSERT_TRUE(decompress_charge_data(compressed.data(), compressed.size(),
                                       restored.data(), original.size()));
    for (size_t i = 0; i < original.size(); ++i)
        EXPECT_NEAR(original[i], restored[i], kTolerance);
}

TEST_F(ChargeCompressionTest, WireFormatValidatesMagic)
{
    size_t n = 100;
    auto original = gen_random(n, 1);
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(compress_charge_data(original.data(), n, compressed));

    // check magic
    uint32_t magic = 0;
    std::memcpy(&magic, compressed.data(), 4);
    EXPECT_EQ(magic, COMPRESS_MAGIC);

    // check count field
    uint64_t count = 0;
    std::memcpy(&count, compressed.data() + 4, 8);
    EXPECT_EQ(count, static_cast<uint64_t>(n));
}

TEST_F(ChargeCompressionTest, DecompressRejectsBadMagic)
{
    std::vector<uint8_t> bad(HEADER_SIZE + 10, 0xFF);
    double dst[10];
    EXPECT_FALSE(decompress_charge_data(bad.data(), bad.size(), dst, 10));
}

TEST_F(ChargeCompressionTest, DecompressRejectsWrongCount)
{
    size_t n = 100;
    auto original = gen_random(n, 1);
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(compress_charge_data(original.data(), n, compressed));

    // try to decompress with wrong element count
    std::vector<double> restored(200);
    EXPECT_FALSE(decompress_charge_data(compressed.data(), compressed.size(),
                                        restored.data(), 200));
}

TEST_F(ChargeCompressionTest, RejectsTooSmallBuffer)
{
    size_t n = 100;
    auto original = gen_random(n, 1);
    std::vector<uint8_t> compressed;
    ASSERT_TRUE(compress_charge_data(original.data(), n, compressed));

    // buffer smaller than header
    double small_buf[1];
    EXPECT_FALSE(decompress_charge_data(compressed.data(), 4, small_buf, n));
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
        compress_charge_data(data.data(), data.size(), compressed);
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
        compress_charge_data(data.data(), data.size(), compressed);
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
        compress_charge_data(data.data(), data.size(), compressed);
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
        compress_charge_data(c.data.data(), n, compressed);
        double cmb = static_cast<double>(compressed.size()) / 1048576.0;
        double ratio = cmb / data_mb;
        printf("[BENCH] CompressRatio_%-12s  size=%6.1f MB -> %6.2f MB  ratio=%5.1f%%\n",
               c.name, data_mb, cmb, ratio * 100.0);
    }
}

// === Reserved slots for OpenMP parallel compression benchmarks ===
// TEST_F(ChargeCompressionTest, Bench_Compress_OpenMP_nthreads4) { ... }
// TEST_F(ChargeCompressionTest, Bench_Compress_OpenMP_nthreads8) { ... }

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
