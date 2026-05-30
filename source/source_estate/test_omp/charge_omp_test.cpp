// =============================================================================
// Task 2 (homework/task127/task2) - Stage 2 skeleton
//
// Goal of this file at Stage 2:
//   - Compile against ../module_charge/charge_mpi.cpp with _OPENMP enabled
//   - Provide a valid GTest + MPI entry point
//   - Carry exactly one trivial sanity test so ctest reports a real result
//
// Real OpenMP correctness / performance tests for
//   reorder_pool_rank_to_uniform and extract_uniform_to_local
// will be added in Stage 6.
// =============================================================================

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define private public
#include "source_base/matrix3.h"
#include "source_base/parallel_global.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_charge/charge.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/parameter.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

bool XC_Functional::ked_flag = false;

Charge::Charge() {}
Charge::~Charge()
{
    delete[] rec;
    delete[] dis;
    delete[] chgmpi_tmp_;
    delete[] chgmpi_tot_;
    delete[] chgmpi_tot_aux_;
}

TEST(ChargeOmpSkeleton, OpenMPMacroEnabled)
{
#ifdef _OPENMP
    SUCCEED() << "_OPENMP is defined; omp_get_max_threads = "
              << omp_get_max_threads();
#else
    FAIL() << "_OPENMP is NOT defined for MODULE_ESTATE_charge_omp_test. "
              "Check source/source_estate/test_omp/CMakeLists.txt.";
#endif
}

// =============================================================================
// Stage 4: Correctness test for OpenMP-parallelized reorder_pool_rank_to_uniform
//
// Strategy:
//   - Build a minimal PW_Basis stub by directly assigning nx, ny, nz, numz[],
//     startz[] (bypassing initgrids/setuptransform).
//   - Construct a deterministic input array_tot using std::mt19937 (thread-safe
//     because generation is sequential, not parallel).
//   - Run reorder_pool_rank_to_uniform once and check the result against a
//     hand-written, single-threaded reference implementation.
//   - The function under test internally uses #pragma omp parallel for; if any
//     write-write race or wrong indexing existed, EXPECT_EQ would fail.
// =============================================================================

namespace
{

// A self-contained serial reference that mirrors the *original* (pre-OpenMP)
// loop body of reorder_pool_rank_to_uniform. Used as ground truth.
void serial_reorder_reference(const double* array_tot,
                              double* array_tot_aux,
                              int ncxy,
                              int nz,
                              int numz_ip,
                              int startz_ip)
{
    for (int ir = 0; ir < ncxy; ++ir)
    {
        for (int iz = 0; iz < numz_ip; ++iz)
        {
            array_tot_aux[nz * ir + startz_ip + iz]
                = array_tot[numz_ip * ir + startz_ip * ncxy + iz];
        }
    }
}

// Build a PW_Basis with manually-populated rank layout. nproc ranks split nz
// planes; rank ip owns layout_numz[ip] planes starting at layout_startz[ip].
ModulePW::PW_Basis* make_stub_pw_basis(int nx,
                                       int ny,
                                       int nz,
                                       const std::vector<int>& layout_numz,
                                       const std::vector<int>& layout_startz)
{
    auto* pw = new ModulePW::PW_Basis();
    pw->nx = nx;
    pw->ny = ny;
    pw->nz = nz;
    pw->nxy = nx * ny;
    pw->nxyz = nx * ny * nz;
    const int nproc = static_cast<int>(layout_numz.size());
    pw->numz = new int[nproc];
    pw->startz = new int[nproc];
    for (int ip = 0; ip < nproc; ++ip)
    {
        pw->numz[ip] = layout_numz[ip];
        pw->startz[ip] = layout_startz[ip];
    }
    return pw;
}

void destroy_stub_pw_basis(ModulePW::PW_Basis* pw)
{
    delete[] pw->numz;
    pw->numz = nullptr;
    delete[] pw->startz;
    pw->startz = nullptr;
    delete pw;
}

} // namespace

TEST(ChargeOmpReorder, MatchesSerialReferenceUniformSplit)
{
    const int nx = 4;
    const int ny = 5;
    const int nz = 8;
    const std::vector<int> layout_numz = {2, 2, 2, 2};
    const std::vector<int> layout_startz = {0, 2, 4, 6};
    ASSERT_EQ(layout_numz.size(), layout_startz.size());

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);

    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(2026u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    std::vector<double> out_omp(nxyz, 0.0);
    std::vector<double> out_ref(nxyz, 0.0);

    for (int ip = 0; ip < static_cast<int>(layout_numz.size()); ++ip)
    {
        charge.reorder_pool_rank_to_uniform(array_tot.data(), out_omp.data(), ip);
        serial_reorder_reference(array_tot.data(),
                                 out_ref.data(),
                                 ncxy,
                                 nz,
                                 layout_numz[ip],
                                 layout_startz[ip]);
    }

    for (int i = 0; i < nxyz; ++i)
    {
        ASSERT_EQ(out_omp[i], out_ref[i]) << "mismatch at index " << i;
    }

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}

TEST(ChargeOmpReorder, MatchesSerialReferenceUnevenSplit)
{
    const int nx = 3;
    const int ny = 3;
    const int nz = 7;
    const std::vector<int> layout_numz = {3, 1, 0, 3};
    const std::vector<int> layout_startz = {0, 3, 4, 4};
    ASSERT_EQ(layout_numz.size(), layout_startz.size());

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);

    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(42u);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    std::vector<double> out_omp(nxyz, 0.0);
    std::vector<double> out_ref(nxyz, 0.0);
    for (int ip = 0; ip < static_cast<int>(layout_numz.size()); ++ip)
    {
        charge.reorder_pool_rank_to_uniform(array_tot.data(), out_omp.data(), ip);
        serial_reorder_reference(array_tot.data(),
                                 out_ref.data(),
                                 ncxy,
                                 nz,
                                 layout_numz[ip],
                                 layout_startz[ip]);
    }

    for (int i = 0; i < nxyz; ++i)
    {
        ASSERT_EQ(out_omp[i], out_ref[i]) << "mismatch at index " << i;
    }

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}

// =============================================================================
// Stage 5: Correctness test for OpenMP-parallelized extract_uniform_to_local
//
// Strategy:
//   - Same PW_Basis stub idea as Stage 4, plus setting startz_current and
//     GlobalV::RANK_IN_POOL so the function under test picks the right slab.
//   - Compare OpenMP output vs. a hand-written serial reference, byte-for-byte.
// =============================================================================

namespace
{

void serial_extract_reference(const double* array_tot,
                              double* array_rho,
                              int ncxy,
                              int nz,
                              int numz_local,
                              int startz_local)
{
    for (int ir = 0; ir < ncxy; ++ir)
    {
        for (int iz = 0; iz < numz_local; ++iz)
        {
            array_rho[numz_local * ir + iz]
                = array_tot[nz * ir + startz_local + iz];
        }
    }
}

} // namespace

TEST(ChargeOmpExtract, MatchesSerialReferenceForEachRank)
{
    const int nx = 4;
    const int ny = 5;
    const int nz = 8;
    const std::vector<int> layout_numz = {3, 1, 0, 4};
    const std::vector<int> layout_startz = {0, 3, 4, 4};
    ASSERT_EQ(layout_numz.size(), layout_startz.size());

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);

    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(7u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    const int saved_rank = GlobalV::RANK_IN_POOL;
    for (int ip = 0; ip < static_cast<int>(layout_numz.size()); ++ip)
    {
        const int numz_local = layout_numz[ip];
        const int startz_local = layout_startz[ip];
        const int local_size = std::max(1, numz_local * ncxy);

        GlobalV::RANK_IN_POOL = ip;
        pw->startz_current = startz_local;

        std::vector<double> rho_omp(local_size, 0.0);
        std::vector<double> rho_ref(local_size, 0.0);

        charge.extract_uniform_to_local(array_tot.data(), rho_omp.data());
        serial_extract_reference(array_tot.data(),
                                 rho_ref.data(),
                                 ncxy,
                                 nz,
                                 numz_local,
                                 startz_local);

        for (int i = 0; i < numz_local * ncxy; ++i)
        {
            ASSERT_EQ(rho_omp[i], rho_ref[i])
                << "rank " << ip << " mismatch at index " << i;
        }
    }
    GlobalV::RANK_IN_POOL = saved_rank;

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}

// =============================================================================
// Stage 6: Boundary tests required by problem statement
//   - ncxy = 1                (degenerate xy plane)
//   - numz contains 0         (rank holds zero z-planes)
//   - single-rank pool        (nproc = 1)
//   - thread-count invariance (results stable across OMP_NUM_THREADS at runtime)
// =============================================================================

TEST(ChargeOmpBoundary, ReorderNcxyOne)
{
    const int nx = 1;
    const int ny = 1;
    const int nz = 6;
    const std::vector<int> layout_numz = {2, 2, 1, 1};
    const std::vector<int> layout_startz = {0, 2, 4, 5};

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);
    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    ASSERT_EQ(ncxy, 1);
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(101u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    std::vector<double> out_omp(nxyz, 0.0);
    std::vector<double> out_ref(nxyz, 0.0);
    for (int ip = 0; ip < static_cast<int>(layout_numz.size()); ++ip)
    {
        charge.reorder_pool_rank_to_uniform(array_tot.data(), out_omp.data(), ip);
        serial_reorder_reference(array_tot.data(),
                                 out_ref.data(),
                                 ncxy,
                                 nz,
                                 layout_numz[ip],
                                 layout_startz[ip]);
    }
    for (int i = 0; i < nxyz; ++i)
    {
        ASSERT_EQ(out_omp[i], out_ref[i]) << "ncxy=1 reorder mismatch at " << i;
    }

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}

TEST(ChargeOmpBoundary, ReorderWithZeroSlabRank)
{
    // numz contains both 0 and 1; covers the numz=[0,1]-style scenario the
    // problem statement asks for, mixed with two full-sized ranks.
    const int nx = 2;
    const int ny = 3;
    const int nz = 5;
    const std::vector<int> layout_numz = {0, 1, 2, 2};
    const std::vector<int> layout_startz = {0, 0, 1, 3};

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);
    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(202u);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    std::vector<double> out_omp(nxyz, 0.0);
    std::vector<double> out_ref(nxyz, 0.0);
    for (int ip = 0; ip < static_cast<int>(layout_numz.size()); ++ip)
    {
        // numz[ip] = 0 means: inner loop runs zero iterations. The function
        // must not crash and must not write anywhere.
        charge.reorder_pool_rank_to_uniform(array_tot.data(), out_omp.data(), ip);
        serial_reorder_reference(array_tot.data(),
                                 out_ref.data(),
                                 ncxy,
                                 nz,
                                 layout_numz[ip],
                                 layout_startz[ip]);
    }
    for (int i = 0; i < nxyz; ++i)
    {
        ASSERT_EQ(out_omp[i], out_ref[i]) << "zero-slab mismatch at " << i;
    }

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}

TEST(ChargeOmpBoundary, ReorderSingleRankPool)
{
    // nproc = 1: one rank owns the whole z range.
    const int nx = 3;
    const int ny = 4;
    const int nz = 5;
    const std::vector<int> layout_numz = {5};
    const std::vector<int> layout_startz = {0};

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);
    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(303u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    std::vector<double> out_omp(nxyz, 0.0);
    std::vector<double> out_ref(nxyz, 0.0);
    charge.reorder_pool_rank_to_uniform(array_tot.data(), out_omp.data(), 0);
    serial_reorder_reference(array_tot.data(),
                             out_ref.data(),
                             ncxy,
                             nz,
                             layout_numz[0],
                             layout_startz[0]);
    for (int i = 0; i < nxyz; ++i)
    {
        ASSERT_EQ(out_omp[i], out_ref[i]) << "single-rank mismatch at " << i;
    }

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}

TEST(ChargeOmpBoundary, ExtractNcxyOne)
{
    const int nx = 1;
    const int ny = 1;
    const int nz = 6;
    const std::vector<int> layout_numz = {2, 2, 1, 1};
    const std::vector<int> layout_startz = {0, 2, 4, 5};

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);
    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    ASSERT_EQ(ncxy, 1);
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(404u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    const int saved_rank = GlobalV::RANK_IN_POOL;
    for (int ip = 0; ip < static_cast<int>(layout_numz.size()); ++ip)
    {
        const int numz_local = layout_numz[ip];
        const int startz_local = layout_startz[ip];
        const int local_size = std::max(1, numz_local * ncxy);

        GlobalV::RANK_IN_POOL = ip;
        pw->startz_current = startz_local;

        std::vector<double> rho_omp(local_size, 0.0);
        std::vector<double> rho_ref(local_size, 0.0);
        charge.extract_uniform_to_local(array_tot.data(), rho_omp.data());
        serial_extract_reference(array_tot.data(),
                                 rho_ref.data(),
                                 ncxy,
                                 nz,
                                 numz_local,
                                 startz_local);

        for (int i = 0; i < numz_local * ncxy; ++i)
        {
            ASSERT_EQ(rho_omp[i], rho_ref[i])
                << "ncxy=1 extract rank " << ip << " mismatch at " << i;
        }
    }
    GlobalV::RANK_IN_POOL = saved_rank;

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}

#ifdef _OPENMP
TEST(ChargeOmpBoundary, ReorderThreadCountInvariance)
{
    // Same input must yield byte-identical output at runtime thread counts
    // 1, 2 and 4 (assuming the host has at least 2 logical cores; if not,
    // omp_set_num_threads silently caps at the platform max, still safe).
    const int nx = 6;
    const int ny = 6;
    const int nz = 9;
    const std::vector<int> layout_numz = {3, 2, 0, 4};
    const std::vector<int> layout_startz = {0, 3, 5, 5};

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);
    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(505u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    std::vector<double> out_t1(nxyz, 0.0);
    std::vector<double> out_t2(nxyz, 0.0);
    std::vector<double> out_t4(nxyz, 0.0);

    const int saved_threads = omp_get_max_threads();

    auto run_once = [&](std::vector<double>& out, int nthreads) {
        omp_set_num_threads(nthreads);
        std::fill(out.begin(), out.end(), 0.0);
        for (int ip = 0; ip < static_cast<int>(layout_numz.size()); ++ip)
        {
            charge.reorder_pool_rank_to_uniform(array_tot.data(), out.data(), ip);
        }
    };

    run_once(out_t1, 1);
    run_once(out_t2, 2);
    run_once(out_t4, 4);

    for (int i = 0; i < nxyz; ++i)
    {
        ASSERT_EQ(out_t1[i], out_t2[i]) << "t1 vs t2 mismatch at " << i;
        ASSERT_EQ(out_t1[i], out_t4[i]) << "t1 vs t4 mismatch at " << i;
    }

    omp_set_num_threads(saved_threads);

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}

// =============================================================================
// Stage 7: Performance benchmark (gated by CHARGE_OMP_BENCH=1 env var).
// Reports wall-clock time over OMP_NUM_THREADS = 1, 2, 4, 8 for both
// reorder_pool_rank_to_uniform and extract_uniform_to_local on a large grid,
// so the user can read off speedup and fit Amdahl's law in the report.
//
// We skip by default so the regular ctest invocation stays fast.
// =============================================================================
TEST(ChargeOmpPerf, ReorderAndExtractSpeedup)
{
    const char* env = std::getenv("CHARGE_OMP_BENCH");
    if (env == nullptr || std::string(env) != "1")
    {
        GTEST_SKIP() << "Set CHARGE_OMP_BENCH=1 to enable the perf benchmark.";
    }

    const int nx = 128;
    const int ny = 128;
    const int nz = 128;
    const int nproc = 4;
    const int slab = nz / nproc; // 32
    std::vector<int> layout_numz(nproc, slab);
    std::vector<int> layout_startz(nproc, 0);
    for (int ip = 1; ip < nproc; ++ip)
    {
        layout_startz[ip] = layout_startz[ip - 1] + layout_numz[ip - 1];
    }

    auto* pw = make_stub_pw_basis(nx, ny, nz, layout_numz, layout_startz);
    Charge charge;
    charge.rhopw = pw;

    const int ncxy = nx * ny;
    const int nxyz = ncxy * nz;
    std::vector<double> array_tot(nxyz);
    std::mt19937 rng(909u);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < nxyz; ++i)
    {
        array_tot[i] = dist(rng);
    }

    std::vector<double> out(nxyz, 0.0);

    const int saved_threads = omp_get_max_threads();
    const int saved_rank = GlobalV::RANK_IN_POOL;

    const int thread_counts[] = {1, 2, 4, 8};
    const int repeats = 5;

    std::printf("\n[ChargeOmpPerf] grid=%dx%dx%d (nxyz=%d), nproc=%d, repeats=%d\n",
                nx, ny, nz, nxyz, nproc, repeats);
    std::printf("[ChargeOmpPerf] %-8s %-18s %-18s\n",
                "threads", "reorder_ms_avg", "extract_ms_avg");

    for (int t : thread_counts)
    {
        omp_set_num_threads(t);

        // warm-up to avoid the first-touch / thread-pool-spawn artifact
        for (int ip = 0; ip < nproc; ++ip)
        {
            charge.reorder_pool_rank_to_uniform(array_tot.data(), out.data(), ip);
        }

        double reorder_ms = 0.0;
        for (int r = 0; r < repeats; ++r)
        {
            auto t0 = std::chrono::steady_clock::now();
            for (int ip = 0; ip < nproc; ++ip)
            {
                charge.reorder_pool_rank_to_uniform(array_tot.data(), out.data(), ip);
            }
            auto t1 = std::chrono::steady_clock::now();
            reorder_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        reorder_ms /= repeats;

        double extract_ms = 0.0;
        for (int r = 0; r < repeats; ++r)
        {
            auto t0 = std::chrono::steady_clock::now();
            for (int ip = 0; ip < nproc; ++ip)
            {
                GlobalV::RANK_IN_POOL = ip;
                pw->startz_current = layout_startz[ip];
                std::vector<double> rho(layout_numz[ip] * ncxy, 0.0);
                charge.extract_uniform_to_local(array_tot.data(), rho.data());
            }
            auto t1 = std::chrono::steady_clock::now();
            extract_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        extract_ms /= repeats;

        std::printf("[ChargeOmpPerf] %-8d %-18.3f %-18.3f\n",
                    t, reorder_ms, extract_ms);
    }

    omp_set_num_threads(saved_threads);
    GlobalV::RANK_IN_POOL = saved_rank;

    charge.rhopw = nullptr;
    destroy_stub_pw_basis(pw);
}
#endif

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &GlobalV::NPROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalV::MY_RANK);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
