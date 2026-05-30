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
