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
#include "source_estate/module_charge/charge.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/parameter.h"

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
