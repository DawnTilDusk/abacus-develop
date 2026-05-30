# =============================================================================
# Task 2 (homework/task127/task2): OpenMP unit-test target for charge_mpi.cpp.
#
# This directory mirrors test_mpi/ but intentionally keeps -D_OPENMP enabled,
# so that OpenMP pragmas inside reorder_pool_rank_to_uniform and
# extract_uniform_to_local become active when this target is built.
#
# Task 1's MODULE_ESTATE_charge_mpi_test target lives in ../test_mpi/ and is
# left untouched.
# =============================================================================

remove_definitions(-D__EXX)
remove_definitions(-D__CUDA)
remove_definitions(-D__UT_USE_CUDA)
remove_definitions(-D__UT_USE_ROCM)
remove_definitions(-D__ROCM)
remove_definitions(-D__MLALGO)

AddTest(
  TARGET MODULE_ESTATE_charge_omp_test
  LIBS parameter ${math_libs} psi base device planewave
  SOURCES charge_omp_test.cpp ../module_charge/charge_mpi.cpp
)

add_test(NAME MODULE_ESTATE_charge_omp_test_4np
      COMMAND mpirun -np 4 ./MODULE_ESTATE_charge_omp_test;
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
