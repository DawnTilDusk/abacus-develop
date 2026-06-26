//==========================================================
// AUTHOR : mohan
// DATE : 2008-11-10
//==========================================================

#include "source_main/driver.h"
#include "fftw3.h"
#include "source_base/parallel_global.h"
#include "source_io/parse_args.h"
#include "source_io/module_parameter/parameter.h"
#include "source_main/version.h"
#include "source_io/module_async_io/async_io_manager.h"

// ============================================================================
// OMP task 异步 I/O (OpenMP 4.0 task+depend, 无 std::thread)
// 用 #pragma omp parallel + #pragma omp single + #pragma omp task
// 替代独立 std::thread 工作线程池。通过 -DUSE_OMP_TASK_ASYNC 启用。
// ============================================================================
#ifdef USE_OMP_TASK_ASYNC
#include "source_io/module_async_io/omp_task/omp_task_manager.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ctime>
#include <iostream>

namespace
{
void print_welcome_banner()
{
#ifdef VERSION
    const char* version = VERSION;
#else
    const char* version = "unknown";
#endif
#ifdef COMMIT_INFO
#include "commit.h"
    const char* commit = COMMIT;
#else
    const char* commit = "unknown";
#endif
    std::cout << "                                                                                     "
              << std::endl
              << "                              ABACUS " << version << std::endl
              << std::endl
              << "               Atomic-orbital Based Ab-initio Computation at UStc                    "
              << std::endl
              << std::endl
              << "                     Website: http://abacus.ustc.edu.cn/                             "
              << std::endl
              << "               Documentation: https://abacus.deepmodeling.com/                       "
              << std::endl
              << "                  Repository: https://github.com/abacusmodeling/abacus-develop       "
              << std::endl
              << "                              https://github.com/deepmodeling/abacus-develop         "
              << std::endl
              << "                      Commit: " << commit << std::endl
              << std::endl;
    time_t time_now = time(nullptr);
    std::cout << " " << ctime(&time_now);
}
} // namespace

int main(int argc, char** argv)
{
    /*
    read the arguement in the command-line,
    with "abacus -v", the program exit and returns version info,
    with no arguments, the program continues.
    */
    ModuleIO::parse_args(argc, argv);

    /*
    read the mpi parameters in the command-line,
    initialize the mpi environment.
    */
    int nproc = 1;
    int my_rank = 0;
    int nthread_per_proc = 1;
    Parallel_Global::read_pal_param(argc, argv, nproc, nthread_per_proc, my_rank);
    if (my_rank == 0)
    {
        print_welcome_banner();
    }
#ifdef _OPENMP
    // ref: https://www.fftw.org/fftw3_doc/Usage-of-Multi_002dthreaded-FFTW.html
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif
    PARAM.set_pal_param(my_rank, nproc, nthread_per_proc);

    /*
    main program for doing electronic structure calculations.
    */

    // ================================================================
    // 启动异步 I/O 管理器
    // ================================================================
#ifdef USE_OMP_TASK_ASYNC
    // ── OpenMP 4.0 task+depend 路径 ──
    // 用 #pragma omp parallel + single 包裹主计算,
    // 在 single 区域内 spawn I/O worker task 链,
    // 然后执行 DD.init() (SCF 循环)。
    // parallel 区域结束时的隐式 taskwait 保证所有 I/O 完成。
    {
        OMPTaskManager& omp_mgr = OMPTaskManager::instance();

        // 允许嵌套并行: 外层 parallel (I/O workers + 主计算)
        // 内层 parallel for (DD.init() 内部的计算)
        omp_set_max_active_levels(2);

        // 预留 I/O worker 线程数 + 主计算线程
        // 例如 4 workers → 请求 5 个外层线程
        const int num_workers = 4;
        omp_mgr.init(num_workers);

        #pragma omp parallel num_threads(num_workers + 1)
        #pragma omp single
        {
            // Phase A: spawn I/O worker 递归链 (立即返回)
            omp_mgr.spawn_io_workers();

            // Phase B: 主计算 (master 线程执行)
            Driver DD;
            DD.init();

            // Phase C: 通知 I/O workers 可以退出了
            omp_mgr.signal_done();
        }
        // 隐式 taskwait / barrier:
        //   所有 I/O task 链完成 + 主计算完成 → parallel 区域结束
        //   不需要显式 wait_all() / stop()
    }
#else
    // ── 原有 std::thread 路径 ──
    AsyncIOManager::instance().start(4);

    Driver DD;
    DD.init();

    /*
    After running mpi version of abacus, release the mpi resources.
    */

    // ================================================================
    // 等待所有异步 I/O 任务完成，然后停止 I/O 工作线程
    // ================================================================
    AsyncIOManager::instance().wait_all();
    AsyncIOManager::instance().stop();
#endif

#ifdef __MPI
    Parallel_Global::finalize_mpi();
#endif
#ifdef _OPENMP
    fftw_cleanup_threads();
#endif

    return 0;
}
