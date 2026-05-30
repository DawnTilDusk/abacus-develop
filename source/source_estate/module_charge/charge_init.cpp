#include <vector>
#include <algorithm>

#include "charge.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_io/module_parameter/parameter.h"
#include "source_base/libm/libm.h"
#include "source_base/math_integral.h"
#include "source_base/math_sphbes.h"
#include "source_base/parallel_reduce.h"
#include "source_base/parallel_comm.h"
#include "source_base/timer.h"
#include "source_base/tool_threading.h"
#include "source_estate/magnetism.h"
#include "source_pw/module_pwdft/parallel_grid.h"
#include "source_io/module_output/cube_io.h"
#include "source_io/module_chgpot/rhog_io.h"
#include "source_io/module_wf/read_wf2rho_pw.h"
#include "source_io/module_restart/restart.h"
#include "source_io/module_async_io/async_io_manager.h"
#include "source_io/module_async_io/io_buffer.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_cell/klist.h"

// ============================================================================
// AsyncIOScopeGuard: RAII 辅助类
// 确保在异常退出时正确等待 I/O 任务完成, 避免资源泄漏。
// ============================================================================
class AsyncIOScopeGuard
{
  public:
    explicit AsyncIOScopeGuard(AsyncIOManager& mgr) : mgr_(mgr) {}
    ~AsyncIOScopeGuard()
    {
        // 如果管理器仍在运行且有待完成任务, 等待完成
        if (mgr_.is_running())
        {
            mgr_.wait_all();
        }
    }
    AsyncIOScopeGuard(const AsyncIOScopeGuard&) = delete;
    AsyncIOScopeGuard& operator=(const AsyncIOScopeGuard&) = delete;

  private:
    AsyncIOManager& mgr_;
};

void Charge::init_rho(const UnitCell& ucell,
                      const Parallel_Grid& pgrid,
                      const ModuleBase::ComplexMatrix& strucFac,
                      ModuleSymmetry::Symmetry& symm,
                      const void* klist,
                      const void* wfcpw)
{
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "init_chg", PARAM.inp.init_chg);

    const int nspin = PARAM.inp.nspin;
    assert(nspin>0);

    std::string init_chg_upper = PARAM.inp.init_chg;
    std::transform(init_chg_upper.begin(), init_chg_upper.end(), init_chg_upper.begin(), ::toupper);
    std::cout << " START CHARGE         : " << init_chg_upper << std::endl;

    // we need to set the omega for the charge density
    set_omega(&ucell.omega);
    this->pgrid = &pgrid;

    bool read_error = false;
    bool read_kin_error = false;
    // RAII 守卫: 确保函数退出时等待所有异步 I/O 完成
    AsyncIOManager& async_mgr = AsyncIOManager::instance();
    AsyncIOScopeGuard io_guard(async_mgr);

    if (PARAM.inp.init_chg == "file" || PARAM.inp.init_chg == "auto")
    {
        GlobalV::ofs_running << " Read electron density from file" << std::endl;

        // ==================================================================
        // 路径 1: 异步二进制读取 (重叠 I/O 与初始化计算)
        // ==================================================================
        std::stringstream binary;
        binary << PARAM.globalv.global_readin_dir << PARAM.inp.suffix + "-CHARGE-DENSITY.restart";
        const std::string binary_path = binary.str();

        // ---------- 计时器: 异步读取路径 ----------
        ModuleBase::timer::start("Charge", "init_rho_async_binary");

        // 预构建 fftixyz2ig 映射 (不依赖文件数据, 可与 I/O 重叠)
        const int nx = rhopw->nx;
        const int ny = rhopw->ny;
        const int nz = rhopw->nz;
        std::vector<int> fftixyz2ig(rhopw->nxyz, -1);
        for (int ig = 0; ig < rhopw->npw; ++ig)
        {
            int isz = rhopw->ig2isz[ig];
            int iz = isz % nz;
            int is_tmp = isz / nz;
            int ixy = rhopw->is2fftixy[is_tmp];
            int ixyz = iz + nz * ixy;
            fftixyz2ig[ixyz] = ig;
        }
// async_mgr 已在上文通过 RAII 守卫初始化
        // 获取异步 I/O 管理器实例 (已在 main.cpp 中启动)
        AsyncIOManager& async_mgr = AsyncIOManager::instance();

        // rank 0 (in pool) 提交异步读取任务
        // 注意: 仅 rank 0 执行文件 I/O, 其余进程在 Phase 2 中等待广播
        bool async_read_submitted = false;
        if (GlobalV::RANK_IN_POOL == 0)
        {
            IOBuffer read_buf = IOBuffer::make_binary_read_result(binary_path, rhopw->npw);
            async_read_submitted = async_mgr.submit_rhog_read(std::move(read_buf));
            if (async_read_submitted)
            {
                GlobalV::ofs_running << " [TIMER] Async rhog read submitted: " << binary_path << std::endl;
            }
        }

        // ---- Phase 2: 在 I/O 进行时, 主线程执行不依赖数据的初始化 (重叠) ----
        // 为所有自旋清零 rhog 数组 (O(nspin * npw), 不涉及文件 I/O)
        for (int is = 0; is < nspin; ++is)
        {
            ModuleBase::GlobalFunc::ZEROS(rhog[is], rhopw->npw);
        }

        // ---- Phase 3: 等待异步读取结果 ----
        bool async_read_ok = false;
        size_t async_data_bytes = 0;
        if (async_read_submitted && GlobalV::RANK_IN_POOL == 0)
        {
            IOBuffer result = async_mgr.wait_next_completed();
            if (!result.has_error() && !result.data().empty())
            {
                // 验证头部一致性
                if (result.gamma_only_in() != rhopw->gamma_only)
                {
                    GlobalV::ofs_running << " WARNING: gamma_only mismatch in restart file (file="
                                         << result.gamma_only_in() << ", expected="
                                         << rhopw->gamma_only << ")" << std::endl;
                }

                // ---- Phase 4: Miller 索引映射 (依赖 I/O 结果) ----
                const int npwtot_in = result.npwtot_in();
                const int nspin_in = result.nspin_in();
                const std::vector<double>& data = result.data();
                const std::vector<int>& miller = result.miller();

                async_data_bytes = data.size() * sizeof(double);

                for (int is = 0; is < nspin_in && is < nspin; ++is)
                {
                    const double* spin_data = data.data() + static_cast<size_t>(is) * 2 * npwtot_in;
                    for (int i = 0; i < npwtot_in; ++i)
                    {
                        int ix = miller[i * 3];
                        int iy = miller[i * 3 + 1];
                        int iz = miller[i * 3 + 2];

                        if (ix <= -int((nx + 1) / 2) || ix >= int(nx / 2) + 1
                            || iy <= -int((ny + 1) / 2) || iy >= int(ny / 2) + 1
                            || iz <= -int((nz + 1) / 2) || iz >= int(nz / 2) + 1)
                        {
                            continue;
                        }

                        if (ix < 0) ix += nx;
                        if (iy < 0) iy += ny;
                        if (iz < 0) iz += nz;

                        int fftixy = iy + rhopw->fftny * ix;
                        if (GlobalV::RANK_IN_POOL == rhopw->fftixy2ip[fftixy])
                        {
                            int fftixyz = iz + nz * fftixy;
                            int ig = fftixyz2ig[fftixyz];
                            if (ig >= 0 && ig < rhopw->npw)
                            {
                                rhog[is][ig] = std::complex<double>(
                                    spin_data[2 * i], spin_data[2 * i + 1]);
                            }
                        }
                    }
                }

                // nspin=2 -> 4 转换
                if (nspin_in == 2 && nspin == 4)
                {
                    for (int ig = 0; ig < rhopw->npw; ++ig)
                    {
                        rhog[3][ig] = rhog[1][ig];
                    }
                    ModuleBase::GlobalFunc::ZEROS(rhog[1], rhopw->npw);
                    ModuleBase::GlobalFunc::ZEROS(rhog[2], rhopw->npw);
                }

                async_read_ok = true;
                GlobalV::ofs_running << " [TIMER] Async read OK: " << binary_path
                                     << " (" << async_data_bytes << " bytes, "
                                     << npwtot_in << " G-vectors, "
                                     << nspin_in << " spins)" << std::endl;
            }
            else
            {
                GlobalV::ofs_running << " [TIMER] Async read FAILED for " << binary_path
                                     << ": " << result.error_message() << std::endl;
            }
        }

#ifdef __MPI
        // 广播异步读取结果标志, 确保所有进程进入同一路径
        {
            int async_ok_int = async_read_ok ? 1 : 0;
            MPI_Bcast(&async_ok_int, 1, MPI_INT, 0, POOL_WORLD);
            async_read_ok = (async_ok_int != 0);
        }

        // 将 rank 0 的 rhog 广播到 pool 内所有进程
        if (async_read_ok)
        {
            for (int is = 0; is < nspin; ++is)
            {
                MPI_Bcast(rhog[is], rhopw->npw, MPI_DOUBLE_COMPLEX, 0, POOL_WORLD);
            }
            size_t bcast_bytes = static_cast<size_t>(nspin) * rhopw->npw * sizeof(std::complex<double>);
            GlobalV::ofs_running << " [TIMER] Broadcast rhog done: " << bcast_bytes << " bytes" << std::endl;
        }
#endif

        ModuleBase::timer::end("Charge", "init_rho_async_binary");

        // ==================================================================
        // 路径 2: 同步 read_rhog 回退
        // ==================================================================
        bool sync_read_ok = false;
        if (!async_read_ok)
        {
            ModuleBase::timer::start("Charge", "init_rho_sync_binary");
            if (ModuleIO::read_rhog(binary_path, rhopw, rhog))
            {
                sync_read_ok = true;
                // 输出文件大小信息
                size_t file_bytes = static_cast<size_t>(rhopw->npwtot) * nspin * sizeof(std::complex<double>);
                GlobalV::ofs_running << " [TIMER] Sync read OK: " << binary_path
                                     << " (npwtot=" << rhopw->npwtot
                                     << ", nspin=" << nspin
                                                     << ", estimated " << file_bytes << " bytes)" << std::endl;
            }
            ModuleBase::timer::end("Charge", "init_rho_sync_binary");
        }

        // ==================================================================
        // 路径 3: MPI-IO (read_rhog_mpi) 回退
        // ==================================================================
        bool mpiio_read_ok = false;
        if (!async_read_ok && !sync_read_ok)
        {
#ifdef __MPI
            ModuleBase::timer::start("Charge", "init_rho_mpiio");
            if (ModuleIO::read_rhog_mpi(binary_path, rhopw, rhog, POOL_WORLD))
            {
                mpiio_read_ok = true;
                size_t file_bytes = static_cast<size_t>(rhopw->npwtot) * nspin * sizeof(std::complex<double>);
                GlobalV::ofs_running << " [TIMER] MPI-IO read OK: " << binary_path
                                     << " (" << file_bytes << " bytes)" << std::endl;
            }
            ModuleBase::timer::end("Charge", "init_rho_mpiio");
#endif
        }

        // ==================================================================
        // 路径 4: Cube 文件读取 (二进制全部失败时)
        // ==================================================================
        if (async_read_ok || sync_read_ok || mpiio_read_ok)
        {
            ModuleBase::timer::start("Charge", "init_rho_recip2real");
            for (int is = 0; is < nspin; ++is)
            {
                rhopw->recip2real(rhog[is], rho[is]);
            }
            ModuleBase::timer::end("Charge", "init_rho_recip2real");

            GlobalV::ofs_running << " [TIMER] recip2real done for " << nspin << " spin(s)." << std::endl;

            // ---- 输出 I/O 统计信息 ----
            size_t async_submitted = async_mgr.total_submitted();
            size_t async_completed = async_mgr.total_completed();
            size_t async_rejected = async_mgr.total_rejected();
            GlobalV::ofs_running << " [STATS] AsyncIOManager: submitted=" << async_submitted
                                 << ", completed=" << async_completed
                                 << ", rejected=" << async_rejected << std::endl;
            GlobalV::ofs_running << " [STATS] Charge density read: npwtot=" << rhopw->npwtot
                                 << ", npw=" << rhopw->npw
                                 << ", nspin=" << nspin
                                 << ", nrxx=" << rhopw->nrxx
                                 << ", nxyz=" << rhopw->nxyz << std::endl;
        }
        else
        {
            ModuleBase::timer::start("Charge", "init_rho_cube_read");
            for (int is = 0; is < nspin; ++is)
            {
				std::stringstream ssc;

				if(nspin==1)
				{
                    ssc << PARAM.globalv.global_readin_dir << "chg.cube";
				}
				else
				{
					ssc << PARAM.globalv.global_readin_dir << "chgs" << is + 1 << ".cube";
				}


                if (ModuleIO::read_vdata_palgrid(pgrid,
                    (PARAM.inp.esolver_type == "sdft" ? GlobalV::RANK_IN_BPGROUP : GlobalV::MY_RANK),
                    GlobalV::ofs_running,
                    ssc.str(),
                    this->rho[is],
                    ucell.nat))
                {
                    GlobalV::ofs_running << " [TIMER] Cube read OK: " << ssc.str() << std::endl;
                }
                else if (is > 0)    // nspin=2 or 4
                {
                    if (is == 1)    // failed at the second spin
                    {
                        std::cout << " Incomplete electron density file." << std::endl;
                        read_error = true;
                        break;
                    }
                    else if (is == 2)   // read 2 files when nspin=4
                    {
                        GlobalV::ofs_running << " Didn't read in the electron density but would rearrange it later. "
                            << std::endl;
                    }
                    else if (is == 3)   // read 2 files when nspin=4
                    {
                        GlobalV::ofs_running << " rearrange electron density " << std::endl;
                        for (int ir = 0; ir < this->rhopw->nrxx; ir++)
                        {
                            this->rho[3][ir] = this->rho[0][ir] - this->rho[1][ir];
                            this->rho[0][ir] = this->rho[0][ir] + this->rho[1][ir];
                            this->rho[1][ir] = 0.0;
                            this->rho[2][ir] = 0.0;
                        }
                    }
                }
                else
                {
                    read_error = true;
                    break;
                }
            }
            ModuleBase::timer::end("Charge", "init_rho_cube_read");
        }

        if (read_error)
        {
            const std::string warn_msg
                = " WARNING: \"init_chg\" is enabled but ABACUS failed to read\n charge density from file.\n"
                  " Please check if there is chg.cube (for nspin=1) or chgsx.cube (x=1,2,etc.) or\n {suffix}-CHARGE-DENSITY.restart in the "
                  "directory.\n";
            std::cout << warn_msg;
            if (PARAM.inp.init_chg == "file")
            {
                ModuleBase::WARNING_QUIT("Charge::init_rho",
                                         "Failed to read in charge density from file.\n For initializing atomic "
                                         "charge in calculations,\n please set init_chg to atomic in INPUT.");
            }
        }

        if (XC_Functional::get_ked_flag())
        {
            // If the charge density is not read in, then the kinetic energy density is not read in either
            if (!read_error)
            {
                GlobalV::ofs_running << " try to read kinetic energy density from file" << std::endl;
                // try to read charge from binary file first, which is the same as QE
                std::vector<std::complex<double>> kin_g_space(nspin * this->ngmc, {0.0, 0.0});
                std::vector<std::complex<double>*> kin_g;
                for (int is = 0; is < nspin; is++)
                {
                    kin_g.push_back(kin_g_space.data() + is * this->ngmc);
                }

                std::stringstream binary;
                binary << PARAM.globalv.global_readin_dir << PARAM.inp.suffix + "-TAU-DENSITY.restart";

                // [TIMER] 动能力密度二进制读取
                ModuleBase::timer::start("Charge", "init_rho_tau_binary");
                bool tau_binary_ok = ModuleIO::read_rhog(binary.str(), rhopw, kin_g.data());
                ModuleBase::timer::end("Charge", "init_rho_tau_binary");

                if (tau_binary_ok)
                {
                    GlobalV::ofs_running << " [TIMER] Tau binary read OK: " << binary.str() << std::endl;
                    // [TIMER] recip2real 变换
                    ModuleBase::timer::start("Charge", "init_rho_tau_recip2real");
                    for (int is = 0; is < nspin; ++is)
                    {
                        rhopw->recip2real(kin_g[is], this->kin_r[is]);
                    }
                    ModuleBase::timer::end("Charge", "init_rho_tau_recip2real");
                    GlobalV::ofs_running << " [TIMER] Tau recip2real done for " << nspin << " spin(s)." << std::endl;
                }
                else
                {
                    // [TIMER] 动能力密度 Cube 文件读取
                    ModuleBase::timer::start("Charge", "init_rho_tau_cube");
                    for (int is = 0; is < nspin; is++)
                    {
                        std::stringstream ssc;
                        ssc << PARAM.globalv.global_readin_dir << "SPIN" << is + 1 << "_TAU.cube";
                        // mohan update 2012-02-10, sunliang update 2023-03-09
                        if (ModuleIO::read_vdata_palgrid(
                                pgrid,
                                (PARAM.inp.esolver_type == "sdft" ? GlobalV::RANK_IN_BPGROUP : GlobalV::MY_RANK),
                                GlobalV::ofs_running,
                                ssc.str(),
                                this->kin_r[is],
                                ucell.nat))
                        {
                            GlobalV::ofs_running << " [TIMER] Tau cube read OK: " << ssc.str() << std::endl;
                        }
                        else
                        {
                            read_kin_error = true;
                            std::cout << " WARNING: \"init_chg\" is enabled but ABACUS failed to read kinetic energy "
                                         "density from file.\n"
                                         " Please check if there is SPINX_TAU.cube (X=1,...) or "
                                         "{suffix}-TAU-DENSITY.restart in the directory.\n"
                                      << std::endl;
                            break;
                        }
                    }
                    ModuleBase::timer::end("Charge", "init_rho_tau_cube");
                }
            }
            else
            {
                read_kin_error = true;
            }
        }
    }

    if (PARAM.inp.init_chg == "atomic" || read_error)
    {
        if (read_error)
        {
            std::cout << " Charge::init_rho: use atomic initialization instead." << std::endl;
        }
        this->atomic_rho(nspin, ucell.omega, rho, strucFac, ucell);
    }

    // initial tau = 3/5 rho^2/3, Thomas-Fermi
    if (XC_Functional::get_ked_flag())
    {
        if (PARAM.inp.init_chg == "atomic" || read_kin_error)
        {
            if (read_kin_error)
            {
                std::cout << " Charge::init_rho: init kinetic energy density from rho." << std::endl;
            }
            const double fact = (3.0 / 5.0) * pow(3.0 * ModuleBase::PI * ModuleBase::PI, 2.0 / 3.0);
            for (int is = 0; is < nspin; ++is)
            {
                for (int ir = 0; ir < this->rhopw->nrxx; ++ir)
                {
                    kin_r[is][ir] = fact * pow(std::abs(rho[is][ir]) * nspin, 5.0 / 3.0) / nspin;
                }
            }
        }
    }

    // Peize Lin add 2020.04.04
    if (GlobalC::restart.info_load.load_charge && !GlobalC::restart.info_load.load_charge_finish)
    {
        for (int is = 0; is < nspin; ++is)
        {
            try
            {
                GlobalC::restart.load_disk("charge", is, this->nrxx, rho[is]);
            }
            catch (const std::exception& e)
            {
                // try to load from the output of `out_chg`
                std::stringstream ssc;
                ssc << PARAM.globalv.global_readin_dir << "chgs" << is + 1 << ".cube";
                if (ModuleIO::read_vdata_palgrid(pgrid,
                    (PARAM.inp.esolver_type == "sdft" ? GlobalV::RANK_IN_BPGROUP : GlobalV::MY_RANK),
                    GlobalV::ofs_running,
                    ssc.str(),
                    this->rho[is],
                    ucell.nat))
                {
                    GlobalV::ofs_running << " Read in electron density: " << ssc.str() << std::endl;
                }
            }
        }
        GlobalC::restart.info_load.load_charge_finish = true;
    }

#ifdef __MPI
    this->init_chgmpi();
#endif
    if (PARAM.inp.init_chg == "wfc")
    {
        if (wfcpw == nullptr)
        {
            ModuleBase::WARNING_QUIT("Charge::init_rho", "wfc is only supported for PW-KSDFT.");
        }

        const ModulePW::PW_Basis_K* pw_wfc = reinterpret_cast<ModulePW::PW_Basis_K*>(const_cast<void*>(wfcpw));
        const K_Vectors* kv = reinterpret_cast<const K_Vectors*>(klist);

		ModuleIO::read_wf2rho_pw(pw_wfc, symm, *this,
                PARAM.globalv.global_readin_dir,
				GlobalV::KPAR, GlobalV::MY_POOL, GlobalV::MY_RANK,
                GlobalV::NPROC_IN_POOL, GlobalV::RANK_IN_POOL,
				PARAM.inp.nbands, nspin, PARAM.globalv.npol,
				kv->get_nkstot(),kv->ik2iktot,kv->isk,GlobalV::ofs_running);
    }

    // ---- 输出 I/O 计时摘要 ----
    GlobalV::ofs_running << " [TIMER] === I/O Performance Summary ===" << std::endl;
    GlobalV::ofs_running << " [TIMER] init_rho completed. read_error=" << read_error
                         << ", read_kin_error=" << read_kin_error << std::endl;
}

//==========================================================
// computes the core charge on the real space 3D mesh.
//==========================================================
void Charge::set_rho_core(const UnitCell& ucell,
                          const ModuleBase::ComplexMatrix& structure_factor,
                          const bool* numeric)
{
    ModuleBase::TITLE("Charge","set_rho_core");
    ModuleBase::timer::start("Charge","set_rho_core");

    bool bl = false;
    for (int it = 0; it<ucell.ntype; it++)
    {
        if (ucell.atoms[it].ncpp.nlcc)
        {
            bl = true;
            break;
        }
    }

    if (!bl)
    {
        ModuleBase::GlobalFunc::ZEROS( this->rho_core, this->rhopw->nrxx);
    	ModuleBase::timer::end("Charge","set_rho_core");
        return;
    }

    double *rhocg = new double[this->rhopw->ngg];
    ModuleBase::GlobalFunc::ZEROS(rhocg, this->rhopw->ngg );

	// three dimension.
    std::complex<double> *vg = new std::complex<double>[this->rhopw->npw];

    for (int it = 0; it < ucell.ntype;it++)
    {
        if (ucell.atoms[it].ncpp.nlcc)
        {
//----------------------------------------------------------
// EXPLAIN : drhoc compute the radial fourier transform for
// each shell of g vec
//----------------------------------------------------------
            this->non_linear_core_correction(
                numeric,
                ucell.omega,
                ucell.tpiba2,
                ucell.atoms[it].ncpp.msh,
                ucell.atoms[it].ncpp.r.data(),
                ucell.atoms[it].ncpp.rab.data(),
                ucell.atoms[it].ncpp.rho_atc.data(),
                rhocg);
//----------------------------------------------------------
// EXPLAIN : multiply by the structure factor and sum
//----------------------------------------------------------
            for (int ig = 0; ig < this->rhopw->npw ; ig++)
            {
                vg[ig] += structure_factor(it, ig) * rhocg[this->rhopw->ig2igg[ig]];
            }
        }
    }

	// for tmp use.
	for(int ig=0; ig< this->rhopw->npw; ig++)
	{
		this->rhog_core[ig] = vg[ig];
	}

    this->rhopw->recip2real(vg, this->rho_core);

    // test on the charge and computation of the core energy
    double rhoima = 0.0;
    double rhoneg = 0.0;
    for (int ir = 0; ir < this->rhopw->nrxx; ir++)
    {
        rhoneg += std::min(0.0, this->rhopw->fft_bundle.get_auxr_data<double>()[ir].real());
        rhoima += std::abs(this->rhopw->fft_bundle.get_auxr_data<double>()[ir].imag());
        // NOTE: Core charge is computed in reciprocal space and brought to real
        // space by FFT. For non smooth core charges (or insufficient cut-off)
        // this may result in negative values in some grid points.
        // Up to October 1999 the core charge was forced to be positive definite.
        // This induces an error in the force, and probably stress, calculation if
        // the number of grid points where the core charge would be otherwise neg
        // is large. The error disappears for sufficiently high cut-off, but may be
        // rather large and it is better to leave the core charge as it is.
        // If you insist to have it positive definite (with the possible problems
        // mentioned above) uncomment the following lines.  SdG, Oct 15 1999
    }

	// mohan fix bug 2011-04-03
    Parallel_Reduce::reduce_pool(rhoneg);
    Parallel_Reduce::reduce_pool(rhoima);

	// mohan changed 2010-2-2, make this same as in atomic_rho.
	// still lack something......
    rhoneg /= this->rhopw->nxyz * ucell.omega;
    rhoima /= this->rhopw->nxyz * ucell.omega;

    // calculate core_only exch-corr energy etxcc=E_xc[rho_core] if required
    // The term was present in previous versions of the code but it shouldn't
    delete [] rhocg;
    delete [] vg;
    ModuleBase::timer::end("Charge","set_rho_core");
    return;
} // end subroutine set_rhoc


void Charge::non_linear_core_correction
(
    const bool &numeric,
    const double omega,
    const double tpiba2,
    const int mesh,
    const double *r,
    const double *rab,
    const double *rhoc,
    double *rhocg) const
{
    ModuleBase::TITLE("charge","drhoc");

	// use labmda instead of repeating codes
	const auto kernel = [&](int num_threads, int thread_id)
	{

	double gx = 0.0;
    double rhocg1 = 0.0;
    double *aux = nullptr;

    // here we compute the fourier transform is the charge in numeric form
    if (numeric)
    {
        aux = new double [mesh];
        // G=0 term

        int igl0 = 0;
        if (this->rhopw->gg_uniq [0] < 1.0e-8)
        {
			// single thread term
			if (thread_id == 0)
			{
				for (int ir = 0;ir < mesh; ir++)
				{
					aux [ir] = r [ir] * r [ir] * rhoc [ir];
				}
				ModuleBase::Integral::Simpson_Integral(mesh, aux, rab, rhocg1);
				//rhocg [1] = fpi * rhocg1 / omega;
				rhocg [0] = ModuleBase::FOUR_PI * rhocg1 / omega;//mohan modify 2008-01-19
			}
            igl0 = 1;
        }

		int igl_beg, igl_end;
		// exclude igl0
		ModuleBase::TASK_DIST_1D(num_threads, thread_id, this->rhopw->ngg - igl0, igl_beg, igl_end);
		igl_beg += igl0;
		igl_end += igl_beg;

        // G <> 0 term
        for (int igl = igl_beg; igl < igl_end;igl++)
        {
            gx = sqrt(this->rhopw->gg_uniq[igl] * tpiba2);
            ModuleBase::Sphbes::Spherical_Bessel(mesh, r, gx, 0, aux);
            for (int ir = 0;ir < mesh; ir++)
            {
                aux [ir] = r[ir] * r[ir] * rhoc [ir] * aux [ir];
            } //  enddo
            ModuleBase::Integral::Simpson_Integral(mesh, aux, rab, rhocg1);
            rhocg [igl] = ModuleBase::FOUR_PI * rhocg1 / omega;
        } //  enddo
        delete [] aux;
    }
    else
    {
        // here the case where the charge is in analytic form,
        // check old version before 2008-12-9
    }

	}; // end kernel

	// do not use omp parallel when this function is already in parallel block
	//
	// it is called in parallel block in Forces::cal_force_cc,
	// but not in other funtcion such as Stress_Func::stress_cc.
	ModuleBase::TRY_OMP_PARALLEL(kernel);

    return;
}
