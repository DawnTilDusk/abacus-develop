#include "charge.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/parameter.h"
#include <cstring>
#include <vector>
#ifdef __MPI
void Charge::init_chgmpi()
{
    if (KP_WORLD == MPI_COMM_NULL)
    {
        delete[] rec;
        rec = new int[GlobalV::NPROC_IN_POOL];
        delete[] dis;
        dis = new int[GlobalV::NPROC_IN_POOL];

        const int ncxy = this->rhopw->nx * this->rhopw->ny;
        for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ip++)
        {
            rec[ip] = this->rhopw->numz[ip] * ncxy;
            dis[ip] = this->rhopw->startz[ip] * ncxy;
        }
        
        //release memory before re-allocating
        delete[] chgmpi_tmp_;
        delete[] chgmpi_tot_;
        delete[] chgmpi_tot_aux_;
        chgmpi_tmp_ = new double[this->rhopw->nxyz];
        chgmpi_tot_ = new double[this->rhopw->nxyz];
        chgmpi_tot_aux_ = new double[this->rhopw->nxyz];
    }
}

void Charge::reorder_pool_to_uniform(const double* array_tot, double* array_tot_aux) const
{
    for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip)
    {
        reorder_pool_rank_to_uniform(array_tot, array_tot_aux, ip);
    }
}

void Charge::reorder_pool_rank_to_uniform(const double* array_tot, double* array_tot_aux, const int ip) const
{
    const int ncxy = this->rhopw->nx * this->rhopw->ny;
    const int numz_ip = this->rhopw->numz[ip];
    const int startz_ip = this->rhopw->startz[ip];
    const int nz = this->rhopw->nz;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int ir = 0; ir < ncxy; ++ir)
    {
        for (int iz = 0; iz < numz_ip; ++iz)
        {
            array_tot_aux[nz * ir + startz_ip + iz]
                = array_tot[numz_ip * ir + startz_ip * ncxy + iz];
        }
    }
}

void Charge::extract_uniform_to_local(const double* array_tot, double* array_rho) const
{
    const int ncxy = this->rhopw->nx * this->rhopw->ny;
    const int numz_local = this->rhopw->numz[GlobalV::RANK_IN_POOL];
    const int startz_local = this->rhopw->startz_current;
    const int nz = this->rhopw->nz;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int ir = 0; ir < ncxy; ir++)
    {
        for (int iz = 0; iz < numz_local; iz++)
        {
            array_rho[numz_local * ir + iz]
                = array_tot[nz * ir + startz_local + iz];
        }
    }
}

void Charge::gather_pool_data_nonblocking(const double* array_tmp, double* array_tot, double* array_tot_aux) const
{
    const int my_rank = GlobalV::RANK_IN_POOL;
    const int nproc = GlobalV::NPROC_IN_POOL;
    constexpr int gather_tag = 1024;

    std::memcpy(array_tot + dis[my_rank], array_tmp, sizeof(double) * rec[my_rank]);
    reorder_pool_rank_to_uniform(array_tot, array_tot_aux, my_rank);

    if (nproc <= 1)
    {
        return;
    }

    std::vector<MPI_Request> recv_requests(nproc, MPI_REQUEST_NULL);
    std::vector<MPI_Request> send_requests(nproc, MPI_REQUEST_NULL);
    std::vector<int> completed_indices(nproc, 0);

    for (int ip = 0; ip < nproc; ++ip)
    {
        if (ip == my_rank)
        {
            continue;
        }
        MPI_Irecv(array_tot + dis[ip], rec[ip], MPI_DOUBLE, ip, gather_tag, POOL_WORLD, &recv_requests[ip]);
    }

    for (int ip = 0; ip < nproc; ++ip)
    {
        if (ip == my_rank)
        {
            continue;
        }
        MPI_Isend(array_tmp, rec[my_rank], MPI_DOUBLE, ip, gather_tag, POOL_WORLD, &send_requests[ip]);
    }

    int remaining_recv = nproc - 1;
    while (remaining_recv > 0)
    {
        int outcount = 0;
        MPI_Waitsome(nproc, recv_requests.data(), &outcount, completed_indices.data(), MPI_STATUSES_IGNORE);
        if (outcount == MPI_UNDEFINED)
        {
            break;
        }
        for (int i = 0; i < outcount; ++i)
        {
            const int ip = completed_indices[i];
            reorder_pool_rank_to_uniform(array_tot, array_tot_aux, ip);
        }
        remaining_recv -= outcount;
    }

    MPI_Waitall(nproc, send_requests.data(), MPI_STATUSES_IGNORE);
}

void Charge::reduce_diff_pools(double* array_rho) const
{
    ModuleBase::TITLE("Charge", "reduce_diff_pools");
    ModuleBase::timer::start("Charge", "reduce_diff_pools");
    if (KP_WORLD != MPI_COMM_NULL)
    {
        MPI_Allreduce(MPI_IN_PLACE, array_rho, this->nrxx, MPI_DOUBLE, MPI_SUM, KP_WORLD);
    }
    else
    {
        double* array_tmp = this->chgmpi_tmp_;
        double* array_tot = this->chgmpi_tot_;
        double* array_tot_aux = this->chgmpi_tot_aux_;
        //==================================
        // Collect the rho in each pool
        //==================================
        for (int ir = 0; ir < this->rhopw->nrxx; ++ir)
        {
            array_tmp[ir] = array_rho[ir] / GlobalV::NPROC_IN_POOL;
        }

        //=================================================
        // Gather the rho in each pool 
        // replace MPI_Allgatherv with nonblocking version
        //=================================================
        gather_pool_data_nonblocking(array_tmp, array_tot, array_tot_aux);

        //==================================
        // Reduce all the rho in each cpu
        //==================================
        MPI_Allreduce(array_tot_aux, array_tot, this->rhopw->nxyz, MPI_DOUBLE, MPI_SUM, INT_BGROUP);

        //=====================================
        // Change the order of rho in each cpu
        //=====================================
        extract_uniform_to_local(array_tot, array_rho);
    }
    if(PARAM.globalv.all_ks_run && PARAM.inp.bndpar > 1)
    {
        MPI_Allreduce(MPI_IN_PLACE, array_rho, this->nrxx, MPI_DOUBLE, MPI_SUM, BP_WORLD);
    }
    ModuleBase::timer::end("Charge", "reduce_diff_pools");
}

void Charge::rho_mpi()
{
    ModuleBase::TITLE("Charge", "rho_mpi");
	if (GlobalV::KPAR * PARAM.inp.bndpar <= 1) 
	{
		return;
	}
    ModuleBase::timer::start("Charge", "rho_mpi");

    for (int is = 0; is < PARAM.inp.nspin; ++is)
    {
        reduce_diff_pools(this->rho[is]);
        if (XC_Functional::get_ked_flag() || PARAM.inp.out_elf[0] > 0)
        {
            reduce_diff_pools(this->kin_r[is]);
        }
    }

    ModuleBase::timer::end("Charge", "rho_mpi");
    return;
}

void Charge::kin_r_mpi()
{
    ModuleBase::TITLE("Charge", "kin_r_mpi");
    if (GlobalV::KPAR * PARAM.inp.bndpar <= 1)
    {
        return;
    }
    ModuleBase::timer::start("Charge", "kin_r_mpi");

    if (XC_Functional::get_ked_flag() || PARAM.inp.out_elf[0] > 0)
    {
        for (int is = 0; is < PARAM.inp.nspin; ++is)
        {
            reduce_diff_pools(this->kin_r[is]);
        }
    }

    ModuleBase::timer::end("Charge", "kin_r_mpi");
    return;
}
#endif
