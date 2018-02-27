// Skeleton code for HPCSE I (2017HS) Exam, 22.12.2017
// Prof. P. Koumoutsakos, Dr. P. Hadjidoukas
// Coding 2 : Diffusion Statistics

#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <stdio.h>
#include "timer.hpp"

// TODO: subquestion (b)
// Adjust the code so as to provide a hybrid MPI+OpenMP implementation

class Diffusion2D_MPI {
public:
    Diffusion2D_MPI(const double D,
                const double L,
                const int N,
                const double dt,
                const int rank,
                const int procs)
    : D_(D), L_(L), N_(N), dt_(dt), rank_(rank), procs_(procs)
    {
        // initialize to zero
        t_  = 0.0;

        /// real space grid spacing
        dr_ = L_ / (N_ - 1);

        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);

        // number of rows per process
        local_N_ = N_ / procs_;

        // small correction for the last process
        if (rank_ == procs - 1) local_N_ += (N_ % procs_);

        // actual dimension of a row (+2 for the ghosts)
        real_N_ = N_ + 2;
        Ntot = (local_N_ + 2) * (N_+2);

        rho_.resize(Ntot, 0.);                // zero values
        rho_tmp.resize(Ntot, 0.);        // zero values

        // check that the timestep satisfies the restriction for stability
        if (rank_ == 0)
                std::cout << "timestep from stability condition is " << dr_*dr_/(4.*D_) << std::endl;

        initialize_density();
    }

    double advance()
    {
        MPI_Request req[4];
        MPI_Status status[4];

        int prev_rank = rank_ - 1;
        int next_rank = rank_ + 1;

        if (prev_rank >= 0) {
                MPI_Irecv(&rho_[           0*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&rho_[           1*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &req[1]);
        } else {
                req[0] = MPI_REQUEST_NULL;
                req[1] = MPI_REQUEST_NULL;
        }

        if (next_rank < procs_) {
                MPI_Irecv(&rho_[(local_N_+1)*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &req[2]);
                MPI_Isend(&rho_[    local_N_*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &req[3]);
        } else {
                req[2] = MPI_REQUEST_NULL;
                req[3] = MPI_REQUEST_NULL;
        }

        /// Dirichlet boundaries; central differences in space, forward Euler
        /// in time
        // update the interior rows
        for(int i = 2; i < local_N_; ++i) {
          for(int j = 1; j <= N_; ++j) {
            rho_tmp[i*real_N_ + j] = rho_[i*real_N_ + j] +
            fac_
            *
            (
             rho_[i*real_N_ + (j+1)]
             +
             rho_[i*real_N_ + (j-1)]
             +
             rho_[(i+1)*real_N_ + j]
             +
             rho_[(i-1)*real_N_ + j]
             -
             4.*rho_[i*real_N_ + j]
             );
          }
        }

        // ensure boundaries have arrived
        MPI_Waitall(4, req, status);

        // update the first and the last rows
        for(int i = 1; i <= local_N_; i += (local_N_-1)) {
          for(int j = 1; j <= N_; ++j) {
            rho_tmp[i*real_N_ + j] = rho_[i*real_N_ + j] +
            fac_
            *
            (
             rho_[i*real_N_ + (j+1)]
             +
             rho_[i*real_N_ + (j-1)]
             +
             rho_[(i+1)*real_N_ + j]
             +
             rho_[(i-1)*real_N_ + j]
             -
             4.*rho_[i*real_N_ + j]
             );
          }
        }

        /// use swap instead of rho_=rho_tmp. this is much more efficient, because it does not have to copy
        /// element by element.
        using std::swap;
        swap(rho_tmp, rho_);

        t_ += dt_;

        return t_;
    }

    void compute_histogram_hybrid()
    {
        // TODO: subquestion (a)
        // This routine uses OpenMP and MPI to compute the overall histogram of density values.
        // Only rank zero must print the histogram.

        //

    }


private:

    void initialize_density()
    {
        int gi;
        /// initialize rho(x,y,t=0)
        double bound = 1/2.;

        for (int i = 1; i <= local_N_; ++i) {
            gi = rank_ * (N_ / procs_) + i;        // convert local index to global index
            for (int j = 1; j <= N_; ++j) {
                if (std::abs((gi-1)*dr_ - L_/2.) < bound && std::abs((j-1)*dr_ - L_/2.) < bound) {
                    rho_[i*real_N_ + j] = 1 + 0.001*((gi-1)+(j-1));
                } else {
                    rho_[i*real_N_ + j] = 0;
                }
            }
        }

    }

    double D_, L_;
    int N_, Ntot, local_N_, real_N_;

    double dr_, dt_, t_, fac_;

    int rank_, procs_;

    std::vector<double> rho_, rho_tmp;
};

int main(int argc, char* argv[])
{
    const double D  = 1;
    const double L  = 1;
    const int  N  = 1024;
    const double dt = 1e-7;

    MPI_Init(&argc, &argv);

    int rank, procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    if (rank == 0)
        std::cout << "Running with " << procs  << " MPI processes" << std::endl;

    Diffusion2D_MPI system(D, L, N, dt, rank, procs);

    const double tmax = 10000 * dt;
    double time = 0;

    timer t;

    int i = 0;
    t.start();
    while (time < tmax) {
        time = system.advance();
        i++;
        if (i == 100) break; // 100 steps are enough
    }
    t.stop();


    if (rank == 0)
      std::cout << "Timing: " << N << " " << t.get_timing() << std::endl;

    system.compute_histogram_hybrid();

    MPI_Finalize();
    return 0;
}
