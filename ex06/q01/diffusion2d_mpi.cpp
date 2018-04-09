#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <chrono>
#include <cmath>

#include <mpi.h>

typedef double value_type;
typedef std::size_t size_type;

class Diffusion2D
{

public:

    Diffusion2D(
                const value_type D,
                const value_type rmax,
                const value_type rmin,
                const size_type N
                )
    : D_(D)
    , rmax_(rmax)
    , rmin_(rmin)
    , N_(N)
    {
        /// DONE: build periodic MPI topology with cartesian communicator
        /// and obtain the rank's neighbors for each direction
        int world_size;
        int world_rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        //MPI_Comm cart_comm;
        int dims[2] = {0, 0};
        int periods[2] = {true, true};
        MPI_Dims_create(world_size, 2, dims);
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm_);

        int cart_size;
        int cart_rank;
        MPI_Comm_size(cart_comm_, &cart_size);
        MPI_Comm_rank(cart_comm_, &cart_rank);
        int coords[2];
        MPI_Cart_coords(cart_comm_, cart_rank, 2, coords);
        std::cout << "r: " << cart_rank << ", x: " << coords[0] << ", y: " << coords[1] << std::endl;

        cart_rank_ = cart_rank;
        MPI_Cart_shift(cart_comm_, 0, 1, &cart_rank_xm_, &cart_rank_xp_);
        MPI_Cart_shift(cart_comm_, 1, 1, &cart_rank_ym_, &cart_rank_yp_);
        /*
        if (world_rank == 0) {
          std::cout << "dims: "<< dims[0] << ", " << dims[1] << std::endl;
        }
        */

        /// DONE: create global and local grid
        N_tot = N_*N_; // global
        Nx_ = N_ / dims[0];
        Ny_ = N_ / dims[1];

        Offx_ = Nx_*coords[0];
        Offy_ = Ny_*coords[1];

        if (coords[0] == dims[0]-1) {
          Nx_ += N_ % dims[0];
        }

        if (coords[1] == dims[1]-1) {
          Ny_ += N_ % dims[1];
        }
        N_tot_loc_ = Nx_ * Ny_;
        //std::cout << "r: " << cart_rank << ", Nx_: " << Nx_ << ", Ny_: " << Ny_ << std::endl;


        /// DONE: build contiguous (rows) and strided vectors (columns) for boundaries.
        /// each process has a rectangular tile in the cartesian grid
        MPI_Type_contiguous(Nx_, MPI_DOUBLE, &datatype_row_);
        MPI_Type_vector(Ny_, 1, Nx_+2, MPI_DOUBLE, &datatype_col_);
        MPI_Type_commit(&datatype_row_);
        MPI_Type_commit(&datatype_col_);

        /// real space grid spacing
        dr_ = (rmax_ - rmin_) / (N_ - 1);

        /// dt < dx*dx / (4*D) for stability
        dt_ = dr_ * dr_ / (6 * D_);

        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);

        // DONE: Modify respectively for MPI implementation
        // Mark that you need to account for the ghost cells in your allocated vector size
        rho_ = new value_type[(Nx_+2) * (Ny_+2)];    // N_tot
        rho_tmp = new value_type[(Nx_+2) * (Ny_+2)]; // N_tot

        std::fill(rho_, rho_+((Nx_+2) * (Ny_+2)),0.0);
        std::fill(rho_tmp, rho_tmp+((Nx_+2) * (Ny_+2)),0.0);

        InitializeSystem();
    }

    ~Diffusion2D()
    {
        delete[] rho_;
        delete[] rho_tmp;

        // DONE: Free created Datatypes and Communicator
        MPI_Comm_free(&cart_comm_);
        MPI_Type_free(&datatype_col_);
        MPI_Type_free(&datatype_row_);
    }

    void PropagateDensity();

    value_type GetSize() {
        value_type sum = 0;
        // Done: Change for block communication
        // Calculate partial sum
        for(size_type i = 0; i < Ny_; ++i)
            for(size_type j = 0; j < Nx_; ++j) {
                sum += rho_[(i+1)*(Nx_+2) + (j+1)];
            }
        // DONE: fetch partial sum from all processes
        value_type sum_all = 0;
        MPI_Allreduce(&sum, &sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return dr_*dr_*sum_all;
    }

    value_type GetMoment() {
        value_type sum = 0;
        // DONE: Change for block communication
        // Calculate partial sum
        for(size_type i = 0; i < Ny_; ++i)
            for(size_type j = 0; j < Nx_; ++j) {
                value_type x = (j + Offx_)*dr_ + rmin_;
                value_type y = (i + Offy_)*dr_ + rmin_;
                sum += rho_[(i+1)*(Nx_+2) + (j+1)] * (x*x + y*y);
            }

        value_type sum_all = 0;
        MPI_Allreduce(&sum, &sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return dr_*dr_*sum_all;
    }
    // TODO: fetch partial sum from all processes
    value_type GetTime() const {return time_;} // not same for all?

private:

    void InitializeSystem();

    const value_type D_, rmax_, rmin_;
    const size_type N_;
    size_type N_tot;

    // DONE: define local and global dimensions
    // create MPI Datatypes and Communicator
    MPI_Comm cart_comm_;
    size_type Nx_; // local
    size_type Ny_;
    size_type N_tot_loc_;
    size_type Offx_;
    size_type Offy_;
    MPI_Datatype datatype_row_;
    MPI_Datatype datatype_col_;
    int cart_rank_;
    int cart_rank_xp_;
    int cart_rank_xm_;
    int cart_rank_yp_;
    int cart_rank_ym_;

    value_type dr_, dt_, fac_;

    value_type time_;

    value_type *rho_, *rho_tmp;
};

void Diffusion2D::PropagateDensity()
{
    using std::swap;

    MPI_Request req[8];
    MPI_Status status[8];

    // DONE: Exchange boundaries in x-direction
    MPI_Irecv(&rho_[(0+1)*(Nx_+2) + (-1+1)], 1, datatype_col_, cart_rank_xm_, 123, cart_comm_, &req[0]);
    MPI_Isend(&rho_[(0+1)*(Nx_+2) + (0+1)], 1, datatype_col_, cart_rank_xm_, 123, cart_comm_, &req[1]);

    MPI_Irecv(&rho_[(0+1)*(Nx_+2) + (Nx_+1)], 1, datatype_col_, cart_rank_xp_, 123, cart_comm_, &req[2]);
    MPI_Isend(&rho_[(0+1)*(Nx_+2) + (Nx_-1+1)], 1, datatype_col_, cart_rank_xp_, 123, cart_comm_, &req[3]);

    // DONE: Exchange boundaries in y-direction
    MPI_Irecv(&rho_[(0-1+1)*(Nx_+2) + (0+1)], 1, datatype_row_, cart_rank_ym_, 123, cart_comm_, &req[4]);
    MPI_Isend(&rho_[(0+1)*(Nx_+2) + (0+1)], 1, datatype_row_, cart_rank_ym_, 123, cart_comm_, &req[5]);

    MPI_Irecv(&rho_[(Ny_+1)*(Nx_+2) + (0+1)], 1, datatype_row_, cart_rank_yp_, 123, cart_comm_, &req[6]);
    MPI_Isend(&rho_[(Ny_-1+1)*(Nx_+2) + (0+1)], 1, datatype_row_, cart_rank_yp_, 123, cart_comm_, &req[7]);

    // TODO: Update interior of subdomain
    for(size_type i = 1; i < Ny_-1; ++i)
        for(size_type j = 1; j < Nx_-1; ++j)
            rho_tmp[(i+1)*(Nx_+2) + (j+1)] =
            rho_[(i+1)*(Nx_+2) + (j+1)]
            +
            fac_
            *
            (
             //(j == Nx_-1 ? 0 : rho_[i*Nx_ + (j+1)])
             rho_[(i+1)*(Nx_+2) + (j+1+1)]
             +
             //(j == 0 ? 0 : rho_[i*Nx_ + (j-1)])
             rho_[(i+1)*(Nx_+2) + (j+1-1)]
             +
             //(i == N_-1 ? 0 : rho_[(i+1)*Nx_ + j])
             rho_[(i+1+1)*(Nx_+2) + (j+1)]
             +
             //(i == 0 ? 0 : rho_[(i-1)*Nx_ + j])
             rho_[(i+1-1)*(Nx_+2) + (j+1)]
             -
             4*rho_[(i+1)*(Nx_+2) + (j+1)]
             );


    // ensure boundaries have arrived
    MPI_Waitall(8, req, status);
    // update boundaries

    for(size_type i = 0; i < Ny_; ++i)
      if (i == 0 || i == Ny_-1) {
        for(size_type j = 0; j < Nx_; ++j){
            rho_tmp[(i+1)*(Nx_+2) + (j+1)] =
            rho_[(i+1)*(Nx_+2) + (j+1)]
            +
            fac_
            *
            (
             //(j == Nx_-1 ? 0 : rho_[i*Nx_ + (j+1)])
             rho_[(i+1)*(Nx_+2) + (j+1+1)]
             +
             //(j == 0 ? 0 : rho_[i*Nx_ + (j-1)])
             rho_[(i+1)*(Nx_+2) + (j+1-1)]
             +
             //(i == N_-1 ? 0 : rho_[(i+1)*Nx_ + j])
             rho_[(i+1+1)*(Nx_+2) + (j+1)]
             +
             //(i == 0 ? 0 : rho_[(i-1)*Nx_ + j])
             rho_[(i+1-1)*(Nx_+2) + (j+1)]
             -
             4*rho_[(i+1)*(Nx_+2) + (j+1)]
             );
           }
         }else{
           size_type j = 0;

           rho_tmp[(i+1)*(Nx_+2) + (j+1)] =
           rho_[(i+1)*(Nx_+2) + (j+1)]
           +
           fac_
           *
           (
            //(j == Nx_-1 ? 0 : rho_[i*Nx_ + (j+1)])
            rho_[(i+1)*(Nx_+2) + (j+1+1)]
            +
            //(j == 0 ? 0 : rho_[i*Nx_ + (j-1)])
            rho_[(i+1)*(Nx_+2) + (j+1-1)]
            +
            //(i == N_-1 ? 0 : rho_[(i+1)*Nx_ + j])
            rho_[(i+1+1)*(Nx_+2) + (j+1)]
            +
            //(i == 0 ? 0 : rho_[(i-1)*Nx_ + j])
            rho_[(i+1-1)*(Nx_+2) + (j+1)]
            -
            4*rho_[(i+1)*(Nx_+2) + (j+1)]
            );

            j = Nx_-1;

            rho_tmp[(i+1)*(Nx_+2) + (j+1)] =
            rho_[(i+1)*(Nx_+2) + (j+1)]
            +
            fac_
            *
            (
             //(j == Nx_-1 ? 0 : rho_[i*Nx_ + (j+1)])
             rho_[(i+1)*(Nx_+2) + (j+1+1)]
             +
             //(j == 0 ? 0 : rho_[i*Nx_ + (j-1)])
             rho_[(i+1)*(Nx_+2) + (j+1-1)]
             +
             //(i == N_-1 ? 0 : rho_[(i+1)*Nx_ + j])
             rho_[(i+1+1)*(Nx_+2) + (j+1)]
             +
             //(i == 0 ? 0 : rho_[(i-1)*Nx_ + j])
             rho_[(i+1-1)*(Nx_+2) + (j+1)]
             -
             4*rho_[(i+1)*(Nx_+2) + (j+1)]
             );
         }

    /// Dirichlet boundaries; central differences in space, forward Euler
    /// in time



    swap(rho_tmp,rho_);

    time_ += dt_;
}

void Diffusion2D::InitializeSystem()
{
    time_ = 0;

    /// initialize rho(x,y,t=0)
    value_type bound = 1./2;

    for(size_type i = 0; i < Ny_; ++i)
        for(size_type j = 0; j < Nx_; ++j){
            if(std::fabs((i + Offx_)*dr_+rmin_) < bound && std::fabs((j + Offy_)*dr_+rmin_) < bound){
                rho_[(i+1)*(Nx_+2) + (j+1)] = 1;
            }
            else{
                rho_[(i+1)*(Nx_+2) + (j+1)] = 0;
            }

        }
}

int main(int argc, char* argv[])
{
    // DONE: Initialize MPI domain
    MPI_Init(&argc, &argv);
    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    assert(argc == 2);

    const value_type D = 1;
    const value_type tmax = 0.001;
    const value_type rmax = 1;
    const value_type rmin = -1;

    const size_type N_ = 1<<std::stoul(argv[1]);

    Diffusion2D System(D, rmax, rmin, N_);
    // DONE: Make sure you wait for all ranks to initialize
    MPI_Barrier(MPI_COMM_WORLD);

    // DONE: measure time with MPI commands
    value_type time = 0;
    //std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    //start = std::chrono::high_resolution_clock::now();
    double t0 = MPI_Wtime();

    while(time < tmax){
        System.PropagateDensity();
        time = System.GetTime();

        value_type getsize = System.GetSize();
        value_type getmoment = System.GetMoment();
        if (world_rank == 0) {
          std::cout << time << '\t' << getsize << '\t' << getmoment << std::endl;
        }
    }

    //end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1-t0;
    //double elapsed = std::chrono::duration<double>(end-start).count();

    if (world_rank == 0) {
      std::cout << N_ << '\t' << elapsed << std::endl;
    }

    MPI_Finalize();

    return 0;
}
