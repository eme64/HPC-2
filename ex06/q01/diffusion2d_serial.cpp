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
        /// TODO: create global and local grid
        N_tot = N_*N_;
        
        /// TODO: build periodic MPI topology with cartesian communicator
        /// and obtain the rank's neighbors for each direction
        
        /// TODO: build contiguous (rows) and strided vectors (columns) for boundaries.
        /// each process has a rectangular tile in the cartesian grid
        
        /// real space grid spacing
        dr_ = (rmax_ - rmin_) / (N_ - 1);
        
        /// dt < dx*dx / (4*D) for stability
        dt_ = dr_ * dr_ / (6 * D_);
        
        /// stencil factor
        fac_ = dt_ * D_ / (dr_ * dr_);
        
        // TODO: Modify respectively for MPI implementation
        // Mark that you need to account for the ghost cells in your allocated vector size
        rho_ = new value_type[N_tot];
        rho_tmp = new value_type[N_tot];
        
        std::fill(rho_, rho_+N_tot,0.0);
        std::fill(rho_tmp, rho_tmp+N_tot,0.0);
        
        InitializeSystem();
    }
    
    ~Diffusion2D()
    {
        delete[] rho_;
        delete[] rho_tmp;
        
        // TODO: Free created Datatypes and Communicator
    }
    
    void PropagateDensity();
    
    value_type GetSize() {
        value_type sum = 0;
        // TODO: Change for block communication
        // Calculate partial sum
        for(size_type i = 0; i < N_tot; ++i)
            sum += rho_[i];
        // TODO: fetch partial sum from all processes
        return dr_*dr_*sum;
    }
    
    value_type GetMoment() {
        value_type sum = 0;
        // TODO: Change for block communication
        // Calculate partial sum
        for(size_type i = 0; i < N_; ++i)
            for(size_type j = 0; j < N_; ++j) {
                value_type x = j*dr_ + rmin_;
                value_type y = i*dr_ + rmin_;
                sum += rho_[i*N_ + j] * (x*x + y*y);
            }
        
        return dr_*dr_*sum;
    }
    // TODO: fetch partial sum from all processes
    value_type GetTime() const {return time_;}
    
private:
    
    void InitializeSystem();
    
    const value_type D_, rmax_, rmin_;
    const size_type N_;
    size_type N_tot;
    
    // TODO: define local and global dimensions
    // create MPI Datatypes and Communicator
    
    value_type dr_, dt_, fac_;
    
    value_type time_;
    
    value_type *rho_, *rho_tmp;
};

void Diffusion2D::PropagateDensity()
{
    using std::swap;
    // TODO: Exchange boundaries in x-direction
    
    // TODO: Exchange boundaries in y-direction
    
    // TODO: Update interior of subdomain
    // ensure boundaries have arrived
    // update boundaries
    
    /// Dirichlet boundaries; central differences in space, forward Euler
    /// in time
    
    for(size_type i = 0; i < N_; ++i)
        for(size_type j = 0; j < N_; ++j)
            rho_tmp[i*N_ + j] =
            rho_[i*N_ + j]
            +
            fac_
            *
            (
             (j == N_-1 ? 0 : rho_[i*N_ + (j+1)])
             +
             (j == 0 ? 0 : rho_[i*N_ + (j-1)])
             +
             (i == N_-1 ? 0 : rho_[(i+1)*N_ + j])
             +
             (i == 0 ? 0 : rho_[(i-1)*N_ + j])
             -
             4*rho_[i*N_ + j]
             );
    
    swap(rho_tmp,rho_);
    
    time_ += dt_;
}

void Diffusion2D::InitializeSystem()
{
    time_ = 0;
    
    /// initialize rho(x,y,t=0)
    value_type bound = 1./2;
    
    for(size_type i = 0; i < N_; ++i)
        for(size_type j = 0; j < N_; ++j){
            if(std::fabs(i*dr_+rmin_) < bound && std::fabs(j*dr_+rmin_) < bound){
                rho_[i*N_ + j] = 1;
            }
            else{
                rho_[i*N_ + j] = 0;
            }
            
        }
}

int main(int argc, char* argv[])
{
    // TODO: Initialize MPI domain
    assert(argc == 2);
    
    const value_type D = 1;
    const value_type tmax = 0.001;
    const value_type rmax = 1;
    const value_type rmin = -1;
    
    const size_type N_ = 1<<std::stoul(argv[1]);
    
    Diffusion2D System(D, rmax, rmin, N_);
    // TODO: Make sure you wait for all ranks to initialize
    
    value_type time = 0;
    
    // TODO: measure time with MPI commands
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    
    while(time < tmax){
        System.PropagateDensity();
        time = System.GetTime();
        std::cout << time << '\t' << System.GetSize() << '\t' << System.GetMoment() << std::endl;
    }
    
    end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double>(end-start).count();
    
    std::cout << N_ << '\t' << elapsed << std::endl;
    
    // TODO: Finalize MPI processes
    
    return 0;
}
