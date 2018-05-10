#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "containers.h"
#include "gravity.h"
#include "helper_math.h"

//=======================================================================================================================
// Naive: one thread per particle
//=======================================================================================================================
template<typename Interaction>
__global__ void nbodyNaiveKernel(const float3* __restrict__ coordinates, float3* forces, int n, float L, Interaction interaction)
{
	// Get unique id of the thread
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;

	// Thread id is mapped onto particle id
	// If the id >= than the total number of particles, just exit that thread
	if (pid >= n) return;

	// Load particle coordinates
	float3 dst = coordinates[pid];

	// Loop over all the other particles, compute the force and accumulate it
	float3 f = make_float3(0);
	for (int i=0; i<n; i++)
		if (i != pid)
			f += interaction(dst, coordinates[i], L);

	// Write back the force
	forces[pid] = f;
}

template<typename Interaction>
void nbodyNaive(int L, PinnedBuffer<float3>& coordinates, PinnedBuffer<float3>& forces, Interaction interaction)
{
	int nparticles = coordinates.size();

	// Use 4 warps in a block, calculate number of blocks,
	// such that total number of threads is >= than number of particles
	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	nbodyNaiveKernel<<< nblocks, nthreads >>> (coordinates.devPtr(), forces.devPtr(), nparticles, L, interaction);
}

//=======================================================================================================================
// One thread per particle + shared memory
//=======================================================================================================================
template<typename Interaction>
__global__ void nbodySharedKernel(const float3* coordinates, float3* forces, int n, float L, Interaction interaction)
{
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;

	// Use shared memory to cache the source particles
	extern __shared__ float3 cache[];

	// Since threads will fill shared memory, all of them in the block must be active
	// But only valid threads should load destination particles
	float3 dst = pid < n ? coordinates[pid] : make_float3(0);

	// Loop over all the other particles, compute the force and accumulate it
	float3 f = make_float3(0);
	for (int i=0; i<n; i += blockDim.x)
	{
		// All the threads in a block read in a coalesced manner into the shared memory
		if (i+threadIdx.x < n) cache[threadIdx.x] = coordinates[i+threadIdx.x];

		// Wait untill all the warps in the block are done
		__syncthreads();


		// Use the cached values in the shared memory to compute the interactions
#pragma unroll 9
		for (int j=0; j<min(blockDim.x, n-i); j++)
			if (pid != i+j)
				// interact:
				f += interaction(dst, cache[j], L);

		// Again wait until all the warps are done before moving on
		__syncthreads();
	}

	// If the id is valid, write back the force
	if (pid < n) forces[pid] = f;
}


template<typename Interaction>
void nbodyShared(float L, PinnedBuffer<float3>& coordinates, PinnedBuffer<float3>& forces, Interaction interaction)
{
	int nparticles = coordinates.size();

	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	// Allocate shared memory: nthreads*sizeof(float3) PER BLOCK
	nbodySharedKernel<<< nblocks, nthreads, nthreads*sizeof(float3) >>> (coordinates.devPtr(), forces.devPtr(), nparticles, L, interaction);
}

//=======================================================================================================================
// One thread per N particles + shared memory + split
//=======================================================================================================================
template<int nDestParticles, typename Interaction>
__global__ void nbodySharedPlusILPKernel(const float3* coordinates, float3* forces, int n, float L, Interaction interaction)
{
	const int chunkId = blockIdx.y;
	const int startId = blockIdx.x * blockDim.x + threadIdx.x;
	const int dstStart = startId * nDestParticles;

	float3 dsts[nDestParticles], f[nDestParticles];
	extern __shared__ float3 cache[];

    for (int i=0; i<nDestParticles; i++)
        f[i] = make_float3(0.0f, 0.0f, 0.0f);


    for (int i=0; i<nDestParticles; i++)
    	if (dstStart+i < n) dsts[i] = coordinates[dstStart + i];


    const int chSize = (n+gridDim.y-1) / gridDim.y;
    const int start = chunkId*chSize;
    const int end =  min( (chunkId+1)*chSize, n );

    for (int i = start; i < end; i+=blockDim.x)
	{
    	if (i+threadIdx.x < n) cache[threadIdx.x] = coordinates[i+threadIdx.x];

    	__syncthreads();

#pragma unroll 4
    	for (int j=0; j<min(blockDim.x, end-i); j++)
    	{
    		const float3 src = cache[j];

			for (int d=0; d<nDestParticles; d++)
				if ( dstStart + d != i+j )
					f[d] += interaction(dsts[d], src, L);
    	}

		__syncthreads();
	}


    for (int i=0; i<nDestParticles; i++)
    {
    	atomicAdd(&forces[dstStart + i].x, f[i].x);
    	atomicAdd(&forces[dstStart + i].y, f[i].y);
    	atomicAdd(&forces[dstStart + i].z, f[i].z);
    }
}

template<typename Interaction>
void nbodySharedPlusILP(float L, PinnedBuffer<float3>& coordinates, PinnedBuffer<float3>& forces, Interaction interaction)
{
	int nparticles = coordinates.size();

	const int ndsts = 3;
	const int nthreads = 128;
	const int nblocks =  ( (nparticles + ndsts-1) / ndsts + nthreads-1 ) / nthreads;

	const dim3 nthreads3(nthreads, 1, 1);
	const dim3 nblocks3(nblocks, 10, 1);

	forces.clearDevice(0);
	nbodySharedPlusILPKernel<ndsts> <<< nblocks3, nthreads3, nthreads*sizeof(float3) >>> (
			coordinates.devPtr(), forces.devPtr(), nparticles, L, interaction);
}

template<typename Interaction>
std::pair<double, double> diffNorms(const float3* coordinates, float3* forces, int nparticles, float L, int nchecks, Interaction interaction)
{
	const int stride = max(1, nparticles / nchecks);

	double linf = 0, l2 = 0;

#pragma omp parallel for reduction(+:l2) reduction(max:linf)
	for (int i=0; i<nparticles; i+=stride)
	{
		double3 totalF = make_double3(0, 0, 0);
		for (int j=0; j<nparticles; j++)
			if (i != j)
			{
				float3 curF = interaction(coordinates[i], coordinates[j], L);
				totalF.x += curF.x;
				totalF.y += curF.y;
				totalF.z += curF.z;
			}

		double3 relDiff;
		relDiff.x = (totalF.x - forces[i].x) / totalF.x;
		relDiff.y = (totalF.y - forces[i].y) / totalF.y;
		relDiff.z = (totalF.z - forces[i].z) / totalF.z;

		linf = std::max({ linf,
				fabs(relDiff.x),
				fabs(relDiff.y),
				fabs(relDiff.z) });

		l2 += relDiff.x*relDiff.x + relDiff.y*relDiff.y + relDiff.z*relDiff.z;

		//printf("Particle %d:  reference [%f %f %f],  got [%f %f %f]\n", i, f.x, f.y, f.z,  forces[i].x, forces[i].y, forces[i].z);
	}

	return { linf, sqrt(l2) / nchecks };
}

template<typename Kernel, typename Interaction>
void runCheckReport(
		float L, // domain size
		PinnedBuffer<float3>& coordinates, PinnedBuffer<float3>& forces,
		int nchecks, int nrepetitions,
		std::string kernelName,
		Kernel kernel, Interaction interaction)
{
	// Check for input consistency
	assert(coordinates.size() == forces.size());
	const int nparticles = coordinates.size();

	// Total execution time of the kernel
	float totalTime = 0;

	// Clear the forces
	forces.clear(0);

	// Allocate CUDA events to measure kernel runtime
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Compute the forces on the GPU
	// Do it several times for more precise timings
	for (int i=0; i<nrepetitions; i++)
	{
		cudaEventRecord(start);
		kernel(L, coordinates, forces, interaction);
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);

		float ms = 0;
		cudaEventElapsedTime(&ms, start, stop);
		totalTime += ms;
	}

	coordinates.downloadFromDevice(0);
	forces.     downloadFromDevice(0);

	// Perform check against CPU
	auto errs = diffNorms(coordinates.hostPtr(), forces.hostPtr(), nparticles, L, nchecks, interaction);

	printf("Kernel '%s' statistics:\n  avg runtime: %.3fms\n  errors: Linf: %e, L2 %e\n\n",
			kernelName.c_str(), totalTime / nrepetitions, errs.first, errs.second);
}


//=======================================================================================================================
// Naive: one thread per particle
//=======================================================================================================================

//=======================================================================================================================
// Naive: one thread per particle
//=======================================================================================================================
__global__ void nbodyNaiveKernel_pos(float3* coordinates, const float3* forces, int n, const float3* velocity, const float dt)
{
	// Get unique id of the thread
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;

	// Thread id is mapped onto particle id
	// If the id >= than the total number of particles, just exit that thread
	if (pid >= n) return;

	// Load
	float3 r_old = coordinates[pid];
	float3 v_old = velocity[pid];
	float3 a_old = forces[pid];

	// save:
	coordinates[pid] = r_old + v_old*dt + 0.5*a_old*dt*dt;
}
void nbody_posKernel(float dt, PinnedBuffer<float3>& coordinates, PinnedBuffer<float3>& forces, PinnedBuffer<float3>& velocity)
{
	int nparticles = coordinates.size();

	// Use 4 warps in a block, calculate number of blocks,
	// such that total number of threads is >= than number of particles
	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	nbodyNaiveKernel_pos<<< nblocks, nthreads >>> (coordinates.devPtr(), forces.devPtr(), nparticles, velocity.devPtr(), dt);
}

__global__ void nbodyNaiveKernel_velo(const float3* old_forces, const float3* new_forces, int n, float3* velocity, const float dt)
{
	// Get unique id of the thread
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;

	// Thread id is mapped onto particle id
	// If the id >= than the total number of particles, just exit that thread
	if (pid >= n) return;

	// Load
	float3 a_old = old_forces[pid];
	float3 a_new = new_forces[pid];

	float3 v_old = velocity[pid];

	// save:
	velocity[pid] = v_old + 0.5*(a_old + a_new)*dt;
}
void nbody_veloKernel(float dt, PinnedBuffer<float3>& old_forces, PinnedBuffer<float3>& new_forces, PinnedBuffer<float3>& velocity)
{
	int nparticles = velocity.size();

	// Use 4 warps in a block, calculate number of blocks,
	// such that total number of threads is >= than number of particles
	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	nbodyNaiveKernel_velo<<< nblocks, nthreads >>> (old_forces.devPtr(), new_forces.devPtr(), nparticles, velocity.devPtr(), dt);
}

// ------------------------------- EX10:
__global__ void nbodyNaiveKernel_Ekin(int n, const float3* velocity, float* Ekin_tot)
{
	// Get unique id of the thread
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	const int laneid = threadIdx.x % 32;

	// Thread id is mapped onto particle id
	// If the id >= than the total number of particles, just exit that thread

	float ek_loc = 0;
	if (pid < n)
	{
		float3 v = velocity[pid];
		ek_loc = dot(v,v)/2.0;
	}

	// sum within warp:
	#pragma unroll
	for(int mask = 32 / 2 ; mask > 0 ; mask >>= 1)
		ek_loc += __shfl_xor(ek_loc, mask);
	// The ek_loc variable of laneid 0 contains the reduction.
	if (laneid == 0) {
		// write back:
		atomicAdd(Ekin_tot, ek_loc);
	}
}
void nbodyKernel_Ekin(PinnedBuffer<float3>& velocity, PinnedBuffer<float>& Ekin_tot)
{
	int nparticles = velocity.size();

	// Use 4 warps in a block, calculate number of blocks,
	// such that total number of threads is >= than number of particles
	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	Ekin_tot.clearDevice(0);
	nbodyNaiveKernel_Ekin<<< nblocks, nthreads >>> (nparticles, velocity.devPtr(), Ekin_tot.devPtr());
}


//=======================================================================================================================
// Naive: one thread per particle
//=======================================================================================================================
template<typename Interaction>
__global__ void nbodyNaiveKernel_Epot(const float3* __restrict__ coordinates, float* Epot_total, int n, float L, Interaction interaction)
{
	// Get unique id of the thread
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;
	const int laneid = threadIdx.x % 32;

	// Thread id is mapped onto particle id
	// If the id >= than the total number of particles, just exit that thread

	float Epot_local = 0;
	if (pid >= n) return;
	
	// Load particle coordinates
	float3 dst = coordinates[pid];

	// Loop over all the other particles, compute the force and accumulate it

	for (int i=0; i<n; i++)
		if (i > pid)
			Epot_local += interaction.energy(dst, coordinates[i], L);

	// sum within warp:
	#pragma unroll
	for(int mask = 32 / 2 ; mask > 0 ; mask >>= 1)
		Epot_local += __shfl_xor(Epot_local, mask);
	// The ek_loc variable of laneid 0 contains the reduction.
	if (laneid == 0) {
		// write back:
		atomicAdd(Epot_total, Epot_local);
	}
}

template<typename Interaction>
void nbodyNaive_Epot(int L, const PinnedBuffer<float3>& coordinates, PinnedBuffer<float>& Epot_total, Interaction interaction)
{
	int nparticles = coordinates.size();

	// Use 4 warps in a block, calculate number of blocks,
	// such that total number of threads is >= than number of particles
	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	Epot_total.clearDevice(0);
	nbodyNaiveKernel_Epot<<< nblocks, nthreads >>> (coordinates.devPtr(), Epot_total.devPtr(), nparticles, L, interaction);
}

void init_data(
	PinnedBuffer<float3> &coords,
	PinnedBuffer<float3> &velocity,
	PinnedBuffer<float3> &forces,
	const int n,
	const float L
)
{
	const int n_side = std::pow(n, 1.0/3.0)+1;
	const float dl = L / (float)n_side;

	for (size_t x = 0; x < n_side; x++) {
		for (size_t y = 0; y < n_side; y++) {
			for (size_t z = 0; z < n_side; z++) {
				int id = (x*n_side + y)*n_side + z;
				if (id<n) {
					coords[id] = make_float3(x*dl, y*dl, z*dl);
					velocity[id] = make_float3(0,0,0);
					forces[id] = make_float3(0,0,0);
				}
			}
		}
	}
}

template<typename Interaction>
void runSimulation(
	PinnedBuffer<float3> &coordinates,
	PinnedBuffer<float3> &velocity,
	PinnedBuffer<float3> &forces,
	const int n,
	const float L,
	const float dt,
	const float T,
	Interaction f_interaction
)
{
	// Check for input consistency
	assert(coordinates.size() == forces.size());
	assert(coordinates.size() == velocity.size());
	const int nparticles = coordinates.size();

	PinnedBuffer<float3> temp_forces(n);

	PinnedBuffer<float> Epot_total(1);
	PinnedBuffer<float> Ekin_total(1);

	// Total execution time of the kernel
	float totalTime = 0;//gpu
	// Allocate CUDA events to measure kernel runtime
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float t = 0;// algo

	int step_counter = 0;

	while (t < T) {
		// update r (coordinates):
		nbody_posKernel(dt, coordinates, forces, velocity);

		if (false) {
			coordinates.downloadFromDevice(0);
			printf("coordinates[0]: %.4f, %.4f, %.4f\n\n", coordinates[0].x, coordinates[0].y, coordinates[0].z);
		}

		// get new forces:
		nbodyShared<decltype(f_interaction)>(L, coordinates, temp_forces, f_interaction);

		if (false) {
			forces.downloadFromDevice(0);
			printf("force[0]: %.4f, %.4f, %.4f\n\n", forces[0].x, forces[0].y, forces[0].z);
			temp_forces.downloadFromDevice(0);
			printf("temp_forces[0]: %.4f, %.4f, %.4f\n\n", temp_forces[0].x, temp_forces[0].y, temp_forces[0].z);
		}

		// update velocity:
		nbody_veloKernel(dt, forces, temp_forces, velocity);

		if (false) {
			velocity.downloadFromDevice(0);
			printf("velocity[0]: %.4f, %.4f, %.4f\n\n", velocity[0].x, velocity[0].y, velocity[0].z);
		}

		// swap forces:
		std::swap(forces, temp_forces);

		t += dt;
		//printf("t: %.4f\n\n", t);


		if (step_counter % 5 == 0) {
			// calculate energy:
			nbodyNaive_Epot(L, coordinates, Epot_total, f_interaction);
			nbodyKernel_Ekin(velocity, Ekin_total);

			// get values from GPU:
			Epot_total.downloadFromDevice(0);
			Ekin_total.downloadFromDevice(0);

			printf("t: %.4f\n\n", t);
			printf("Epot: %.4f, Ekin: %.4f, E: %.4f\n\n", Epot_total[0], Ekin_total[0], Epot_total[0]+Ekin_total[0]);
		}
		step_counter++;
	}

	coordinates.downloadFromDevice(0);
	forces.     downloadFromDevice(0);
	velocity.   downloadFromDevice(0);

	printf("runtime: %.3fms\n\n", totalTime);
}

int main(int argc, char** argv)
{
	int n = 50000;
	float L = 10;
	float dt = 0.000001;//0.0001;
	float T = 1.0;

	if (argc > 1)
	{
		n = atoi(argv[1]);
		assert(n > 0);
	}

	PinnedBuffer<float3> coordinates(n), forces(n), velocity(n);
	init_data(coordinates, velocity, forces, n, L);

	// Transfer data to the GPU
	coordinates.uploadToDevice(0);
	velocity.uploadToDevice(0);
	forces.uploadToDevice(0);

	//Pairwise_Gravity gravity(10.0);
	Pairwise_LJ ljforce(
		0.1, // epsilon
		0.5 // sigma
	);

	runSimulation(
		coordinates, velocity, forces,
		n, L, dt, T,
		ljforce
	);

	return 0;
}
