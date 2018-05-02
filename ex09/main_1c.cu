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
__global__ void nbodyNaiveKernel_pos(const float3* __restrict__ coordinates, const float3* forces, int n, const float3* speed, const float dt)
{
	// Get unique id of the thread
	const int pid = blockIdx.x * blockDim.x + threadIdx.x;

	// Thread id is mapped onto particle id
	// If the id >= than the total number of particles, just exit that thread
	if (pid >= n) return;

	// Load
	float3 r_old = coordinates[pid];
	float3 v_old = speed[pid];
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

__global__ void nbodyNaiveKernel_velo(const float3* old_forces, const new_float3* forces, int n, const float3* velocity, const float dt)
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
	int nparticles = coordinates.size();

	// Use 4 warps in a block, calculate number of blocks,
	// such that total number of threads is >= than number of particles
	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	nbodyNaiveKernel_pos<<< nblocks, nthreads >>> (old_forces.devPtr(), new_forces.devPtr(), nparticles, velocity.devPtr(), dt);
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
	const float dl = L / (float)n_size;

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

int main(int argc, char** argv)
{
	int n = 50000;
	float L = 10;

	if (argc > 1)
	{
		n = atoi(argv[1]);
		assert(n > 0);
	}

	const int nchecks = 1313;
	const int nrepetitions = 10;

	PinnedBuffer<float3> coordinates(n), forces(n);

	// Fill the input array with random data
	srand48(42);
	for (auto& v : coordinates)
		v = L * make_float3( drand48(), drand48(), drand48() );

	// Transfer data to the GPU
	coordinates.uploadToDevice(0);

	//Pairwise_Gravity gravity(10.0);
	Pairwise_LJ ljforce(
		0.1, // epsilon
		0.5 // sigma
	);


	// Naive implementation
	runCheckReport(L, coordinates, forces, nchecks, nrepetitions, "Naive",               nbodyNaive        <decltype(ljforce)>, ljforce);
	runCheckReport(L, coordinates, forces, nchecks, nrepetitions, "Shared memory",       nbodyShared       <decltype(ljforce)>, ljforce);
	/*
	runCheckReport(coordinates, forces, nchecks, nrepetitions, "Naive",               nbodyNaive        <decltype(gravity)>, gravity);
	runCheckReport(coordinates, forces, nchecks, nrepetitions, "Shared memory",       nbodyShared       <decltype(gravity)>, gravity);
	*/
	//runCheckReport(coordinates, forces, nchecks, nrepetitions, "Shared memory + ILP", nbodySharedPlusILP<decltype(gravity)>, gravity);

	return 0;
}
