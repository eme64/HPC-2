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
__global__ void nbodyNaiveKernel(const float3* __restrict__ coordinates, float3* forces, int n, Interaction interaction)
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
			f += interaction(dst, coordinates[i]);

	// Write back the force
	forces[pid] = f;
}

template<typename Interaction>
void nbodyNaive(PinnedBuffer<float3>& coordinates, PinnedBuffer<float3>& forces, Interaction interaction)
{
	int nparticles = coordinates.size();

	// Use 4 warps in a block, calculate number of blocks,
	// such that total number of threads is >= than number of particles
	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	nbodyNaiveKernel<<< nblocks, nthreads >>> (coordinates.devPtr(), forces.devPtr(), nparticles, interaction);
}

//=======================================================================================================================
// One thread per particle + shared memory
//=======================================================================================================================
template<typename Interaction>
__global__ void nbodySharedKernel(const float3* coordinates, float3* forces, int n, Interaction interaction)
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
				f += interaction(dst, cache[j]);

		// Again wait until all the warps are done before moving on
		__syncthreads();
	}

	// If the id is valid, write back the force
	if (pid < n) forces[pid] = f;
}


template<typename Interaction>
void nbodyShared(PinnedBuffer<float3>& coordinates, PinnedBuffer<float3>& forces, Interaction interaction)
{
	int nparticles = coordinates.size();

	const int nthreads = 128;
	const int nblocks = (nparticles + nthreads - 1) / nthreads;

	// Allocate shared memory: nthreads*sizeof(float3) PER BLOCK
	nbodySharedKernel<<< nblocks, nthreads, nthreads*sizeof(float3) >>> (coordinates.devPtr(), forces.devPtr(), nparticles, interaction);
}

//=======================================================================================================================
// One thread per N particles + shared memory + split
//=======================================================================================================================
template<int nDestParticles, typename Interaction>
__global__ void nbodySharedPlusILPKernel(const float3* coordinates, float3* forces, int n, Interaction interaction)
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
					f[d] += interaction(dsts[d], src);
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
void nbodySharedPlusILP(PinnedBuffer<float3>& coordinates, PinnedBuffer<float3>& forces, Interaction interaction)
{
	int nparticles = coordinates.size();

	const int ndsts = 3;
	const int nthreads = 128;
	const int nblocks =  ( (nparticles + ndsts-1) / ndsts + nthreads-1 ) / nthreads;

	const dim3 nthreads3(nthreads, 1, 1);
	const dim3 nblocks3(nblocks, 10, 1);

	forces.clearDevice(0);
	nbodySharedPlusILPKernel<ndsts> <<< nblocks3, nthreads3, nthreads*sizeof(float3) >>> (
			coordinates.devPtr(), forces.devPtr(), nparticles, interaction);
}

template<typename Interaction>
std::pair<double, double> diffNorms(const float3* coordinates, float3* forces, int nparticles, int nchecks, Interaction interaction)
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
				float3 curF = interaction(coordinates[i], coordinates[j]);
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
		kernel(coordinates, forces, interaction);
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);

		float ms = 0;
		cudaEventElapsedTime(&ms, start, stop);
		totalTime += ms;
	}

	coordinates.downloadFromDevice(0);
	forces.     downloadFromDevice(0);

	// Perform check against CPU
	auto errs = diffNorms(coordinates.hostPtr(), forces.hostPtr(), nparticles, nchecks, interaction);

	printf("Kernel '%s' statistics:\n  avg runtime: %.3fms\n  errors: Linf: %e, L2 %e\n\n",
			kernelName.c_str(), totalTime / nrepetitions, errs.first, errs.second);
}

int main(int argc, char** argv)
{
	int n = 50000;

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
		v = make_float3( drand48(), drand48(), drand48() ) * 2;

	// Transfer data to the GPU
	coordinates.uploadToDevice(0);

	//Pairwise_Gravity gravity(10.0);
	Pairwise_LJ ljforce(
		0.1, // epsilon
		0.5 // sigma
	);
	

	// Naive implementation
	runCheckReport(coordinates, forces, nchecks, nrepetitions, "Naive",               nbodyNaive        <decltype(ljforce)>, ljforce);
	runCheckReport(coordinates, forces, nchecks, nrepetitions, "Shared memory",       nbodyShared       <decltype(ljforce)>, ljforce);
	/*
	runCheckReport(coordinates, forces, nchecks, nrepetitions, "Naive",               nbodyNaive        <decltype(gravity)>, gravity);
	runCheckReport(coordinates, forces, nchecks, nrepetitions, "Shared memory",       nbodyShared       <decltype(gravity)>, gravity);
	*/
	//runCheckReport(coordinates, forces, nchecks, nrepetitions, "Shared memory + ILP", nbodySharedPlusILP<decltype(gravity)>, gravity);

	return 0;
}
