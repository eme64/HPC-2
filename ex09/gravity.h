#pragma once

#include "helper_math.h"

class Pairwise_Gravity
{
public:
	Pairwise_Gravity(float G) :
		G(G)
	{}

	// masses are assumed to be 1
	__device__ __host__ inline float3 operator()(float3 dst, float3 src) const
	{
		const float3 dr = dst - src;
		const float rij2 = dot(dr, dr);

		const float r_1 = 1.0f / sqrtf(rij2);
		const float r_3 = r_1*r_1*r_1;

		const float IfI = G * r_3;

		return dr * (-IfI);
		
	}

private:

	float G;
};

class Pairwise_LJ
{
public:
	Pairwise_LJ(float epsilon, float sigma) :
		epsilon(epsilon), sigma(sigma)
	{
		const float sigma2 = sigma*sigma;
		const float sigma4 = sigma2*sigma2;
		sigma6 = sigma2 * sigma4;
		sigma12 = sigma6 * sigma6;
	}

	// masses are assumed to be 1
	__device__ __host__ inline float3 operator()(float3 dst, float3 src) const
	{
		const float3 dr dst-src;
		const float d2 = dot(dr, dr);
		const float d4 = d2*d2;
		const float d8 = d4*d4;
		const float d6 = d4*d2;
		const float d14 = d8*d6;
		const float left = sigma12 / d14;
		const float right = sigma6 / d8;
		return -4.0*epsilon* dr * (left - right);
	}

private:

	float epsilon;
	float sigma;
	float sigma12;
	float sigma6;
};
