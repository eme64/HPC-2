#include <iostream>
#include <cstdlib>
#include <random>

#include <algorithm>
#include <cmath>

#include <mkl.h>
#include <omp.h>


#define PARAMETER_P 30
#define LAMBDA 0.1
void metropolis_step(
	double* bk, double* X, int N, double* y, double T, double* bk_p, double &Ek_p,
	std::default_random_engine &generator, std::normal_distribution<double> &gaussian,
	std::mt19937 &gen, std::uniform_real_distribution<> &uniform
)
{
	// out bk_p, E
	
	
	// step 1:
	//std::default_random_engine generator;
        //std::normal_distribution<double> gaussian(0.0,LAMBDA);
	for(size_t i=0; i<PARAMETER_P; i++)
	{
		bk_p[i] = bk[i] + gaussian(generator);
	}
	
	// step 2:
	double Ek = 0;
	Ek_p = 0;
	
	for(int line = 0; line < N; line++)
        {
                double ek = -y[line];
		double ek_p = -y[line];
                for(int i = 0; i < PARAMETER_P; i++)
                {
                        ek_p-= X[line*PARAMETER_P + i] * bk[i];
			ek_p-= X[line*PARAMETER_P + i] * bk_p[i];
                }
		Ek += ek*ek;
		Ek_p += ek_p*ek_p;
        }
	
	double h = std::min(
			1.0,
			std::exp(-(Ek - Ek_p)/T)
	);
	
	// step 3:
	//std::random_device rd;
        //std::mt19937 gen(rd());
        //std::uniform_real_distribution<> uniform(-10.0, 10.0);
	double u = uniform(gen);
	
	// step 4:
	if(u <= h)
	{
		// reject new suggestion
		for(size_t i=0; i<PARAMETER_P; i++)
		{
			bk_p[i] = bk[i];
		}
		return;
	}else{
		// keep current suggestion
		return;
	}
}


void mcmc_routine(
        int k_max, double* b_best, int &k_best, double &E_best, double* X, int N, double* y, double (*T)(int),
        std::default_random_engine &generator, std::normal_distribution<double> &gaussian,
        std::mt19937 &gen, std::uniform_real_distribution<> &uniform
)
{
	// initialize a first beta:
	double *b_curr = (double*) std::malloc(sizeof(double)*PARAMETER_P);
	double *b_next = (double*) std::malloc(sizeof(double)*PARAMETER_P);
	std::random_device rd;
        std::mt19937 gen_b(rd());
        std::uniform_real_distribution<> dis_b(-5.0, 15.0);

        for(int i=0; i<PARAMETER_P; i++)
        {
                b_curr[i] = dis_b(gen_b); // uniform
        }

	// do a first step to get some first error:
	metropolis_step(
        	b_curr, X, N, y, T(0), b_next, E_best,
        	generator, gaussian,
        	gen, uniform
	);
	
	// swap
	std::swap(b_next, b_curr);

	// set best for now:
	for(int i=0; i<PARAMETER_P; i++)
        {
                b_best[i] = b_curr[i];
        }
	k_best = 0;

	// do k_max steps:
	for(int k=1; k<k_max; k++)
	{
		double error_now = 0;
		
		metropolis_step(
                	b_curr, X, N, y, T(k), b_next, error_now,
                	generator, gaussian,
        	        gen, uniform
	        );
		
		if(error_now < E_best)
		{
			E_best = error_now;
			// update best beta
			for(int i=0; i<PARAMETER_P; i++)
		        {
                		b_best[i] = b_curr[i];
        		}
			k_best = k;
		}
		

	}
	

	std::free(b_curr);
	std::free(b_next);
}




void generate_data(double *b_opt, double *X, double *y, int N);

double T_invlog(int k)
{
	return 1.0 / std::log(k+1);
}

double T_inv(int k)
{
	return 1.0 / (double)(k+1);
}

double T_invexp(int k)
{
	return std::exp(-k);
}
int main(int argc, char *argv[]){
	/*
 	 * Intel MKL research:
 	 *
 	 * https://software.intel.com/en-us/mkl-developer-reference-c-orthogonal-factorizations-lapack-computational-routines
 	 * section: general matrices, QR factorization
 	 *
 	 * create QR decomp with geqrf
 	 * apply Q to y using ormqr (real) or unmqr (complex)
 	 * somehow solve result for R (upper triangular matrix)
 	 */
	
	int N = 1 << 10; // 15
	std::cout << "N = " << N << std::endl;

	// allocate buffers:
	std::cout << "GENERATE BUFFERS" << std::endl;
	double* beta_optimal = (double*) std::malloc(sizeof(double)*PARAMETER_P);
	double* X = (double*) std::malloc(sizeof(double)* PARAMETER_P*N);
        double* y = (double*) std::malloc(sizeof(double)* N);
	
	// generate data:
	std::cout << "GENERATE DATA" << std::endl;
	generate_data(beta_optimal, X, y, N);

	// do experiments:
	std::cout << "DO EXPERIMENTS" << std::endl;
	
	for(int i=1; i<6; i++)
	{
		double epsilon = std::pow(10, -i);
		std::cout << "EPSILON = " << epsilon << std::endl;
		
		// TODO
	}

	// free buffers:
	std::cout << "FREE DATA" << std::endl;
	std::free(beta_optimal);
	std::free(X);
	std::free(y);
}

void generate_data(double *b_opt, double *X, double *y, int N)
{
	const int p = PARAMETER_P;
	// generate beta
	double* beta = (double*) std::malloc(sizeof(double)*PARAMETER_P);
	double* X_copy = (double*) std::malloc(sizeof(double)* PARAMETER_P*N);
	double* y_copy = (double*) std::malloc(sizeof(double)* N);
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-10.0, 10.0);

	for(int i=0; i<p; i++)
	{
		beta[i] = dis(gen); // uniform
		//std::cout << beta[i] << std::endl;
	}	
	// create random data:
	std::default_random_engine generator;
	std::normal_distribution<double> gaussian(0.0,1.0);

	for(int line = 0; line < N; line++)
	{
		y[line] = gaussian(generator); // gaussian
		for(int i = 0; i < p; i++)
		{
			X[line*p + i] = dis(gen); // uniform
			X_copy[line*p + i] = X[line*p + i];
			y[line]+= X[line*p + i] * beta[i];
		}
		y_copy[line] = y[line];
	}	

	
	// do least squares with dgels:
	double t2 = omp_get_wtime();
	lapack_int info = LAPACKE_dgels (LAPACK_ROW_MAJOR, 'N', N, p, 1, X_copy, p, y_copy, 1);
	if(info != 0)
        {
                std::cout << "dgels error: " << info << std::endl;
        }else{
                //std::cout << "success dgels !" << std::endl;
        }
	double t3 = omp_get_wtime();
	std::cout << "dgels time: " << t3-t2 << std::endl;

	for(int i=0; i<PARAMETER_P; i++)
	{
		b_opt[i] = y[i];
	}
	
	std::free(beta);
	std::free(X_copy);
	std::free(y_copy);
}
