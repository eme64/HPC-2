#include <iostream>
#include <cstdlib>
#include <random>

#include <algorithm>
#include <cmath>

#include <mkl.h>
#include <omp.h>


#define PARAMETER_P 30
#define LAMBDA 1.0

/* COMMENT:
 *
 * It was important to also control the variance of q(x|y)
 * if it does not decrease with the error (or other schedule)
 * then we will not have fast convergence
 * */

/* MEASUREMENTS (usual print outs)
 * 
 
-------------------------------
N = 32768
T:   T_inv 
GENERATE BUFFERS
GENERATE DATA
dgels time: 0.0477881
DO EXPERIMENTS
REACHED 1e+10, k: 0
REACHED 1e+09, k: 1
REACHED 1e+08, k: 2
REACHED 1e+07, k: 3
REACHED 1e+06, k: 4
REACHED 100000, k: 5
REACHED 10000, k: 6
REACHED 1000, k: 193
REACHED 100, k: 531
REACHED 10, k: 878
REACHED 1, k: 1278
REACHED 0.1, k: 1672
REACHED 0.01, k: 2229
REACHED 0.001, k: 2478
REACHED 0.0001, k: 2857
REACHED 1e-05, k: 3265
REACHED 1e-06, k: 3728
REACHED 1e-07, k: 4111
REACHED 1e-08, k: 4494
TERMINATE: 4495
FREE DATA

-------------------------------
N = 32768
T:   T_invlog 
GENERATE BUFFERS
GENERATE DATA
dgels time: 0.0279679
DO EXPERIMENTS
REACHED 1e+10, k: 103
REACHED 1e+09, k: 416
REACHED 1e+08, k: 731
REACHED 1e+07, k: 1095
REACHED 1e+06, k: 1328
REACHED 100000, k: 1669
REACHED 10000, k: 1971
REACHED 1000, k: 2406
REACHED 100, k: 2875
REACHED 10, k: 3228
REACHED 1, k: 3641
REACHED 0.1, k: 4032
REACHED 0.01, k: 4444
REACHED 0.001, k: 4847
REACHED 0.0001, k: 5165
REACHED 1e-05, k: 5597
REACHED 1e-06, k: 6394
^C ---> timeout / no fast convergence.

-------------------------------
N = 32768
T:   T_invexp 
GENERATE BUFFERS
GENERATE DATA
dgels time: 0.0238519
DO EXPERIMENTS
REACHED 1e+10, k: 0
REACHED 1e+09, k: 1
REACHED 1e+08, k: 2
REACHED 1e+07, k: 3
REACHED 1e+06, k: 4
REACHED 100000, k: 5
REACHED 10000, k: 6
REACHED 1000, k: 155
REACHED 100, k: 475
REACHED 10, k: 1010
REACHED 1, k: 1444
REACHED 0.1, k: 1751
REACHED 0.01, k: 2231
REACHED 0.001, k: 2499
REACHED 0.0001, k: 2987
REACHED 1e-05, k: 3234
REACHED 1e-06, k: 3736
REACHED 1e-07, k: 4226
REACHED 1e-08, k: 4470
TERMINATE: 4471
FREE DATA



 * */
void metropolis_step(
	double* bk, double* X, int N, double* y, double T, double* bk_p, double &Ek_p,
	std::default_random_engine &generator, std::normal_distribution<double> &gaussian,
	double lambda,
	std::mt19937 &gen, std::uniform_real_distribution<> &uniform
)
{
	// out bk_p, E
	
	
	// step 1:
	//std::default_random_engine generator;
        //std::normal_distribution<double> gaussian(0.0,LAMBDA);
	for(size_t i=0; i<PARAMETER_P; i++)
	{
		bk_p[i] = bk[i] + gaussian(generator)*lambda;
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
                        ek+= X[line*PARAMETER_P + i] * bk[i];
			ek_p+= X[line*PARAMETER_P + i] * bk_p[i];
                }
		Ek += ek*ek;
		Ek_p += ek_p*ek_p;
        }
	
	double h = std::min(
			1.0,
			std::exp(-(Ek_p - Ek)/T)
	);
	//std::cout << "h: " << h << ", Ek: " << Ek << ", Ek_p: " << Ek_p << std::endl;
	
	// step 3:
	//std::random_device rd;
        //std::mt19937 gen(rd());
        //std::uniform_real_distribution<> uniform(-10.0, 10.0);
	double u = uniform(gen);
	
	// step 4:
	if(u > h)
	{
		// reject new suggestion
		for(size_t i=0; i<PARAMETER_P; i++)
		{
			bk_p[i] = bk[i];
		}
		//std::cout << "back" << std::endl;
		return;
	}else{
		// keep current suggestion
		//std::cout << "keep" << std::endl;
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
		1.0,
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
			1.0,
        	        gen, uniform
	        );
		
		std::swap(b_curr, b_next);
		
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
	return 1.0 * std::exp(-k);
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
	
	int N = 1 << 15; // 15
	std::cout << "N = " << N << std::endl;
	
	double (*T)(int) = T_invexp;
	std::cout << "T:   T_invexp " << std::endl;
	
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
	
	double *b_curr = (double*) std::malloc(sizeof(double)*PARAMETER_P);
        double *b_next = (double*) std::malloc(sizeof(double)*PARAMETER_P);
        // rnd for first guess
	std::random_device rd;
        std::mt19937 gen_init(rd());
        std::uniform_real_distribution<> dis_init(-10,10);
	

	// rnd for updates, delta_beta and u
	std::default_random_engine generator;
	std::normal_distribution<double> gaussian(0.0,LAMBDA);

	std::random_device rd2;
	std::mt19937 gen(rd2());
	std::uniform_real_distribution<> uniform(0.0, 1.0);
	

	//for(int i=-10; i<6; i++)
	{
		double epsilon = std::pow(10, 10);
		double epsilon_end = std::pow(10, -8);
		
		// generate first guess:
		for(int i=0; i<PARAMETER_P; i++)
        	{
                	b_curr[i] = dis_init(gen_init); // uniform
        	}
		
		int k = 0;
		double distance = epsilon*10;
		do{
			// do a step:
			double error_now = 0;

	                metropolis_step(
	                        b_curr, X, N, y, T(k), b_next, error_now,
	                        generator, gaussian,
	                        std::sqrt(distance) * 0.1,
				gen, uniform
	                );
	
	                std::swap(b_curr, b_next);

			// calculate distance to opt_beta:
			distance = 0;
			for(int i=0; i<PARAMETER_P; i++)
			{
				double e = beta_optimal[i] - b_curr[i];
				distance+=e*e;
				//std::cout << e << std::endl;
			}
			// check:
			
			//if(k%1000 == 0) std::cout << "k: " << k << ", dist: " << distance << std::endl;
			
			if(distance < epsilon)
			{
				std::cout << "REACHED " << epsilon << ", k: " << k << std::endl;
				epsilon = epsilon/10.0;
			}
			k++;
		}while(distance >= epsilon_end);
		std::cout << "TERMINATE: " << k << std::endl;
	}

	// free buffers:
	std::cout << "FREE DATA" << std::endl;
	std::free(beta_optimal);
	std::free(X);
	std::free(y);
	
	std::free(b_curr);
        std::free(b_next);
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
	std::normal_distribution<double> gaussian(0.0,0.1);

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
		b_opt[i] = y_copy[i];
	}
	
	std::free(beta);
	std::free(X_copy);
	std::free(y_copy);
}
