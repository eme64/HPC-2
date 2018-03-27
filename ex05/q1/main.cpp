#include <iostream>
#include <cstdlib>
#include <random>

#include <mkl.h>
#include <omp.h>

//#include "qr.hpp"



void qr(int N, int p = 30);

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
	
	for(int thr = 1; thr<24; thr++)
	{
		std::cout << "######################################## NUM THREADS: " << thr << std::endl;
		mkl_set_num_threads(thr);
		for(int i=15; i<21; i++)
		{
			qr(i);
		}
	}
}

void qr(int K, int p)
{
	int N = 1 << K; // N >= p, else problem not well posed !
	//int p = 30;
	
	std::cout << "#############################  K: " << K << ", N: " << N << ", p: " << p << std::endl;

	// generate beta
	double* beta = (double*) std::malloc(sizeof(double)*p);
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-10.0, 10.0);

	for(int i=0; i<p; i++)
	{
		beta[i] = dis(gen); // uniform
		//std::cout << beta[i] << std::endl;
	}	
	// create random data:
	double* X = (double*) std::malloc(sizeof(double)* p*N);
	double* y = (double*) std::malloc(sizeof(double)* std::max(N, p)); // must have enough space for solving later
	

	// copy data:
	double* X_copy = (double*) std::malloc(sizeof(double)* p*N);
        double* y_copy = (double*) std::malloc(sizeof(double)* std::max(N, p));
	
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
		//std::cout << "y: " << y[line] << std::endl;
		y_copy[line] = y[line];
	}	

	// QR decomposition
	double* tau = (double*) std::malloc(sizeof(double) * std::max(p, N));
	
	double t0 = omp_get_wtime();
	lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, N, p, X, p, tau);		
	if(info != 0)
	{
		std::cout << "dgeqrf error: " << info << std::endl;
	}else{
		//std::cout << "success dgeqrf !" << std::endl;
	}
	
	// apply Q to y
	int k = std::min(p, N);
	info = LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'T', N, 1, k, X, k, tau, y, 1);
	if(info != 0)
        {
                std::cout << "dormqr error: " << info << std::endl;
        }else{
                //std::cout << "success dormqr !" << std::endl;
        }
	
	// solve result for R
	info = LAPACKE_dtrtrs (LAPACK_ROW_MAJOR, 'U', 'N', 'N', k, 1, X, k, y, 1);
	if(info != 0)
        {
                std::cout << "dtrtrs error: " << info << std::endl;
        }else{
                //std::cout << "success dtrtrs !" << std::endl;
        }
	double t1 = omp_get_wtime();	
	std::cout << "my time: " << t1-t0 << std::endl;
	
	// compare betas:
	double error = 0;
	for(int i=0; i<p; i++)
	{
		//std::cout << "[" << i << "] b: " << beta[i] << ", b_est: " << y[i] << std::endl;
		double e = beta[i] - y[i];
		error += e*e;
	}
	std::cout << "my Err: " << std::sqrt(error) << std::endl;

	// compare to dgels on copy of data:
	double t2 = omp_get_wtime();
	info = LAPACKE_dgels (LAPACK_ROW_MAJOR, 'N', N, p, 1, X_copy, p, y_copy, 1);
	if(info != 0)
        {
                std::cout << "dgels error: " << info << std::endl;
        }else{
                //std::cout << "success dgels !" << std::endl;
        }
	double t3 = omp_get_wtime();
	std::cout << "dgels time: " << t3-t2 << std::endl;
	// compare betas:
	double error_dgels = 0;
	for(int i=0; i<p; i++)
	{
		//std::cout << "[" << i << "] b: " << beta[i] << ", b_est: " << y[i] << std::endl;
		double e = beta[i] - y[i];
	        error_dgels += e*e;
	}
	std::cout << "dgels Err: " << std::sqrt(error_dgels) << std::endl;
	
	// compare my to dgels:
	double difference = 0;
	for(int i=0; i<p; i++)
	{
		//std::cout << "[" << i << "] my: " << y[i] << ", dgels: " << y_copy[i] << std::endl;
		double e = y[i] - y_copy[i];
		difference+= e*e;
	}
	std::cout << "my vs dgels: " << difference << std::endl;


	// free data:
	std::free(beta);
	std::free(y);
	std::free(X);
	std::free(tau);

	std::free(y_copy);
	std::free(X_copy);
}
