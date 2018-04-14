#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <random>
#include <iostream>


#include <torc.h>
#include <sys/wait.h>


inline double f(double x)
{
	return sin(cos(x));
}

// WolframAlpha: integral sin(cos(x)) from 0 to 1  =	0.738643
//							0.73864299803689018
//							0.7386429980368901838000902905852160417480209422447648518714116299

double work(int *taskid_p, int *numsteps_p, double *a_p, double *b_p, double *res_p)
{
	double a=*a_p;
	double b=*b_p;
	int taskid=*taskid_p;
	int numsteps=*numsteps_p;
	//std::cout << "w: " << taskid << ", " << numsteps << std::endl;	
	int seed = 42 + taskid;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> udistr(a, b);

        double local_res = 0;
        double xi;
        long j;
        for (j = 0; j < numsteps; j++) {
        	xi = udistr(gen);
                local_res += f(xi);
        }
	//std::cout << local_res << std::endl;
	*res_p = local_res;
	return local_res;
}


int main(int argc, char *argv[])
{
	std::cout << "register" << std::endl;
    	torc_register_task((void*)work);
    	std::cout << "torc_init" << std::endl;
    	torc_init(argc, argv, MODE_MW);
    	std::cout << "do work..." << std::endl;

	double a = 0.0;
	double b = 1.0;
	unsigned long n = 24e6;//24e8;
	unsigned long ntasks = 24*10;

	if (argc == 2)
		n = atol(argv[1]);

	const double h = (b-a)/n;
	const double ref = 0.73864299803689018;
	double res = 0;
	double t0, t1;
	unsigned long i;

	double res_array[ntasks];

	t0 = omp_get_wtime();
	//#pragma omp parallel
	{
	//#pragma omp single nowait
		for (i = 0; i < ntasks; i++)
		{
			int numsteps = n/ntasks;
			res_array[i] = 0;
			torc_task(
				-1, (void (*)())work, 5,
				1, MPI_INT, CALL_BY_COP,
				1, MPI_INT, CALL_BY_COP,
				1, MPI_DOUBLE, CALL_BY_COP,
				1, MPI_DOUBLE, CALL_BY_COP,
				1, MPI_DOUBLE, CALL_BY_RES,
				&i, &numsteps, &a, &b,
				&res_array[i]
			);
			/*
			//#pragma omp task firstprivate(i) shared(res)
			{
				int seed = 42 + i;
				std::mt19937 gen(seed);
				std::uniform_real_distribution<float> udistr(a, b);

				double local_res = 0;
				double xi;
				long j, steps = n/ntasks;
				for (j = 0; j < steps; j++) {
					xi = udistr(gen);
					local_res += f(xi);
				}

				#pragma omp atomic
				res += local_res;
			}
			*/
		} // for
		torc_waitall();
	} // parallel
	
	for(i = 0; i<ntasks; i++)
	{
		res+=res_array[i];
		//std::cout << "for(" << i << ") " << res_array[i] << std::endl;
	}
	res *= h;
	t1 = omp_get_wtime();

	printf("Result=%.16f Error=%e Rel.Error=%e Time=%lf seconds\n", res, fabs(res-ref), fabs(res-ref)/ref, t1-t0);

	return 0;
}
