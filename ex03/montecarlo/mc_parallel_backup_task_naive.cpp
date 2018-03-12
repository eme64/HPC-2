#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <random>

/*
COMMENTS:

simply parallel for: linear speedup.
have one single res, add each f(xi) with a task atomically: worse than seq.

*/


inline double f(double x)
{
	return sin(cos(x));
}

// WolframAlpha: integral sin(cos(x)) from 0 to 1  =	0.738643
//							0.73864299803689018
//							0.7386429980368901838000902905852160417480209422447648518714116299

int main(int argc, char *argv[])
{
	double a = 0.0;
	double b = 1.0;
	unsigned long n = 1e7;

	if (argc == 2)
		n = atol(argv[1]);

	const double h = (b-a)/n;
	const double ref = 0.73864299803689018;
	double res = 0;
	double t0, t1;
	unsigned long i;


	int running_on_num = omp_get_max_threads();
	const int cacheline_size = 64;
	// create res array and random array. including padding
	const int res_padding = cacheline_size / sizeof(double);
	double res_omp[running_on_num*res_padding];

	//int rnd_padding = cacheline_size / sizeof(std::mt19937);
	//std::mt19937 omp_rnd[running_on_num*rnd_padding];

	// initialize:
	for(int c = 0; c < running_on_num; c++)
	{
		res_omp[c*res_padding] = 0;
	}

	t0 = omp_get_wtime();

	#pragma omp parallel num_threads(running_on_num)
	{
		int omp_num =  omp_get_num_threads();

		unsigned long local_size = n/omp_num;
		//printf("starting\n");
		
		#pragma omp for
		for (i = 0; i < n; i++) {
			#pragma omp task untied shared(res_omp)
			{
				int omp_id = omp_get_thread_num();
				int seed = 42+i;
				std::mt19937 gen(seed);
				std::uniform_real_distribution<float> udistr(a, b);
				double xi;
				xi = udistr(gen);
				//printf("%d\n", omp_id*res_padding);
				res_omp[omp_id*res_padding] += f(xi);
			}
		}

		// may not be necessary:
		#pragma omp barrier
		int omp_id = omp_get_thread_num();

		#pragma omp atomic
		res+=res_omp[omp_id*res_padding];
	}
	res *= h;
	t1 = omp_get_wtime();

	printf("Result=%.16f Error=%e Rel.Error=%e Time=%lf seconds\n", res, fabs(res-ref), fabs(res-ref)/ref, t1-t0);
	return 0;
}
