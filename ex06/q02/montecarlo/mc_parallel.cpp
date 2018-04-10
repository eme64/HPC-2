#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <random>

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
	unsigned long n = 24e8;
	unsigned long ntasks = 24*10;

	if (argc == 2)
		n = atol(argv[1]);

	const double h = (b-a)/n;
	const double ref = 0.73864299803689018;
	double res = 0;
	double t0, t1;
	unsigned long i;

	t0 = omp_get_wtime();
#pragma omp parallel
	{
#pragma omp single nowait
		for (i = 0; i < ntasks; i++)
		{
			#pragma omp task firstprivate(i) shared(res)
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
		} // for
	} // parallel

	res *= h;
	t1 = omp_get_wtime();

	printf("Result=%.16f Error=%e Rel.Error=%e Time=%lf seconds\n", res, fabs(res-ref), fabs(res-ref)/ref, t1-t0);

	return 0;
}
