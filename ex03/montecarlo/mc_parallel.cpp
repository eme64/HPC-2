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
parallelize, single each i a task, each its own rand -> worse than seq.
parallelize, res and rand per thread -> same as seq.
parallelize, res and rand per thread, batchsize -> linear speedup
-> euler tests:
serial:
Result=0.7386544234432187 Error=1.142541e-05 Rel.Error=1.546810e-05 Time=8.466123 seconds
start num threads: 12
Result=0.7386328418724246 Error=1.015616e-05 Rel.Error=1.374976e-05 Time=0.769537 seconds

speedup: 8.466123 / 0.769537 = 11 -> allmost linear !
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

	if (argc > 1)
		n = atol(argv[1]);

	int running_on_num = omp_get_max_threads();

	int batch_size = 10;

	if (argc > 2)
		running_on_num = atol(argv[2]);

	if (argc > 4)
		batch_size = atol(argv[3]);

	const double h = (b-a)/n;
	const double ref = 0.73864299803689018;
	double res = 0;
	double t0, t1;

	const int cacheline_size = 64;
	// create res array and random array. including padding
	const int res_padding = cacheline_size / sizeof(double);
	double res_omp[running_on_num*res_padding];

	int rnd_1_padding = cacheline_size / sizeof(std::mt19937);
	if(rnd_1_padding < 1){rnd_1_padding = 1;}
	std::mt19937 omp_rnd_gen[running_on_num*rnd_1_padding];
	int rnd_2_padding = cacheline_size / sizeof(std::uniform_real_distribution<float>);
	if(rnd_2_padding < 1){rnd_2_padding = 1;}
	std::uniform_real_distribution<float> omp_rnd_udistr[running_on_num*rnd_2_padding];

	//printf("%d, %d\n", rnd_1_padding, rnd_2_padding);

	// initialize:
	for(int c = 0; c < running_on_num; c++)
	{
		res_omp[c*res_padding] = 0;

		int seed = 42+c;
		std::mt19937 gen(seed);
		std::uniform_real_distribution<float> udistr(a, b);

		omp_rnd_gen[c*rnd_1_padding] = gen;
		omp_rnd_udistr[c*rnd_2_padding] = udistr;
	}

	t0 = omp_get_wtime();
	printf("start num threads: %d\n", running_on_num);
	#pragma omp parallel num_threads(running_on_num)
	{
		int omp_id_init = omp_get_thread_num();

		int local_size = n/running_on_num;
		//unsigned long local_offset = local_size*omp_id_init;

		if (omp_id_init == running_on_num-1) {
			local_size+= n % running_on_num;
		}

		//batch_size
		//printf("offset %d\n", local_offset);
		//printf("size %d\n", local_size);

		//#pragma omp for
		for (unsigned long i = 0; i < local_size; i+=10) {
			#pragma omp task untied shared(res_omp, omp_rnd_udistr, omp_rnd_gen)
			{
				int omp_id = omp_get_thread_num();

				for (size_t k = 0; k < batch_size; k++) {
					double xi;
					xi = omp_rnd_udistr[omp_id*rnd_2_padding]( omp_rnd_gen[omp_id*rnd_1_padding] );
					res_omp[omp_id*res_padding] += f(xi);
				}
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
