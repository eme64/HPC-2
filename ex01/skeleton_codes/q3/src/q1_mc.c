#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "q1_mc.h"
#include "q1_integrands.h"

#define SEED 35791246

/*
** unbiased? Yes, since x are iid, with law of big numbers
** variance? ???
**
** barriers: No, but random seeding may be an issue.
** speedup: can be seen when program is run.
** trapz scales, mc not so well...
**
** bonus: ???
*/
double q1_mc_integrate(double (*f)(const double),
                                   const double a,
                                   const double b,
                                   const int N,
                                   const int nthreads)
{
    // TODO: Implement the Monte Carlo integration routine
    //srand(SEED);
    double h = (b-a) / (double)(N);

    double sum = 0;

    #pragma omp parallel
    {
      int seed = SEED+omp_get_thread_num();
      #pragma omp for reduction(+:sum)
      for (size_t i = 0; i < N; i++) {
        // how is the seeding to be done?
        double rnd = (double)rand_r(&i)/RAND_MAX;
        double x = a+rnd*(b-a);
        sum += f(x);
      }
    }

    return h*sum;
}

typedef double (*analytic_fun_t)();
typedef double (*integrand_fun_t)(const double);
void q1_mc() {

    // Up to 2^expsize;
    const int expsize = 15;

    // TODO: set the correct lower bounds, each element for one integrand
    const double a[] = {-1.0, 1.,};

    // TODO: set the correct upper bound, each element for one integrand
    const double b[] = {1., pow(2,expsize),};

    const integrand_fun_t funs [] = {integrand_1, integrand_2};
    const analytic_fun_t analytic[] = {analytic_solution_1, analytic_solution_2};


    // Validation
    {
        printf("validation error\n");
        for(int nfun = 0; nfun < 2; ++nfun){
            printf("function: %3d\n", nfun);
            printf("%10s %10s\n", "N", "error");
            for(long N = 2, i = 0; N <= (1 << expsize);  i += 1, N <<= 1){
                const double trapz =
                    q1_mc_integrate(funs[nfun], a[nfun], b[nfun], N, 1);
                const double analy = (*analytic[nfun])();
                const double err = fabs(trapz - analy);
                printf("%10ld %10.7f\n", N, err);
            }
        }
    }

    // Serial execution:
    {
        printf("\nserial profile\n");
        const int n_repeats = 30;
        const int n_runs = 3;
        double first_time = 0.;
        printf("%10s %10s\n", "N", "rel. time");
        for(long N = 2, i = 0; N <= (1 << expsize);  i += 1, N <<= 1){
            // Run n_repeats experiments.
            double avg_time = 0.;
            for(int k = 0; k < n_repeats; ++k) {

                // Timing over three runs, to minimize
                // the overhead of the clock.
                const double t1 = omp_get_wtime();
                for(int t = 0; t < n_runs; ++t)
                    q1_mc_integrate(integrand_1, a[0], b[0], N, 1);
                const double t2 = omp_get_wtime();

                const double dt = (double) (t2 - t1) / n_runs;
                avg_time += dt;
            }
            avg_time /= n_repeats;
            first_time = i == 0 ? avg_time : first_time;
            printf("%10ld %10.7f\n", N, avg_time / first_time);
        }
    }

    // Parallel strong scaling:
    {
        printf("\nparallel profile: strong scaling\n");
        const long N = 1 << expsize, n_repeats = 30, n_runs = 3;
        double first_time = 0.;

        // Run n_repeats experiments.
        printf("%10s %10s %10s\n", "nthreads", "N", "rel. time");
        for(int nthr = 1; nthr <= _NTHREADS; ++nthr){

            double avg_time = 0.;
            for(int k = 0; k < n_repeats; ++k) {

                // Timing over three runs, to minimize
                // the overhead of the clock.
                const double t1 = omp_get_wtime();
                for(int t = 0; t < n_runs; ++t)
                    q1_mc_integrate(integrand_1, a[0], b[0], N, nthr);
                const double t2 = omp_get_wtime();

                const double dt = (double) (t2 - t1) / n_runs;
                avg_time += dt;
            }
            avg_time /= n_repeats;
            first_time = nthr == 1 ? avg_time : first_time;
            printf("%10d %10ld %10.7f\n", nthr, N, avg_time / first_time);
        }
    }
}
