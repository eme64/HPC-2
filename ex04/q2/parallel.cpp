/*
report on efficiency:
0.0138619

this I got because I dared to change the function f to be a bit lower.
the current setting is the one where I just about do not get errors,
the sampling is never higher than 1
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <omp.h>
#include <time.h>

typedef float real;

enum {
    M     = 3,     // number of parameters alpha
    NBINS = 50
};

static const real sigmasq = 0.1;

/* dump names */
static const char *hist_names[M] = {"a0.txt", "a1.txt", "a2.txt"};

/* bounds for the proposal distribution g */
static const real lower_bound[M] = {0.7, 1.7, 2.7};
static const real upper_bound[M] = {1.3, 2.3, 3.2};

/* upper bound for rejection sampling */
static const real U = 1.0;

int neval = 0;

struct Data {
    int n;
    real *x, *y;
};

struct Params {
    real a[M]; // alphas
};

struct Histogram {
    int *a[M];
};

static void usage() {
    fprintf(stderr, "usage: ./main <data.txt> <nsamples>\n");
    exit(0);
}

static void read_data(const char *fname, Data *d) {
    FILE *f;
    real x, y;
    int i = 0;

    f = fopen(fname, "r");
    while (2 == fscanf(f, "%g\t%g\n", &x, &y)) {++i;}

    d->n = i;
    d->x = (real*) malloc(i * sizeof(real));
    d->y = (real*) malloc(i * sizeof(real));

    i = 0;
    rewind(f);
    while (2 == fscanf(f, "%g\t%g\n", &x, &y)) {
        d->x[i] = x;
        d->y[i] = y;
        ++i;
    }

    fclose(f);
}

static real likelihood(const Params *P, const Data D) {
    ++neval;
    /* TODO compute likelihood f */
    real pre_fac =  pow(2*M_PI*sigmasq, -0.5*D.n);
    real exp_sum = 0;
    for (size_t k = 0; k < D.n; k++) {
      real temp = D.y[k] - (1.0 * P->a[0]) - (D.x[k] * P->a[1]) - (D.x[k] * D.x[k] * P->a[2]);
      exp_sum += temp*temp;
    }
    //printf("pre_fac: %f\n", pre_fac);
    //printf("exp_sum: %f\n", exp_sum);
    //printf("e: %f\n", pow(M_E, -exp_sum/(2.0*sigmasq)));
    return pre_fac *  pow(M_E, -exp_sum/(2.0*sigmasq));
}

static void sample_proposal(Params *P, std::mt19937 &gen, std::uniform_real_distribution<float> &udistr) {
  for (size_t k = 0; k < M; k++) {
    P->a[k] = lower_bound[k] + (upper_bound[k] - lower_bound[k])* udistr(gen);//((real)rand() / (real)RAND_MAX);
  }
}

static void sample_posterior(const Data D, Params *P, std::mt19937 &gen, std::uniform_real_distribution<float> &udistr) {
    // draw from g:
    sample_proposal(P, gen, udistr);

    // sample U:
    real u = udistr(gen);//((real)rand() / (real)RAND_MAX);

    real g = 1;
    for (size_t k = 0; k < M; k++) {
      g = g/ (upper_bound[k] - lower_bound[k]);
    }
    real rho = likelihood(P, D) / g * 0.00000005; // what about Tfactor?

    //printf("rho: %f\n", rho);
    if(rho >= 1.0)
    {
      printf("too big! %f\n", rho);
    }
    if (u < rho) {
      return; // done
    }else{
      // resample:
      sample_posterior(D, P, gen, udistr);
    }
}

static void ini_histogram(Histogram *H) {
    size_t sz = NBINS * sizeof(int);
    for (int k = 0; k < M; ++k) {
        H->a[k] = (int*) malloc(sz);
        memset(H->a[k], 0, sz);
    }
}

static void fin_histogram(Histogram *H) {
    for (int k = 0; k < M; ++k) free(H->a[k]);
}

static void add_to_histogram(const Params P, Histogram *H) {
    int binid, k;
    real dx;
    for (k = 0; k < M; ++k) {
        dx = (upper_bound[k] - lower_bound[k]) / NBINS;
        binid = (int) ((P.a[k] - lower_bound[k]) / dx);
        H->a[k][binid] ++;
    }
}

static void merge_histogram(Histogram *H_local, Histogram *H) {
    for (int k = 0; k < M; ++k) {
      for (size_t i = 0; i < NBINS; i++) {
        H->a[k][i] += H_local->a[k][i];
      }
    }
}

static void stats_posterior(long nsamples, const Data D, Histogram *H) {

    #pragma omp parallel
    {
      Params P;

      // create local histogram:
      Histogram H_local;
      ini_histogram(&H_local);

      int seed = 42;
  	  std::mt19937 gen(seed);
  	  std::uniform_real_distribution<float> udistr(0, 1);

      #pragma omp for
      for (int i = 0; i < nsamples; ++i) {
          sample_posterior(D, &P, gen, udistr);
          add_to_histogram(P, &H_local);
      }

      #pragma omp critical
      {
        // merge histogram:
        merge_histogram(&H_local, H);
      }
      fin_histogram(&H_local);
    }
}

static void dump_histogram(const Histogram H) {
    FILE *f;
    int k, i;
    real dx, x;
    for (k = 0; k < M; ++k) {
        f = fopen(hist_names[k], "w");
        dx = (upper_bound[k] - lower_bound[k]) / NBINS;
        for (i = 0; i < NBINS; ++i) {
            x = lower_bound[k] + i * dx;
            fprintf(f, "%f %d\n", x, H.a[k][i]);
        }
        fclose(f);
    }
}

int main(int argc, char **argv) {
    int nsamples;
    char *fname;
    Data D;
    Histogram H;

    if (argc != 3) usage();
    fname = argv[1];
    nsamples = atoi(argv[2]);

    read_data(fname, &D);

    ini_histogram(&H);

    printf("num threads: %d\n", omp_get_max_threads());
    double t0 = omp_get_wtime();
    stats_posterior(nsamples, D, &H);
    double t1 = omp_get_wtime();
    printf("time: %f\n", t1-t0);
    printf("neval = %d\n", neval);
    printf("efficiency = %g\n", (float) nsamples / (float) neval);

    dump_histogram(H);

    fin_histogram(&H);
    free(D.x);
    free(D.y);
    return 0;
}
