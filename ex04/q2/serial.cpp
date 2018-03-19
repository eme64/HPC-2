#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    return 0;
}

static void sample_proposal(Params *P) {
    /* TODO generate parameters from g */
}

static void sample_posterior(const Data D, Params *P) {
    /* TODO implement rejection sampling */
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

static void stats_posterior(long nsamples, const Data D, Histogram *H) {
    Params P;
    int i;

    for (i = 0; i < nsamples; ++i) {
        sample_posterior(D, &P);
        add_to_histogram(P, H);
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
    
    stats_posterior(nsamples, D, &H);

    printf("neval = %d\n", neval);
    printf("efficiency = %g\n", (float) nsamples / (float) neval);

    dump_histogram(H);

    fin_histogram(&H);
    free(D.x);
    free(D.y);
    return 0;
}
