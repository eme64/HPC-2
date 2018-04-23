#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "fitfun.h"
#include "mcmc.h"


/*
------------ RESULTS:
./serial data.txt 10000 128 0 0
9.803550e-03 1.991151e+00 5.000403e+00

./serial data.txt 10000 128 10 10
9.805362e-03 1.992560e+00 5.000288e+00

./serial data.txt 10000 128 -10 -5
9.803682e-03 -1.990678e+00 -5.000364e+00

----------- CONCLUSION:
they are all equivalent, give the model.
*/

struct Args {
    char *fname;
    Params x0;
    long nsteps, M;
};

static void usage() {
    fprintf(stderr, "usage: ./serial <data.txt> nsteps M A0 W0\n");
    exit(2);
}

static bool shift(int *c, char ***v) {
    (*c)--; (*v)++;
    return (*c) > 0;
}

static void parse(int c, char **v, Args *a) {
    if (!shift(&c, &v)) usage();
    a->fname = *v;
    if (!shift(&c, &v)) usage();
    a->nsteps = atol(*v);
    if (!shift(&c, &v)) usage();
    a->M = atol(*v);
    if (!shift(&c, &v)) usage();
    a->x0.x[PAR_A] = atof(*v);
    if (!shift(&c, &v)) usage();
    a->x0.x[PAR_W] = atof(*v);
}

static void ini_T(int M, double *T) {
    for (int i = 0; i < M; ++i) T[i] = 1.0 / (M - i);
}

static void ini_E(const Data *data, int M, const Params *x, double *E) {
    for (int i = 0; i < M; ++i) E[i] = fitfun_eval(&x[i], data);
}

static void exchange(int M, const double *T, Params *x, double *E, double *w, Rng *gen) {
    /* TODO implement the selection and  exchange between chains */

    // compute weights:
    double weights[M-1];
    for (size_t i = 0; i < M-1; i++) {
      weights[i] = std::min(1.0, std::exp( (E[i]-E[i+1])*(1.0/T[i] - 1.0/T[i+1]) ));
    }

    // compte prefix sums:
    double pfs[M];
    pfs[0] = 0;
    for (size_t i = 1; i < M; i++) {
      pfs[i] = pfs[i-1] + weights[i-1];
    }

    // draw u:
    UnifDistr Unif(0., pfs[M-1]);
    double u = Unif(*gen);

    // find index i:
    size_t index = 0;
    for (size_t i = 0; i < M-1; i++) {
      if (u < pfs[i+1]) {
        index = i;
        break;
      }
    }

    //std::cout << "index: " << index << std::endl;

    // exchange i and i+1
    std::swap(x[index], x[index+1]);
    std::swap(E[index], E[index+1]);
}

static void get_best(int M, const Params *x, const double *E, Params *xbest, double *Ebest) {
    const Params *xb = &x[0];
    double Eb  = E[0];
    for (int i = 0; i < M; ++i) {
        if (E[i] < Eb) {
            Eb = E[i];
            xb = &x[i];
        }
    }
    *xbest = *xb;
    *Ebest = Eb;
}

static void print(const Params *x, double E, FILE *f) {
    fprintf(f, "%.6e %.6e %.6e\n", E, x->x[PAR_A], x->x[PAR_W]);
}

static void parallel_tempering(long nsteps, int M, const Params x0, const Data *data, FILE *trace) {
    int i, step;
    Params *x, xb, xbest;
    double *T, *E, *w, Eb, Ebest;
    Rng gen(42);
    x = (Params*) malloc(M * sizeof(Params));
    T = (double*) malloc(M * sizeof(double));
    E = (double*) malloc(M * sizeof(double));
    w = (double*) malloc(M * sizeof(double));

    for (i = 0; i < M; ++i) x[i] = x0;

    ini_T(M, T);
    ini_E(data, M, x, E);
    get_best(M, x, E, &xbest, &Ebest);

    for (step = 0; step < nsteps; ++step) {
        for (i = 0; i < M; ++i)
            mcmc_step(T[i], data, &gen, &x[i], &E[i]);

        exchange(M, T, x, E, w, &gen);
        get_best(M, x, E, &xb, &Eb);

        if (Eb < Ebest) {
            Ebest = Eb;
            xbest = xb;
        }

        //print(&xb, Eb, trace);
    }
    std::cout << "best:" << std::endl;
    print(&xbest, Ebest, stderr);

    free(x);
    free(E);
    free(T);
    free(w);
}

int main(int argc, char **argv) {


    Data data;
    Args a;
    FILE *f;

    parse(argc, argv, &a);

    fitfun_ini(a.fname, &data);

    f = stdout;
    parallel_tempering(a.nsteps, a.M, a.x0, &data, f);

    fitfun_fin(a.fname, &data);

    return 0;
}
