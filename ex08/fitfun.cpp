#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fitfun.h"

static bool read_line(FILE *f, double *x, double *y) {
    return 2 == fscanf(f, "%lg %lg\n", x, y);
}

void fitfun_ini(const char *fname, Data *d) {
    FILE *f;
    double x, y;
    long i = 0;
    
    f = fopen(fname, "r");
    while (read_line(f, &x, &y)) {++i;}

    d->n = i;
    d->x = (double*) malloc(i * sizeof(double));
    d->y = (double*) malloc(i * sizeof(double));

    i = 0;
    rewind(f);
    while (read_line(f, &x, &y)) {
        d->x[i] = x;
        d->y[i] = y;
        ++i;
    }

    fclose(f);
}

void fitfun_fin(const char *fname, Data *d) {
    free(d->x);
    free(d->y);
    d->n = 0;
}

static double model(const Params *p, double x) {
    return p->x[PAR_A] * sin(p->x[PAR_W] * x);
}

double fitfun_eval(const Params *p, const Data *d) {
    double x, y, e, res = 0;
    long i;

    for (i = 0; i < d->n; ++i) {
        x = d->x[i];
        y = d->y[i];
        e = model(p, x) - y;
        res += e * e;
    }
    
    return res / d->n;
}
