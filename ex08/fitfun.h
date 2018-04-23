struct Data {
    double *x, *y;
    long n;
};

enum {
    PAR_A,
    PAR_W,
    NPARAMS
};

struct Params {
    double x[NPARAMS];
};

void fitfun_ini(const char *fname, Data *d);
void fitfun_fin(const char *fname, Data *d);
double fitfun_eval(const Params *p, const Data *d);
