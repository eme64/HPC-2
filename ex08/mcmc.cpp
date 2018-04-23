#include <random>

#include "fitfun.h"
#include "mcmc.h"

static void proposal(const Params *p0, Rng *gen, Params *p) {
    NormalDistr G(0, 0.1);
    for (int i = 0; i < NPARAMS; ++i)
    p->x[i] = p0->x[i] + G(*gen);
}

static bool accept(double T, double E, double Ep, Rng *gen) {
    UnifDistr Unif(0., 1.);
    double dE, exp_prob, u = Unif(*gen);
    dE = Ep - E;
    exp_prob = exp(-dE / T);
    return u < exp_prob;
}

void mcmc_step(double T, const Data *data, Rng *gen, Params *x, double *E) {
    Params xp;
    double Ep;
        
    proposal(x, gen, &xp);
    Ep = fitfun_eval(&xp, data);
    
    if (accept(T, *E, Ep, gen)) {
        *x = xp;
        *E = Ep;
    }
}
