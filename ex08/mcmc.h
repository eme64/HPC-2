typedef std::default_random_engine Rng;
typedef std::normal_distribution<double>        NormalDistr;
typedef std::uniform_real_distribution<double>  UnifDistr;

void mcmc_step(double T, const Data *data, Rng *gen, Params *x, double *E);
