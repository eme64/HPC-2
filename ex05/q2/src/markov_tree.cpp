#include "markov_tree.hpp"

/*
    Sample Y ~ Exp(rate).

    Parameters
    ----------
        seed : unsigned short[3]
            Seed parameter to be passed to erand48.
        rate : double
            Parameter of the exponential distribution.

    Returns
    -------
        rand_exp : double
 */
static double rand_exp(unsigned short seed[3], const double rate) {
    const double y = erand48(seed);
    return -log(1. - y) / rate;
}

int sample_num_children(const double child_probabilities[3],
                        unsigned short seed[3])
{
   // TODO: implement.
   return 0;
}


node_t * init_markov_root(const double value_dist_exp_rate, unsigned short seed[3])
{
   // TODO: implement.
    return NULL;
}

int init_markov_tree(node_t * const node,
                     const double child_probabilities[3][3],
                     const double value_dist_exp_rate,
                     const int depth,
                     const int max_depth,
                     unsigned short seed[3])
{
    // TODO: implement
    return 0;
}

void max_sum_pass1(node_t * const root)
{
   // TODO: implement.
}

void max_sum_pass2(const node_t * const root, const int depth, int * const sequence)
{
   // TODO: implement.
}