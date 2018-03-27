#include "markov_tree.hpp"

void markov_tree_q2()
{
    const double child_probits[3][3] = {
        [0] = {0., 0.5, 0.5},
        [1] = {0.1, 0.7, 0.2},
        [2] = {0.2, 0.7, 0.1}
    };
    const double exp_rate = 1. / 5.;
    unsigned short seed[] = {232, 15, 5};
    const int max_depth = 50;

    node_t * root = init_markov_root(exp_rate, seed);
    const int reached_depth = init_markov_tree(root, child_probits, exp_rate, 0, max_depth, seed);
    printf("%d %d\n", reached_depth, max_depth);

    // TODO:
    // Implement profiling 

    free_node(root);

}

int main(int argc, char const *argv[])
{
    markov_tree_q2();
    return 0;
}