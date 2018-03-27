#ifndef MARKOV_TREE_HPP
#define MARKOV_TREE_HPP

#include <cstdlib>
#include <cstring>
#include "markov_tree.hpp"
#include "node.hpp"
#include <random>
#include <cmath>

/*
    Generate the random number of children given the probabilities.

    Parameters
    ----------
        child_probabilities : const double[3]
            child_probabilities[i] = Prob[num_children = i].
            child_probabilities[i] >= 0 should hold, all i.
            sum_i child_probabilities[i] == 1. should hold.

        seed : unsigned short[3]
            Seed array to be used with erand48.

    Returns
    -------
        num_children : int
            Random variable, number of children distributed according
            to the passed probabilities. 0 <= num_children.
 */
int sample_num_children(const double cumsum_child_probabilities[2], unsigned short seed[3]);

/*
    Generate the root node of the Markov tree

    Parameters
    ----------
        value_dist_exp_rate : double
            Rate of the exponential distribution used to generate the
            node's value.

        seed : unsigned short seed[3]
            Seed passed to the erand48 function to generate random values.

    Returns
    -------
        root : node_t *
            Pointer to the newly created node or NULL if failed to allocate
            memory.        

 */
node_t * init_markov_root(const double value_dist_exp_rate, unsigned short seed[3]);


/*
    Generate the Markov tree according to child probabilities.

    Takes a passed node and generates the number of children. It 
    initializes the said number of children on the passed node,
    generates their value according to the passed exponential rate
    and calls itself on each of the children nodes. 

    If generated number of children is zero, 
    or if the maximum depth has been reached it returns the current depth.

    Parameters
    ----------
        node : node_t *
            Pointer to the node from which to descend the nodes, created
            using init_markov_root.

        child_probabilities : double[3][3]
            child_probabilities[i][j] = Prob[#children(node) = j | #children(parent(node)) = i].

        value_dist_exp_rate : double
            The rate of the exponential distribution, used to generate
            the node's value.

        depth : int
            Depth reached so far, used to control the recursion. 
            Pass 0.

        max_depth : int
            Maximum depth that is allowed to be reached. Used to prevent
            potentially infinite recursion.

        seed : unsigned short[3]
            Seed passed to the erand48 function to generate random values.

    Returns
    -------
        max_depth_reached : int
            The biggest depth that has been reached relative from
            the root node.

 */
int init_markov_tree(node_t * const node,
                     const double child_probabilities[3][3],
                     const double value_dist_exp_rate,
                     const int depth,
                     const int max_depth,
                     unsigned short seed[3]);

/*
    Perform the first pass of the maximum path-sum algorithm.

    root->value should be set to the maximum path-sum after the 
    completion of the routine.

    Parameters
    ----------
        root : node_t *
            Pointer to the node from which to start.
 */
void max_sum_pass1(node_t * const root);

/*
    Perform the second pass of the maximum path-sum algorithm.
    Should be called after max_sum_pass1 has been executed. Otherwise
    the behavior is undefined.

    After the routine is done, sequence should hold the optimal moves at
    each depth of the graph:

    sequence[0] = i means that from root (0th node) we go to the i'th
    child. If i == 0, it means termination. 
    sequence[1] = j means we then go to the j'th child of i'th child of 
    the root, and so on.

    Parameters
    ----------
        root : node_t *
            Pointer to the node from which to start.

        depth : int
            Current depth of the algorithm in the graph, used 
            to control recursion. Pass 0 on first call.

        sequence : int *
            Buffer that stores the optimal moves in the graph.

 */
void max_sum_pass2(const node_t * const root, const int depth, int * const sequence);

#endif //MARKOV_TREE_HPP