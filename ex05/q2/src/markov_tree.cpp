#include "markov_tree.hpp"
#include <iostream>
#include <algorithm>
#include <omp.h>


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
static double rand_exp(unsigned short seed[SEED_ARRAY_SIZE][3], const double rate) {
    const double y = erand48(seed[omp_get_thread_num()*SEED_ARRAY_OFFSETS]);
    return -log(1. - y) / rate;
}

int sample_num_children(const double child_probabilities[3],
                        unsigned short seed[SEED_ARRAY_SIZE][3])
{
   double rnd = erand48(seed[omp_get_thread_num()*SEED_ARRAY_OFFSETS]);
   double sum = 0;
   for (int i = 0; i < 3; i++) {
     if (rnd <= sum+child_probabilities[i]) {
       return i;
     }
     sum+=child_probabilities[i];
   }
   std::cout << "sample_num_children: error over !" << std::endl;
   return 2;
}


node_t * init_markov_root(const double value_dist_exp_rate, unsigned short seed[SEED_ARRAY_SIZE][3])
{
  /*
  node_t* root = new node_t {rand_exp(seed, value_dist_exp_rate), NULL};
  if(root == NULL){return NULL;}

  // default values:
  root->child = NULL;
  root->sibling_left = NULL;
  root->sibling_right = NULL;
  root->num_children = 0;
  */
  node_t* root = init_node(rand_exp(seed, value_dist_exp_rate), NULL);

  return root;
}

int init_markov_tree(node_t * const node,
                     const double child_probabilities[3][3],
                     const double value_dist_exp_rate,
                     const int depth,
                     const int max_depth,
                     unsigned short seed[SEED_ARRAY_SIZE][3])
{
  int parent_num_c = 2;
  if (node->parent != NULL) {
    parent_num_c = node->parent->num_children;
  }
  int n_children = sample_num_children(child_probabilities[parent_num_c], seed);

  if (n_children == 0 || depth >= max_depth) {
    return depth;
  }

  int depth_bellow[3] = {depth, depth, depth};

  for (size_t i = 0; i < n_children; i++) {
    /*
    node_t* c = new node_t {rand_exp(seed, value_dist_exp_rate), node};
    if(c == NULL)
    {
      std::cout << "init_markov_tree: new not sucessful" << std::endl;
      return 0;
    }

    if (node->child != NULL) {
      // I am first child:
      c->child = NULL;
      c->sibling_left = NULL;
      c->sibling_right = NULL;
      c->num_children = 0;

      node->child = c;
    }else{
      // already have a child, attach left of it:
      c->child = NULL;
      c->sibling_left = NULL;
      c->sibling_right = node->child; // sibling to the right
      node->child->sibling_left = c;
      c->num_children = 0;

      node->child = c; // the last inserted child
    }
    */

    #pragma omp task untied shared(depth_bellow) final(max_depth - depth < 10 && i==0)
    {
      node_t* c = init_node(rand_exp(seed, value_dist_exp_rate), node);
      depth_bellow[i] = init_markov_tree(c, child_probabilities, value_dist_exp_rate, depth+1, max_depth, seed);
    }
  }
  #pragma omp taskwait

  int depth_reached_below = depth;
  for (size_t i = 0; i < n_children; i++) {
    depth_reached_below = std::max(
      depth_reached_below,
      depth_bellow[i]
    );
  }

  return depth_reached_below;
}

double max_path_sum(node_t * const node)
{
  // assume node is valid
  double max_sum = node->value;

  node_t * c = node->child; // first one
  while(c != NULL)
  {
    // process c:
    max_sum = std::max(
      max_sum,
      node->value + max_path_sum(c)
    );

    // find next c:
    c = c->sibling_right;
  }

  return max_sum;
}

void max_sum_pass1(node_t * const root)
{
   if(root != NULL)
   {
     double max_sum = root->value;

     double max_below[3] = {0};

     node_t * c = root->child; // first one
     int i = 0;
     while(c != NULL)
     {
       // process c:
       #pragma omp task shared(max_below) final(i == 0)
       {
         max_below[i] = max_path_sum(c);
       }
       /*
       max_sum = std::max(
         max_sum,
         root->value + max_path_sum(c)
       );
       */

       // find next c:
       c = c->sibling_right;
       i++;
     }

     #pragma omp taskwait

     for (size_t i = 0; i < 3; i++) {
       max_sum = std::max(
         max_sum,
         root->value + max_below[i]
       );
     }

     root->value = max_sum;
   }else{
     std::cout << "root not valid!" << std::endl;
     return;
   }
}

void max_sum_pass2(const node_t * const root, const int depth, int * const sequence)
{
  node_t * max_c = NULL; // best one
  double max_found = 0;
  int max_i = -1;

  node_t * c = root->child; // first one
  int i = 0;
  while(c != NULL)
  {
    // process c:

    if (c->value > max_found) {
      max_found = c->value;
      max_c = c;
      max_i = i;
    }

    // find next c:
    c = c->sibling_right;
    i++;
  }

  if (max_c != NULL) {
    // go deeper:
    sequence[depth] = max_i;
    max_sum_pass2(max_c, depth+1, sequence);
  }else{
    // DONE
    sequence[depth] = 0;
    return;
  }
}
