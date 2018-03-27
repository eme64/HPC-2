#include "markov_tree.hpp"
#include <iostream>
#include <algorithm>
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
   double rnd = erand48(seed);

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


node_t * init_markov_root(const double value_dist_exp_rate, unsigned short seed[3])
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
                     unsigned short seed[3])
{
  int parent_num_c = 2;
  if (node->parent != NULL) {
    parent_num_c = node->parent->num_children;
  }
  int n_children = sample_num_children(child_probabilities[parent_num_c], seed);

  if (n_children == 0 || depth >= max_depth) {
    return depth;
  }

  int depth_reached_below = depth;

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
    node_t* c = init_node(rand_exp(seed, value_dist_exp_rate), node);

    depth_reached_below = std::max(
      depth_reached_below,
      init_markov_tree(c, child_probabilities, value_dist_exp_rate, depth+1, max_depth, seed)
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

     node_t * c = root->child; // first one
     while(c != NULL)
     {
       // process c:
       max_sum = std::max(
         max_sum,
         root->value + max_path_sum(c)
       );

       // find next c:
       c = c->sibling_right;
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
