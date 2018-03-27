#include "markov_tree.hpp"
#include <omp.h>
#include <iostream>

/*
comment about parallelizing pass2:
it could be done in theory, but then one would generate multiple arrays.
better would be to change the child pointer in pass one to the biggest one.
then one could simply read off the value.
for now I will not parallelize pass2.


*/

void markov_tree_q2()
{
  const int num_iterations = 10;
  double time_init = 0;
  double time_calc = 0;

  int i = 0;
  while(i<100){
    const int num_threads = omp_get_num_threads();

    const double child_probits[3][3] = {
        [0] = {0., 0.5, 0.5},
        [1] = {0.1, 0.7, 0.2},
        [2] = {0.2, 0.7, 0.1}
    };
    const double exp_rate = 1. / 5.;
    unsigned short seed[SEED_ARRAY_SIZE][3];// = {{232, 15, 5}};
    for (size_t i = 0; i < SEED_ARRAY_SIZE; i++) {
      seed[i][0] = 232+i;
      seed[i][1] = 15;
      seed[i][2] = 5;
    }
    const int max_depth = 190; // 50;

    //std::cout << "init tree:" << std::endl;
    double t0 = omp_get_wtime();
    node_t * root = init_markov_root(exp_rate, seed);

    int reached_depth;

    #pragma omp parallel
    {
      #pragma omp single nowait
      {
        #pragma omp task
        {
          reached_depth = init_markov_tree(root, child_probits, exp_rate, 0, max_depth, seed);
        }
      }
    }

    //reached_depth = init_markov_tree(root, child_probits, exp_rate, 0, max_depth, seed);

    double t1 = omp_get_wtime();
    //printf("%d %d\n", reached_depth, max_depth);


    //std::cout << "time for construction: " << t1-t0 << std::endl;
    //print_tree(root, 0);


    // max sum passes:
    double t2 = omp_get_wtime();

    #pragma omp parallel
    {
      #pragma omp single nowait
      {
        #pragma omp task
        {
          max_sum_pass1(root);
        }
      }

      #pragma omp barrier

      int sequence[max_depth] = {0};
      max_sum_pass2(root, 0, sequence);
    }
    double t3 = omp_get_wtime();

    // something is off with the random calc, I guess...
    // therefore I only take the big fish...

    if (reached_depth == max_depth) {
      time_init+= t1-t0;
      time_calc+= t3-t2;
      i++;
    }
    //std::cout << "time for max_path: " << t3-t2 << std::endl;

    free_node(root);
  }

  std::cout << "time for init: " << time_init << std::endl;
  std::cout << "time for max_path: " << time_calc << std::endl;
}

int main(int argc, char const *argv[])
{
    markov_tree_q2();
    return 0;
}
