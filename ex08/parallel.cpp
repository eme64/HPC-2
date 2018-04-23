#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

#include <mpi.h>

#include "fitfun.h"
#include "mcmc.h"


/*
------------ RESULTS:

[0] M_local: 120, M_offset: 0
time: 43.1551
best:
9.803577e-03 1.991351e+00 5.000387e+00

[[3] M_local: 30, M_offset: 90
[1] M_local: 30, M_offset: 30
2] M_local: 30, M_offset: 60
[0] M_local: 30, M_offset: 0
time: 10.0858
best:
9.809257e-03 5.000679e+00 1.095105e-02

[5[6] M_local: 15, M_offset: 90
[7] M_local: 15, M_offset: 105
] M_local: 15, M_offset: 75
[2] M_local: 15, M_offset: 30
[3] M_local: 15, M_offset: 45
[1] M_local: 15, M_offset: 15
[0] M_local: 15, M_offset: 0
[4] M_local: 15, M_offset: 60
time: 5.30539
best:
9.805452e-03 5.000500e+00 4.554640e+00

[5] M_local: 10, M_offset: 50[6] M_local: 10, M_offset: 60
[9] M_local: 10, M_offset: 90
[2] M_local: 10, M_offset: 20
[3] M_local: 10, M_offset: 30
[0] M_local: 10, M_offset: 0
[11] M_local: 10, M_offset: 110
[4] M_local: 10, M_offset: 40
[1] M_local: 10, M_offset: 10
[8] M_local: 10, M_offset: 80
[7] M_local: 10, M_offset: 70
[10] M_local: 10, M_offset: 100
time: 3.60228
best:
9.834330e-03 4.999987e+00 2.173139e-01

----------- CONCLUSION:

scaling is quite good. The results are a bit off though... (for P>1 threads)
There was a typo, now it should all be fine.
Yes, confirmed, see log_12_2.txt

*/

struct Args {
    char *fname;
    Params x0;
    long nsteps, M;
};

struct MPI_Args
{
  int size;
  int rank;

  int M_local;
  int M_offset;
};

static void usage() {
    fprintf(stderr, "usage: ./serial <data.txt> nsteps M A0 W0\n");
    exit(2);
}

static bool shift(int *c, char ***v) {
    (*c)--; (*v)++;
    return (*c) > 0;
}

static void parse(int c, char **v, Args *a) {
    if (!shift(&c, &v)) usage();
    a->fname = *v;
    if (!shift(&c, &v)) usage();
    a->nsteps = atol(*v);
    if (!shift(&c, &v)) usage();
    a->M = atol(*v);
    if (!shift(&c, &v)) usage();
    a->x0.x[PAR_A] = atof(*v);
    if (!shift(&c, &v)) usage();
    a->x0.x[PAR_W] = atof(*v);
}

static void ini_T(int M, double *T, MPI_Args mpia) {
    for (int i = 0; i < mpia.M_local+1; ++i) T[i] = 1.0 / (M - i - mpia.M_offset);
}

static void ini_E(const Data *data, int M, const Params *x, double *E, MPI_Args mpia) {
    for (int i = 0; i < mpia.M_local; ++i) E[i] = fitfun_eval(&x[i], data);
}

static void exchange(int M, const double *T, Params *x, double *E, double *w, Rng *gen, MPI_Args mpia) {
    // get/send missing E:
    double E_missing;
    MPI_Request req;

    if (mpia.rank+1 < mpia.size) {
      MPI_Irecv(&E_missing, 1, MPI_DOUBLE, mpia.rank+1, 111, MPI_COMM_WORLD, &req);
    }else{
      req = MPI_REQUEST_NULL;
    }

    if (mpia.rank > 0) {
      MPI_Send(&E[0], 1, MPI_DOUBLE, mpia.rank-1, 111, MPI_COMM_WORLD);
    }

    MPI_Wait(&req, MPI_STATUS_IGNORE);

    // compute weights:
    double weights[mpia.M_local]; // highest rank will have one too many
    for (size_t i = 0; i < mpia.M_local-1; i++) {
      weights[i] = std::min(1.0, std::exp( (E[i]-E[i+1])*(1.0/T[i] - 1.0/T[i+1]) ));
    }
    if (mpia.rank+1 < mpia.size) {
      weights[mpia.M_local-1] = std::min(1.0, std::exp( (E[mpia.M_local-1]-E_missing)*(1.0/T[mpia.M_local-1] - 1.0/T[mpia.M_local]) )); // do last by hand.
    }

    // compte prefix sums:
    double pfs[mpia.M_local+1]; // highest rank will have one too many
    pfs[0] = 0;
    for (size_t i = 1; i < mpia.M_local+1; i++) {
      pfs[i] = pfs[i-1] + weights[i-1];
    }

    // get prefix sum of previous ranks:
    double rcv_data[mpia.size*2];
    double send_data[2];
    send_data[0] = pfs[mpia.M_local-1]; // needed for next rank to find out if hits
    send_data[1] = pfs[mpia.M_local]; // for finding rank, total sum
    MPI_Allgather(send_data, 2, MPI_DOUBLE, rcv_data, 2, MPI_DOUBLE, MPI_COMM_WORLD);

    // get total sum of weights:
    double rank_total_sum = 0;
    double rank_prefix_sum = 0;
    for (size_t i = 0; i < mpia.size; i++) {
      rank_total_sum+= rcv_data[i*2 + 1];

      if (i<mpia.rank) {
        rank_prefix_sum+= rcv_data[i*2 + 1];
      }
    }

    // draw u:
    UnifDistr Unif(0., rank_total_sum);
    double u = Unif(*gen);

    // find rank that holds index i:
    double temp_sum = 0;
    size_t index_rank = 0;
    double u_local = u;
    for (size_t i = 0; i < mpia.size; i++) {
      temp_sum+= rcv_data[i*2 + 1];
      if (u < temp_sum) {
        index_rank = i;
        break;
      }
      u_local-= rcv_data[i*2 + 1];
    }

    //std::cout << "[" << mpia.rank << "]index_rank: " << index_rank << std::endl;

    if (index_rank == mpia.rank) {
      // find index i:
      size_t index = 0;
      for (size_t i = 0; i < mpia.M_local; i++) {
        if (u_local < pfs[i+1]) {
          index = i;
          break;
        }
      }

      if (index == mpia.M_local-1) {
        // exchange with rank+1
        //std::cout << "[" << mpia.rank << "]exchange up: " << index_rank << ", index: " << index << std::endl;

        double rcv_data[3];
        MPI_Request req;
        MPI_Irecv(rcv_data, 3, MPI_DOUBLE, mpia.rank+1, 333, MPI_COMM_WORLD, &req);

        double send_data[3];
        send_data[0] = E[mpia.M_local-1];
        send_data[1] = x[mpia.M_local-1].x[PAR_A];
        send_data[2] = x[mpia.M_local-1].x[PAR_W];
        MPI_Send(send_data, 3, MPI_DOUBLE, mpia.rank+1, 222, MPI_COMM_WORLD);

        MPI_Wait(&req, MPI_STATUS_IGNORE);
        E[mpia.M_local-1] = rcv_data[0];
        x[mpia.M_local-1].x[PAR_A] = rcv_data[0];
        x[mpia.M_local-1].x[PAR_W] = rcv_data[0];

      }else{
        // local swap.
        //std::cout << "[" << mpia.rank << "]local: " << index << ", index_rank: " << index_rank << std::endl;

        // exchange i and i+1
        std::swap(x[index], x[index+1]);
        std::swap(E[index], E[index+1]);
      }
    }else if (index_rank == mpia.rank-1) {
      // might hit the first element:
      if (u_local > rcv_data[index_rank*2]) {
        // exchange with rank-1:
        //std::cout << "[" << mpia.rank << "]exchange down: " << index_rank << ", index: -1" << std::endl;

        double rcv_data[3];
        MPI_Request req;
        MPI_Irecv(rcv_data, 3, MPI_DOUBLE, mpia.rank-1, 222, MPI_COMM_WORLD, &req);

        double send_data[3];
        send_data[0] = E[0];
        send_data[1] = x[0].x[PAR_A];
        send_data[2] = x[0].x[PAR_W];
        MPI_Send(send_data, 3, MPI_DOUBLE, mpia.rank-1, 333, MPI_COMM_WORLD);

        MPI_Wait(&req, MPI_STATUS_IGNORE);
        E[0] = rcv_data[0];
        x[0].x[PAR_A] = rcv_data[0];
        x[0].x[PAR_W] = rcv_data[0];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

static void get_best(int M, const Params *x, const double *E, Params *xbest, double *Ebest, MPI_Args mpia) {
    Params xb = x[0];
    double Eb  = E[0];
    for (int i = 0; i < mpia.M_local; ++i) {
        if (E[i] < Eb) {
            Eb = E[i];
            xb = x[i];
        }
    }

    if (mpia.rank == 0) {
      // find global best:
      double rcv_data[mpia.size*3];
      double send_data[3];
      send_data[0] = Eb;
      send_data[1] = xb.x[PAR_A];
      send_data[2] = xb.x[PAR_W];
      MPI_Gather(send_data, 3, MPI_DOUBLE, rcv_data, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // find best of gathered:
      for (size_t i = 0; i < mpia.size; i++) {
        if (rcv_data[3*i] < Eb) {
            Eb = rcv_data[3*i];
            xb.x[PAR_A] = rcv_data[3*i+1];
            xb.x[PAR_W] = rcv_data[3*i+2];
        }
      }
    }else{
      double send_data[3];
      send_data[0] = Eb;
      send_data[1] = xb.x[PAR_A];
      send_data[2] = xb.x[PAR_W];

      MPI_Gather(send_data, 3, MPI_DOUBLE, NULL, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    *xbest = xb;
    *Ebest = Eb;
}

static void print(const Params *x, double E, FILE *f) {
    fprintf(f, "%.6e %.6e %.6e\n", E, x->x[PAR_A], x->x[PAR_W]);
}

static void parallel_tempering(long nsteps, int M, const Params x0, const Data *data, FILE *trace, MPI_Args mpia) {
    int i, step;
    Params *x, xb, xbest;
    double *T, *E, *w, Eb, Ebest;
    Rng gen(42);
    x = (Params*) malloc(mpia.M_local * sizeof(Params));
    T = (double*) malloc((mpia.M_local+1) * sizeof(double));
    E = (double*) malloc(mpia.M_local * sizeof(double));
    w = (double*) malloc(mpia.M_local * sizeof(double));

    for (i = 0; i < mpia.M_local; ++i) x[i] = x0;

    ini_T(M, T, mpia);
    ini_E(data, M, x, E, mpia);
    get_best(M, x, E, &xbest, &Ebest, mpia);

    double t0 = MPI_Wtime();

    for (step = 0; step < nsteps; ++step) {
        for (i = 0; i < mpia.M_local; ++i)
            mcmc_step(T[i], data, &gen, &x[i], &E[i]);

        exchange(M, T, x, E, w, &gen, mpia);
        get_best(M, x, E, &xb, &Eb, mpia);

        if (Eb < Ebest) {
            Ebest = Eb;
            xbest = xb;
        }

        if (mpia.rank == 0) {
          //print(&xb, Eb, trace);
        }
    }

    double t1 = MPI_Wtime();

    if (mpia.rank == 0) {
      std::cout << "time: " << (t1-t0) << std::endl;
      std::cout << "best:" << std::endl;
      print(&xbest, Ebest, stderr);
    }

    free(x);
    free(E);
    free(T);
    free(w);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world_size;
  int world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


  Data data;
  Args a;
  FILE *f;
  MPI_Args mpia;

  parse(argc, argv, &a);

  fitfun_ini(a.fname, &data);

  // make MPI arguments ready
  mpia.size = world_size;
  mpia.rank = world_rank;
  mpia.M_local = a.M/mpia.size;
  mpia.M_offset = mpia.rank * mpia.M_local;
  if (mpia.rank == mpia.size-1) {
    mpia.M_local += a.M % mpia.size;
  }
  std::cout << "[" << mpia.rank <<  "] M_local: " << mpia.M_local << ", M_offset: " << mpia.M_offset << std::endl;

  f = stdout;
  parallel_tempering(a.nsteps, a.M, a.x0, &data, f, mpia);

  fitfun_fin(a.fname, &data);

  MPI_Finalize();

  return 0;
}
