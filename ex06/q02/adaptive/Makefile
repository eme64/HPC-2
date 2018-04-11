CC=g++
CFLAGS=-std=c++11 -O3 -fopenmp -Wall -march=native

.PHONY: all
all: adaptive_integrator_serial adaptive_integrator_parallel adaptive_integrator_torc

adaptive_integrator_serial: adaptive_integrator_serial.cpp
	$(CC) $(CFLAGS) -o adaptive_integrator_serial adaptive_integrator_serial.cpp

adaptive_integrator_parallel: adaptive_integrator_parallel.cpp
	$(CC) $(CFLAGS) -o adaptive_integrator_parallel adaptive_integrator_parallel.cpp

adaptive_integrator_torc: adaptive_integrator_torc.cpp
	export PATH=$PATH:$/home/Documents/HPC-2/ex06/q02/torc_lite_hpcse/bin
	mpicc 'torc_cflags' -o adaptive_integrator_torc adaptive_integrator_torc.cpp 'torc_libs'

.PHONY: clean
clean:
	rm -f adaptive_integrator_serial adaptive_integrator_parallel