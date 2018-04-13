CC=g++
CFLAGS=-std=c++11 -O3 -fopenmp -Wall -march=native

.PHONY: all
all: adaptive_integrator_serial adaptive_integrator_parallel adaptive_integrator_torc

adaptive_integrator_serial: adaptive_integrator_serial.cpp
	$(CC) $(CFLAGS) -o adaptive_integrator_serial adaptive_integrator_serial.cpp

adaptive_integrator_parallel: adaptive_integrator_parallel.cpp
	$(CC) $(CFLAGS) -o adaptive_integrator_parallel adaptive_integrator_parallel.cpp

adaptive_integrator_torc: adaptive_integrator_torc.cpp
	mpic++ `torc_cflags` -o adaptive_integrator_torc adaptive_integrator_torc.cpp `torc_libs` -std=c++11 -fopenmp

.PHONY: clean
clean:
	rm -f adaptive_integrator_serial adaptive_integrator_parallel
