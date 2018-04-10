CC=g++
CFLAGS=-std=c++11 -O3 -fopenmp -Wall -march=native

.PHONY: all
all: adaptive_integrator_serial adaptive_integrator_parallel

adaptive_integrator_serial: adaptive_integrator_serial.cpp
	$(CC) $(CFLAGS) -o adaptive_integrator_serial adaptive_integrator_serial.cpp

adaptive_integrator_parallel: adaptive_integrator_parallel.cpp
	$(CC) $(CFLAGS) -o adaptive_integrator_parallel adaptive_integrator_parallel.cpp

.PHONY: clean
clean:
	rm -f adaptive_integrator_serial adaptive_integrator_parallel
