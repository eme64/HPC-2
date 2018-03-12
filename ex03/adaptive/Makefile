CC=g++
CFLAGS=-std=c++11 -O3 -fopenmp -Wall

.PHONY: all
all: adaptive_integrator_serial ai_p

adaptive_integrator_serial: adaptive_integrator_serial.cpp
	$(CC) $(CFLAGS) -o adaptive_integrator_serial adaptive_integrator_serial.cpp

ai_p: ai_p.cpp
	$(CC) $(CFLAGS) -o ai_p ai_p.cpp

.PHONY: clean
clean:
	rm -f adaptive_integrator_serial ai_p
