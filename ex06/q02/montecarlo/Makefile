CC=g++
CFLAGS=-std=c++11 -O3 -fopenmp -Wall -march=native

.PHONY: all
all: mc_serial mc_parallel mc_torc

mc_serial: mc_serial.cpp
	$(CC) $(CFLAGS) -o mc_serial mc_serial.cpp

mc_parallel: mc_parallel.cpp
	$(CC) $(CFLAGS) -o mc_parallel mc_parallel.cpp

mc_torc: mc_torc.cpp
	mpic++ `torc_cflags` -o mc_torc mc_torc.cpp `torc_libs` -std=c++11 -fopenmp

.PHONY: clean
clean:
	rm -f mc_serial mc_parallel mc_torc
