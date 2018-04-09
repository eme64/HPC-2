# Makefile

SHELL := /bin/bash
CXX?=g++
CXXFLAGS?= -Wall -O3 -std=c++11 -lstdc++
MPICXX?=mpicxx

.DEFAULT: all

.PHONY: all
all: diffusion_serial #diffusion_mpi

diffusion_serial: diffusion2d_serial.cpp
	$(CXX) $(CXXFLAGS) -o diffusion_serial diffusion2d_serial.cpp


.PHONY: clean
clean:
	-rm diffusion_serial
