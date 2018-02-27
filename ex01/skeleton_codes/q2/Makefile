# Makefile for HPCSE I (2017HS) Exam, 22.12.2017
# Prof. P. Koumoutsakos, Dr. P. Hadjidoukas
# Coding 2 : Diffusion Statistics
#
# Task: complete the compile command

all: diffusion2d_mpi diffusion2d_hybrid

diffusion2d_mpi: diffusion2d_mpi.cpp timer.hpp
	mpic++ -O3 -o $@ $<

diffusion2d_hybrid: diffusion2d_hybrid.cpp timer.hpp
	# TODO: fill in

.PHONY: clean
clean:
	-rm -f diffusion2d_mpi diffusion2d_hybrid
