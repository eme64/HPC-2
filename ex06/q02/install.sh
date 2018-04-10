#!/bin/bash

echo "--- Installing TORC on EULER ---"

module load gcc
module load mvapich2

cd torc_lite_hpcse
./configure --prefix=$HOME/usr/torc CC=mpicc F77=mpif90

make
make install

export PATH=$PATH:$HOME/usr/torc/bin

echo "### Installing TORC on EULER ###"

