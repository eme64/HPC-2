BINDIR=bin
SRCDIR=src
INCLUDEDIR=include
SOURCES=$(notdir $(wildcard $(SRCDIR)/*.c))
BINARIES=$(addprefix $(BINDIR)/, $(SOURCES:%.c=%.o))
NTHREADS=4
CC=gcc
CFLAGS=-std=c11 -Wall -march=native -O3 -I./$(INCLUDEDIR) -fopenmp\
-D_NTHREADS=$(NTHREADS)
LDFLAGS=-lm -lgomp

all: main

main: $(BINARIES) main.c
	$(CC) $(CFLAGS) -c main.c
	$(CC) $(BINARIES) $@.o -o $@ $(LDFLAGS)

$(BINDIR)/%.o: $(SRCDIR)/%.c $(INCLUDEDIR)/%.h
	$(CC) $(CFLAGS) -c $(filter %.c, $^) -o $@ $(LDFLAGS)

clean:
	rm -rf main main.o bin/*
