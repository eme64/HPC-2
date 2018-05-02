
NVCC = nvcc

NVCCFLAGS = -O3 -lineinfo -arch=compute_60 -code=sm_60 -std=c++11 -Xcompiler "-fopenmp"
NVCCFLAGS += --use_fast_math

LDFLAGS = -Xcompiler "-fopenmp"

all: nbody

nbody: main.o
	$(NVCC) $(LDFLAGS) $^ -o $@

main.o: main.cu gravity.h containers.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


clean:
	rm -f *.o nbody

.PHONY: clean 
