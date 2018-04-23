CXX ?= g++
MPICXX?=mpicxx
CXXFLAGS = -O3 -Wall -std=c++11
LDFLAGS = -lm

PROGS = serial parallel
all: $(PROGS)

#%.o : %.cpp; $(CXX) $(CXXFLAGS) -o $@ -c $<

serial: serial.cpp fitfun.cpp mcmc.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	
parallel: parallel.cpp fitfun.cpp mcmc.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)


.PHONY: clean
clean:; rm -rf $(PROGS) *.o



