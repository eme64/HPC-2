CXX      ?= g++
CXXFLAGS = -std=c++11 -O3 -fopenmp -Wall
LDFLAGS  = -lm

all: serial parallel

serial: serial.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

parallel: parallel.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) -fopenmp


%.o: %.cpp	
	$(CXX) $(CXXFLAGS) -o $@ -c $^

.PHONY: clean

clean: ; rm -rf *.o serial
