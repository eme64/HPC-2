CXX      ?= gcc
CXXFLAGS = -Wall -O3 -std=c++11
LDFLAGS  = -lm

all: serial

serial: serial.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)


%.o: %.cpp	
	$(CXX) $(CXXFLAGS) -o $@ -c $^

.PHONY: clean

clean: ; rm -rf *.o serial
