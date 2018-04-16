BINDIR=bin
SRCDIR=src
INCLUDEDIR=include
SOURCES=$(notdir $(wildcard $(SRCDIR)/*.cpp))
BINARIES=$(addprefix $(BINDIR)/, $(SOURCES:%.cpp=%.o))
CXX=icpc
NTHREADS=4

CXXFLAGS=\
-std=c++14 \
-Wall \
-march=native \
-g \
-O1 \
-I./$(INCLUDEDIR) \
-D_ISOC11_SOURCE \
-D_NTHREADS=$(NTHREADS) \
-DMKL_ILP64 \
-I$(INTEL_ROOT)/compiler/include \
-I$(INTEL_ROOT)/mkl/include \
-qopenmp

LDFLAGS=\
-L $(INTEL_ROOT)/$(INTEL_LIB) \
-L $(INTEL_ROOT)/$(INTEL_MKLLIB) \
-Wl,-rpath,$(INTEL_ROOT)/$(INTEL_LIB) \
-Wl,-rpath,$(INTEL_ROOT)/$(INTEL_MKLLIB) \
-lmkl_intel_ilp64 \
-lmkl_intel_thread \
-lmkl_core \
-liomp5 \
-lpthread \
-lm \
-ldl

all: main

$(BINDIR):
	mkdir $@

main: $(BINARIES) main.cpp
	$(CXX) $(CXXFLAGS) -c $@.cpp
	$(CXX) $(BINARIES) $@.o -o $@ $(LDFLAGS)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp $(INCLUDEDIR)/%.hpp $(BINDIR)
	$(CXX) $(CXXFLAGS) -c $(filter %.cpp, $^) -o $@

clean:
	rm -rf main main.o bin/*
