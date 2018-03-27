BINDIR=bin
SRCDIR=src
INCLUDEDIR=include
SOURCES=$(notdir $(wildcard $(SRCDIR)/*.cpp))
BINARIES=$(addprefix $(BINDIR)/, $(SOURCES:%.cpp=%.o))
CXX=g++

CXXFLAGS=\
-std=c++14 \
-Wall \
-march=native \
-g \
-O3 \
-I./$(INCLUDEDIR) \
-D_ISOC11_SOURCE \
-fopenmp\

LDFLAGS=\
-lgomp \
-lm \

all: main test

test: $(BINARIES) test.cpp
	$(CXX) $(CXXFLAGS) -c $@.cpp
	$(CXX) $(BINARIES) $@.o -o $@ $(LDFLAGS)

main: $(BINARIES) main.cpp
	$(CXX) $(CXXFLAGS) -c $@.cpp
	$(CXX) $(BINARIES) $@.o -o $@ $(LDFLAGS)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp $(INCLUDEDIR)/%.hpp $(BINDIR)
	$(CXX) $(CXXFLAGS) -c $(filter %.cpp, $^) -o $@

$(BINDIR):
	mkdir $@

clean:
	rm -rf main test $(BINDIR)
