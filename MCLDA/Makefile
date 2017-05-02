CC = g++
CXXFLAGS = -std=c++11 -pedantic -Wall -Wextra -O -g -w
GPROF = -pg
GCOV = -fprofile-arcs -ftest-coverage

all: CollapsedSamplerMain
	
# bin: CollapsedSampler

test: csTest
	echo "Running tests..."
	./csTest

csTest: CollapsedSamplerTest.o CollapsedSampler.o Example.o
	$(CC) $(CXXFLAGS) CollapsedSamplerTest.o CollapsedSampler.o Example.o -o csTest
	
CollapsedSamplerMain: CollapsedSamplerMain.o CollapsedSampler.o Example.o
	$(CC) $(CXXFLAGS) CollapsedSamplerMain.o CollapsedSampler.o Example.o -o CollapsedSamplerMain

# CollapsedSampler: CollapsedSampler.o Example.o
# 	$(CC) $(CXXFLAGS) CollapsedSampler.o Example.o -o CollapsedSampler
	
CollapsedSampler.o: CollapsedSampler.cpp
	$(CC) $(CXXFLAGS) -c CollapsedSampler.cpp
	
Example.o: Example.cpp
	$(CC) $(CXXFLAGS) -c Example.cpp
	
clean:
	rm -f a.out *~ *.o *.gcov *.gclsda *.gcno gmon.out *.gch gibbssampler