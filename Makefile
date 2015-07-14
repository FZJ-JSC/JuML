CC = mpicxx
INCLUDE_DIRS = -Iinclude
CCFLAGS = -O3 -std=c++11 $(INCLUDE_DIRS)
LDFLAGS = -fopenmp

all:
	$(CC) $(CCFLAGS) -c src/stats/Distributions.cpp src/classification/GaussianNaiveBayes.cpp

clean:
	rm *.o
