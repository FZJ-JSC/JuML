// vim: expandtab:shiftwidth=4:softtabstop=4
#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <armadillo>
#include "svm/KernelCache.h"

class KernelEvaluationCounter {
	public:
	const size_t n;
	arma::Mat<unsigned int> counts;
	KernelEvaluationCounter(size_t n_) : n(n_), counts(n,n, arma::fill::zeros) {}

	float evaluate_kernel(int i, int j) {
		counts(i,j)++;
		return 1.0;
	}
};

TEST (KernelCacheTest, TestKernelEvaluationCounter) {
    const int N = 5;
    KernelEvaluationCounter counter(N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ASSERT_EQ(0, counter.counts(i,j));
        }
    }

    counter.evaluate_kernel(0,0);
    for (int i = 0; i < 3; i++) {
        counter.evaluate_kernel(1,3);
    }
    counter.evaluate_kernel(3,1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int expected;
            if (i == 0 && j == 0) {
                ASSERT_EQ(1, counter.counts(i,j));
            } else if (i == 1 && j == 3) {
                ASSERT_EQ(3, counter.counts(i,j));
            } else if (i == 3 && j == 1) {
                ASSERT_EQ(1, counter.counts(i,j));
            } else {
                ASSERT_EQ(0, counter.counts(i,j));
            }
        }
    }
}

TEST (KernelCacheTest, TestCachedKernelWithoutCacheDeletion) {
    const int N = 5;
    KernelEvaluationCounter counter(N);
    //Enough Space to cache the full Kernel
    auto cachedKernel = juml::svm::KernelCache<KernelEvaluationCounter>(counter, sizeof(float) * N * N * 8, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ASSERT_EQ(0, counter.counts(i,j));
        }
    }

    std::vector<unsigned int> idxs = {1, 2, 3};

    for (int z = 0; z < 5; z++ ) {
        cachedKernel.get_col(0, idxs);
        for (int i = 0; i < N; i++) {
            if (i == 0) {
                //Only calculated the requested values once
                ASSERT_EQ(0, counter.counts(i,0));
                ASSERT_EQ(1, counter.counts(i,1));
                ASSERT_EQ(1, counter.counts(i,2));
                ASSERT_EQ(1, counter.counts(i,3));
                ASSERT_EQ(0, counter.counts(i,4));
            } else {
                for (int j = 0; j < N; j++) {
                    ASSERT_EQ(0, counter.counts(i,j));
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int result = -1;

    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    try {
        result = RUN_ALL_TESTS();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
    MPI_Finalize();

    return result;
}
