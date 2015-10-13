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

#define EXPECT_MAT_EQ(a, b) do {\
    ASSERT_TRUE(a.n_rows == b.n_rows && a.n_cols == b.n_cols) << "a and b do not have the same dimensions!"; \
    for (int col = 0; col < a.n_cols; col++) { \
        for (int row = 0; row < a.n_rows; row++) { \
            EXPECT_EQ(a(row, col), b(row, col)) << "a differs from b at index (" << row << ", " << col << ")"; \
        } \
    } \
} while (false)



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
    auto cachedKernel = juml::svm::KernelCache<KernelEvaluationCounter>(counter, sizeof(float) * N * N, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ASSERT_EQ(0, counter.counts(i,j));
        }
    }

    std::vector<unsigned int> idxs = {1, 2, 3};
    //Request some entrys of the first column multiple times
    //Every requested Kernel entry should only have been calculated once
    arma::Mat<unsigned int> expected = {
        {0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };
    for (int z = 0; z < 5; z++ ) {
        cachedKernel.get_col(0, idxs);
        EXPECT_MAT_EQ(expected, counter.counts);
    }
}

TEST (KernelCacheTest, TestCachedKernelWithCacheDeletion) {
    const int N = 5;
    KernelEvaluationCounter counter(N);
    //Only enough space in cache for 2 columns
    auto cachedKernel = juml::svm::KernelCache<KernelEvaluationCounter>(counter, sizeof(float) * N * 2, N);

    //Nothing is cached initially
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ASSERT_EQ(0, counter.counts(i,j));
        }
    }

    std::vector<unsigned int> idxs = {1, 2, 3};

    //Request uncached elements of the first 2 columns.
    for (int z = 0; z < 2; z++ ) {
        cachedKernel.get_col(z, idxs);
    }

    arma::Mat<unsigned int> expected = {
        {0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);

    //Requesting the 3rd column should result in the first column not being cached anymore
    cachedKernel.get_col(2, idxs);
    expected = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };
    EXPECT_MAT_EQ(expected, counter.counts);

    //Requesting the 1st column so it gets calculated again
    cachedKernel.get_col(0, idxs);
    expected = {
        {0, 0, 0, 0, 0},
        {2, 1, 1, 0, 0},
        {2, 1, 1, 0, 0},
        {2, 1, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };
    EXPECT_MAT_EQ(expected, counter.counts);

    //The 1st and 3rd column are cached, so we can request them without recalculation
    for (int i = 0; i < 5; i++) {
        cachedKernel.get_col(0, idxs);
        EXPECT_MAT_EQ(expected, counter.counts);
        cachedKernel.get_col(2, idxs);
        EXPECT_MAT_EQ(expected, counter.counts);
    }
}


TEST(KernelCacheTest, TestCachedKernelPrecomputed) {
    using namespace juml::svm;
    const int N = 5;
    arma::Mat<float> precomputedKernel(N,N, arma::fill::zeros);
    Kernel<KernelType::PRECOMPUTED, float> kernel(precomputedKernel);
    auto cachedKernel = KernelCache<Kernel<KernelType::PRECOMPUTED, float>>(kernel);

    std::vector<unsigned int> idxs = {1,2,3};
    for (int col = 0; col < N; col++) {
        ASSERT_EQ(&(precomputedKernel.at(0, col)), &(cachedKernel.get_col(col, idxs)[0]))
            << "Cached Precomputed Kernel should return pointer into precomputed kernel memory";
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
