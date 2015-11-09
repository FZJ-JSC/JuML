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
		return i + n * j;
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

#define ASSERT_KERNEL_VALUES(data, col, N) do { \
    for (int i = 0; i < N; i++) { \
        ASSERT_FLOAT_EQ(i + col*N, data[i]); \
    } \
} while (false)

#define ASSERT_KERNEL_VALUES_SUBSET(data, col, idxs, N) do { \
    for (int i: idxs) { \
        ASSERT_FLOAT_EQ(i + col*N, data[i]); \
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
        const float *data = cachedKernel.get_col(0, idxs);
        EXPECT_MAT_EQ(expected, counter.counts);
        ASSERT_KERNEL_VALUES_SUBSET(data, 0, idxs, 5);
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

    //Nothing should be cached
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(false, cachedKernel.is_cached(i));
    }

    std::vector<unsigned int> idxs = {1, 2, 3};
    const float *data;
    //Request uncached elements of the first 2 columns.
    for (int z = 0; z < 2; z++ ) {
        data = cachedKernel.get_col(z, idxs);
        ASSERT_KERNEL_VALUES_SUBSET(data, z, idxs, N);
    }

    //First 2 column should report is_cached = true now
    EXPECT_EQ(true,  cachedKernel.is_cached(0));
    EXPECT_EQ(true,  cachedKernel.is_cached(1));
    EXPECT_EQ(false, cachedKernel.is_cached(2));
    EXPECT_EQ(false, cachedKernel.is_cached(3));
    EXPECT_EQ(false, cachedKernel.is_cached(4));

    arma::Mat<unsigned int> expected = {
        {0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);

    //Requesting the 3rd column should result in the first column not being cached anymore
    data = cachedKernel.get_col(2, idxs);
    ASSERT_KERNEL_VALUES_SUBSET(data, 2, idxs, N);

    EXPECT_EQ(false, cachedKernel.is_cached(0));
    EXPECT_EQ(true,  cachedKernel.is_cached(1));
    EXPECT_EQ(true,  cachedKernel.is_cached(2));
    EXPECT_EQ(false, cachedKernel.is_cached(3));
    EXPECT_EQ(false, cachedKernel.is_cached(4));

    expected = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };
    EXPECT_MAT_EQ(expected, counter.counts);

    //Requesting the 1st column so it gets calculated again
    data = cachedKernel.get_col(0, idxs);
    ASSERT_KERNEL_VALUES_SUBSET(data, 0, idxs, N);

    EXPECT_EQ(true,  cachedKernel.is_cached(0));
    EXPECT_EQ(false, cachedKernel.is_cached(1));
    EXPECT_EQ(true,  cachedKernel.is_cached(2));
    EXPECT_EQ(false, cachedKernel.is_cached(3));
    EXPECT_EQ(false, cachedKernel.is_cached(4));

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
        data =cachedKernel.get_col(0, idxs);
        ASSERT_KERNEL_VALUES_SUBSET(data, 0, idxs, N);
        EXPECT_MAT_EQ(expected, counter.counts);

        data = cachedKernel.get_col(2, idxs);
        ASSERT_KERNEL_VALUES_SUBSET(data, 2, idxs, N);
        EXPECT_MAT_EQ(expected, counter.counts);
    }
}

TEST(KernelCacheTest, TestCachedKernelPartialKernelCaching) {
    using std::vector;
    const int N = 5;
    KernelEvaluationCounter counter(N);
    //Enough space to cache the full kernel
    auto cachedKernel = juml::svm::KernelCache<KernelEvaluationCounter>(counter, sizeof(float) * N * N, N);

    arma::Mat<unsigned int> expected = arma::Mat<unsigned int>(N,N, arma::fill::zeros);

    EXPECT_MAT_EQ(expected, counter.counts);

    //Request only a few rows from a specific column
    vector<unsigned int> idxs1 = {1, 2, 3};
    const float *data = cachedKernel.get_col(0, idxs1);
    ASSERT_KERNEL_VALUES_SUBSET(data, 0, idxs1, N);

    expected = {
        {0, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);

    //Request other indixes from the same column. Only the new idxs should be recalculated.
    vector<unsigned int> idxs2 = {0, 1, 4};
    data = cachedKernel.get_col(0, idxs2);
    ASSERT_KERNEL_VALUES_SUBSET(data, 0, idxs2, N);

    expected = {
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);

    //Request the full column
    data = cachedKernel.get_col(1);
    ASSERT_KERNEL_VALUES(data, 1, N);

    expected = {
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);

    //The full column is cached, so requesting special rows does not need any recalculation
    data = cachedKernel.get_col(1, idxs1);
    ASSERT_KERNEL_VALUES(data, 1, N);
    EXPECT_MAT_EQ(expected, counter.counts);

    data = cachedKernel.get_col(1, idxs2);
    ASSERT_KERNEL_VALUES(data, 1, N);
    EXPECT_MAT_EQ(expected, counter.counts);


    //Request a partial column
    data = cachedKernel.get_col(2, idxs1);
    ASSERT_KERNEL_VALUES_SUBSET(data, 2, idxs1, N);

    expected = {
        {1, 1, 0, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 0, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);


    //Requesting the full column after it has been partially calculated
    data = cachedKernel.get_col(2);
    ASSERT_KERNEL_VALUES(data, 2, N);
    expected = {
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);
}

TEST(KernelCacheTest, TestCachedKernelUncachedEvaluation) {
    using std::vector;
    const int N = 3;
    KernelEvaluationCounter counter(N);
    auto cachedKernel = juml::svm::KernelCache<KernelEvaluationCounter>(counter, sizeof(float) * N * N, N);

    arma::Mat<unsigned int> expected = arma::Mat<unsigned int>(N,N, arma::fill::zeros);

    EXPECT_MAT_EQ(expected, counter.counts);

    EXPECT_EQ(false, cachedKernel.is_cached(0));
    EXPECT_EQ(false, cachedKernel.is_cached(1));
    EXPECT_EQ(false, cachedKernel.is_cached(2));

    //Evaluate kernel without caching
    EXPECT_FLOAT_EQ(0.0, cachedKernel.evaluate_kernel(0,0));

    //Kernel has been evaluated, but not cached
    expected = {
        {1, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);

    EXPECT_EQ(false, cachedKernel.is_cached(0));
    EXPECT_EQ(false, cachedKernel.is_cached(1));
    EXPECT_EQ(false, cachedKernel.is_cached(2));

    cachedKernel.get_col(0);
    expected = {
        {2, 0, 0},
        {1, 0, 0},
        {1, 0, 0}
    };

    EXPECT_MAT_EQ(expected, counter.counts);

    EXPECT_EQ(true, cachedKernel.is_cached(0));
    EXPECT_EQ(false, cachedKernel.is_cached(1));
    EXPECT_EQ(false, cachedKernel.is_cached(2));

    //Kernel Cache status should not change after executing evaluete_kernel
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cachedKernel.evaluate_kernel(i,j);
        }
    }

    EXPECT_EQ(true, cachedKernel.is_cached(0));
    EXPECT_EQ(false, cachedKernel.is_cached(1));
    EXPECT_EQ(false, cachedKernel.is_cached(2));

}


TEST(KernelCacheTest, TestCachedKernelPrecomputed) {
    using namespace juml::svm;
    const int N = 5;
    arma::Mat<float> precomputedKernel(N,N, arma::fill::zeros);
    Kernel<KernelType::PRECOMPUTED> kernel(precomputedKernel);
    auto cachedKernel = KernelCache<Kernel<KernelType::PRECOMPUTED>>(kernel);

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
