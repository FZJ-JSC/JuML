#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>

#include "data/Dataset.h"
#include "core/Definitions.h"
#include "clustering/KMeans.h"

static const std::string FILE_PATH = JUML_DATASETS"/iris.h5";
static const std::string SAMPLES = "samples";

static const float CENTROIDS[3][4] = {{5.0059996f, 3.4180002f, 1.4640000f, 0.2440000f},
                                      {5.8836064f, 2.7409837f, 4.3885250f, 1.4344264f},
                                      {6.8538442f, 3.0769234f, 5.7153850f, 2.0538464f}};

TEST(KMEANS_TEST, IRIS_CPU_TEST) {
    juml::KMeans kmeans(
            /*k=*/3,
            /*max_iter=*/100,
            /*method=*/juml::KMeans::Method::RANDOM,
            /*tolerance=*/0.02,
            /*seed=*/42L,
            /*backend=*/juml::Backend::CPU);
    juml::Dataset X(FILE_PATH, SAMPLES);

    kmeans.fit(X);
    const af::array& centroids = kmeans.centroids();

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            ASSERT_FLOAT_EQ(centroids(col, row).scalar<float>(), CENTROIDS[row][col]);
        }
    }
}

#ifdef JUML_OPENCL
TEST(KMEANS_TEST, IRIS_OPENCL_TEST) {
    juml::KMeans kmeans(
            /*k=*/3,
            /*max_iter=*/100,
            /*method=*/juml::KMeans::Method::RANDOM,
            /*tolerance=*/0.02,
            /*seed=*/42L,
            /*backend=*/juml::Backend::OPENCL);
    juml::Dataset X(FILE_PATH, SAMPLES);

    kmeans.fit(X);
    const af::array& centroids = kmeans.centroids();

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            ASSERT_FLOAT_EQ(centroids(col, row).scalar<float>(), CENTROIDS[row][col]);
        }
    }
}
#endif // JUML_OPENCL

#ifdef JUML_CUDA
TEST(KMEANS_TEST, IRIS_CUDA_TEST) {
    juml::KMeans kmeans(
            /*k=*/3,
            /*max_iter=*/100,
            /*method=*/juml::KMeans::Method::RANDOM,
            /*tolerance=*/0.02,
            /*seed=*/42L,
            /*backend=*/juml::Backend::CUDA);
    juml::Dataset X(FILE_PATH, SAMPLES);

    kmeans.fit(X);
    const af::array& centroids = kmeans.centroids();

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            ASSERT_FLOAT_EQ(centroids(col, row).scalar<float>(), CENTROIDS[row][col]);
        }
    }
}
#endif // JUML_CUDA

int main(int argc, char** argv) {
    int result = -1;
    int rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::testing::InitGoogleTest(&argc, argv);
    
    // suppress the output from the other ranks
    if (rank > 0) {
        ::testing::UnitTest& unit_test = *::testing::UnitTest::GetInstance();
        ::testing::TestEventListeners& listeners = unit_test.listeners();
        delete listeners.Release(listeners.default_result_printer());
        listeners.Append(new ::testing::EmptyTestEventListener);
    }
    
    try {
        result = RUN_ALL_TESTS();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
    }
    MPI_Finalize();

    return result;
}
