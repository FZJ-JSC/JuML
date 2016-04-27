#include <exception>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>

#include "core/Test.h"
#include "data/Dataset.h"
#include "clustering/KMeans.h"
#include "spatial/Distances.h"

static const std::string FILE_PATH = JUML_DATASETS"/iris.h5";
static const std::string SAMPLES = "samples";

static const std::string DUMP_KM = "km_model.h5";

static const float EUCLIDEAN_CENTROIDS[3][4] = {{5.0059996f, 3.4180002f, 1.4640000f, 0.2440000f},
                                                {5.8836064f, 2.7409837f, 4.3885250f, 1.4344264f},
                                                {6.8538442f, 3.0769234f, 5.7153850f, 2.0538464f}};

static const float MANHATTAN_CENTROIDS[3][4] = {{5.0059996f, 3.4180002f, 1.4640000f, 0.2440000f},
                                                {5.6921053f, 2.6657896f, 4.1157899f, 1.2736841f},
                                                {6.6112895f, 2.9983876f, 5.3903232f, 1.9225806f}};

TEST_ALL(KMEANS_TEST, IRIS_EUCLIDEAN) {
    juml::KMeans kmeans(
            /*k=*/3,
            /*max_iter=*/100,
            /*method=*/juml::KMeans::Method::RANDOM,
            /*distance=*/juml::euclidean,
            /*tolerance=*/0.02,
            /*seed=*/42L,
            /*backend=*/BACKEND);
    juml::Dataset X(FILE_PATH, SAMPLES);

    kmeans.fit(X);
    const af::array& centroids = kmeans.centroids();

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            ASSERT_NEAR(centroids(col, row).scalar<float>(), EUCLIDEAN_CENTROIDS[row][col],0.0001);
        }
    }
}

TEST_ALL(KMEANS_TEST, IRIS_MANHATTAN) {
    juml::KMeans kmeans(
            /*k=*/3,
            /*max_iter=*/100,
            /*method=*/juml::KMeans::Method::RANDOM,
            /*distance=*/juml::manhattan,
            /*tolerance=*/0.02,
            /*seed=*/42L,
            /*backend=*/BACKEND);
    juml::Dataset X(FILE_PATH, SAMPLES);

    kmeans.fit(X);
    const af::array& centroids = kmeans.centroids();

    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            ASSERT_NEAR(centroids(col, row).scalar<float>(), MANHATTAN_CENTROIDS[row][col], 0.0001);
        }
    }
}

TEST_ALL(KMEANS_TEST, IRIS_PASS_CENTROIDS) {
    af::array centroids = af::constant(2.0f, 4, 3);
    juml::KMeans kmeans(
            /*k=*/3,
            /*centroids=*/centroids,
            /*max_iter=*/0,
            /*distance=*/juml::euclidean,
            /*tolerance=*/0.02,
            /*seed=*/42L,
            /*backend=*/BACKEND);
    juml::Dataset X(FILE_PATH, SAMPLES);
    kmeans.fit(X);

    const af::array& kmeans_centroids = kmeans.centroids();
    ASSERT_EQ(af::allTrue(centroids == kmeans_centroids).scalar<char>(), 1);
}


TEST_ALL (KMEANS_TEST, SAVE_LOAD_TEST) {
    int rank_;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    // create and train gnb
    juml::KMeans km(3,BACKEND);
    juml::Dataset X(FILE_PATH, SAMPLES);
    km.fit(X);
    // save model
    km.save(DUMP_KM);
    std::ifstream fin(DUMP_KM.c_str());
    ASSERT_TRUE(fin.is_open());
    // load model
    juml::KMeans loaded(2, BACKEND);
    loaded.load(DUMP_KM);

    if (rank_ == 0) {
        std::remove(DUMP_KM.c_str());
    }

    const af::array& centroids = loaded.centroids();
    ASSERT_TRUE(af::allTrue<bool>(centroids == km.centroids()));
}

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
