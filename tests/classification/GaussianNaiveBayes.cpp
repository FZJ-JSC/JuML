#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <string>

#include "core/Definitions.h"
#include "classification/GaussianNaiveBayes.h"
#include "data/Dataset.h"

static const std::string FILE_PATH = JUML_DATASETS"/iris.h5";
static const std::string SAMPLES = "samples";
static const std::string LABELS = "labels";

static const float PRIORS[3] = {0.33333333f, 0.33333333f, 0.33333333f};
static const float THETA[3][4] = {{5.00599957f, 3.41800022f, 1.46399999f, 0.24399997f},
                                  {5.93600180f, 2.76999998f, 4.26000023f, 1.32599986f},
                                  {6.58799934f, 2.97400022f, 5.55200052f, 2.02600026f}};
                                    
static const float STDDEV[3][4] = {{0.34894692f, 0.37719490f, 0.17176733f, 0.10613199f},
                                   {0.51098338f, 0.31064451f, 0.46518815f, 0.19576517f},
                                   {0.62948868f, 0.31925543f, 0.54634780f, 0.27188972f}};
                                   
static const float ACCURACY = 0.95999999999999996f;

TEST (GAUSSIAN_NAIVE_BAYES_TEST, IRIS_CPU_TEST) {
    juml::GaussianNaiveBayes gnb(juml::Backend::CPU);
    juml::Dataset X(FILE_PATH, SAMPLES);
    juml::Dataset y(FILE_PATH, LABELS);
    
    gnb.fit(X, y);
    const af::array& prior = gnb.prior();
    const af::array& theta = gnb.theta();
    const af::array& stddev = gnb.stddev();
    
    for (int row = 0; row < 3; ++row) {
        ASSERT_FLOAT_EQ(prior(row).scalar<float>(), PRIORS[row]);
        for (int col = 0; col < 4; ++col) {
            ASSERT_NEAR(theta(col, row).scalar<float>(), THETA[row][col], 0.001);
            ASSERT_FLOAT_EQ(stddev(col, row).scalar<float>(), STDDEV[row][col]);
        }
    }
    ASSERT_FLOAT_EQ(gnb.accuracy(X, y), ACCURACY);

    juml::Dataset X_not_loaded(FILE_PATH, SAMPLES);
    juml::Dataset y_not_loaded(FILE_PATH, LABELS);
    ASSERT_FLOAT_EQ(gnb.accuracy(X_not_loaded, y_not_loaded), ACCURACY);
}

#ifdef JUML_OPENCL
TEST (GAUSSIAN_NAIVE_BAYES_TEST, IRIS_OPENCL_TEST) {
    juml::GaussianNaiveBayes gnb(juml::Backend::OPENCL);
    juml::Dataset X(FILE_PATH, SAMPLES);
    juml::Dataset y(FILE_PATH, LABELS);
    
    gnb.fit(X, y);
    const af::array& prior = gnb.prior();
    const af::array& theta = gnb.theta();
    const af::array& stddev = gnb.stddev();
    
    for (int row = 0; row < 3; ++row) {
        ASSERT_FLOAT_EQ(prior(row).scalar<float>(), PRIORS[row]);
        for (int col = 0; col < 4; ++col) {
            ASSERT_NEAR(theta(col, row).scalar<float>(), THETA[row][col], 0.001);
            ASSERT_FLOAT_EQ(stddev(col, row).scalar<float>(), STDDEV[row][col]);
        }
    }
    ASSERT_FLOAT_EQ(gnb.accuracy(X, y), ACCURACY);

    juml::Dataset X_not_loaded(FILE_PATH, SAMPLES);
    juml::Dataset y_not_loaded(FILE_PATH, LABELS);
    ASSERT_FLOAT_EQ(gnb.accuracy(X_not_loaded, y_not_loaded), ACCURACY);
}

#endif // JUML_OPENCL

#ifdef JUML_CUDA
TEST (GAUSSIAN_NAIVE_BAYES_TEST, IRIS_CUDA_TEST) {
    juml::GaussianNaiveBayes gnb(juml::Backend::CUDA);
    juml::Dataset X(FILE_PATH, SAMPLES);
    juml::Dataset y(FILE_PATH, LABELS);
    
    gnb.fit(X, y);
    const af::array& prior = gnb.prior();
    const af::array& theta = gnb.theta();
    const af::array& stddev = gnb.stddev();
    
    for (int row = 0; row < 3; ++row) {
        ASSERT_FLOAT_EQ(prior(row).scalar<float>(), PRIORS[row]);
        for (int col = 0; col < 4; ++col) {
            ASSERT_NEAR(theta(col, row).scalar<float>(), THETA[row][col], 0.001);
            ASSERT_FLOAT_EQ(stddev(col, row).scalar<float>(), STDDEV[row][col]);
        }
    }
    ASSERT_FLOAT_EQ(gnb.accuracy(X, y), ACCURACY);

    juml::Dataset X_not_loaded(FILE_PATH, SAMPLES);
    juml::Dataset y_not_loaded(FILE_PATH, LABELS);
    ASSERT_FLOAT_EQ(gnb.accuracy(X_not_loaded, y_not_loaded), ACCURACY);
}

#endif // JUML_CUDA

int main(int argc, char** argv) {
    int result = -1;
    int rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::testing::InitGoogleTest(&argc, argv);
    
    // suppress output from the other ranks
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
