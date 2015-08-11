#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <string>

#include "classification/GaussianNaiveBayes.h"
#include "data/Dataset.h"

/*
In [15]: nb.score(X,y)
Out[15]: 0.95999999999999996
*/

static const std::string FILE_PATH = "../../../datasets/iris.h5";
static const std::string SAMPLES = "samples";
static const std::string LABELS = "labels";

static const float PRIORS[3] = {0.33333333,  0.33333333,  0.33333333};
static const float THETA[3][4] = {  {5.00599957,  3.41800022,  1.46399999,  0.24399997},
                                    {5.93600178,  2.76999998,  4.26000023,  1.32599986},
                                    {6.58799934,  2.97400022,  5.55200052,  2.02600026}};
                                    
static const float STDDEV[3][4] = { {0.34894692,  0.3771949 ,  0.17176733,  0.10613199},
                                    {0.51098338,  0.31064451,  0.46518815,  0.19576517},
                                    {0.62948868,  0.31925543,  0.5463478 ,  0.27188972}};

TEST (GAUSSIAN_NAIVE_BAYES_TEST, IRIS_TEST) {
    juml::GaussianNaiveBayes gnb;
    juml::Dataset<float> X(FILE_PATH, SAMPLES);
    juml::Dataset<int> y(FILE_PATH, LABELS);
    
    gnb.fit(X, y);
    const arma::Mat<float>& prior = gnb.prior();
    const arma::Mat<float>& theta = gnb.theta();
    const arma::Mat<float>& stddev = gnb.stddev();
    for (int row = 0; row < 3; ++row) {
        ASSERT_FLOAT_EQ(prior(row), PRIORS[row]);
        for (int col = 0; col < 4; ++col) {
            ASSERT_FLOAT_EQ(theta(row, col), THETA[row][col]);
            ASSERT_FLOAT_EQ(stddev(row, col), STDDEV[row][col]);
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
