#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <string>

#include "classification/GaussianNaiveBayes.h"
#include "data/Dataset.h"

/*
Out[11]: array([ 0.33333333,  0.33333333,  0.33333333])

In [12]: nb.theta_
Out[12]: 
array([[ 5.00599957,  3.41800022,  1.46399999,  0.24399997],
       [ 5.93600178,  2.76999998,  4.26000023,  1.32599986],
       [ 6.58799934,  2.97400022,  5.55200052,  2.02600026]])

In [13]: nb.sigma_
Out[13]: 
array([[ 0.12176395,  0.14227599,  0.02950401,  0.011264  ],
       [ 0.26110402,  0.09650001,  0.21640001,  0.038324  ],
       [ 0.396256  ,  0.10192403,  0.29849592,  0.07392402]])

In [14]: np.sqrt(nb.sigma_)
Out[14]: 
array([[ 0.34894692,  0.3771949 ,  0.17176733,  0.10613199],
       [ 0.51098338,  0.31064451,  0.46518815,  0.19576517],
       [ 0.62948868,  0.31925543,  0.5463478 ,  0.27188972]])

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

TEST (GAUSSIAN_NAIVE_BAYES_TEST, FIT_TEST) {
    juml::GaussianNaiveBayes gnb;
    juml::Dataset<float> X(FILE_PATH, SAMPLES);
    juml::Dataset<int> y(FILE_PATH, LABELS);
    
    gnb.fit(X, y);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        gnb.theta().print();
    }
    const arma::Mat<float>& prior = gnb.prior();
    const arma::Mat<float>& theta = gnb.theta();
    const arma::Mat<float>& stddev = gnb.stddev();
    for (int row = 0; row < 3; ++row) {
        ASSERT_FLOAT_EQ((row), PRIORS[row]);
        for (int col = 0; col < 4; ++col) {
            ASSERT_FLOAT_EQ(gnb.theta()(row, col), THETA[row][col]);
            ASSERT_FLOAT_EQ(gnb.stddev()(row, col), STDDEV[row][col]);
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
