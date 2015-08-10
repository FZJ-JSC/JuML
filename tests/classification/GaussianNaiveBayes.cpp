#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>

#include "classification/GaussianNaiveBayes.h"
#include "data/Dataset.h"

TEST (GAUSSIAN_NAIVE_BAYES_TEST, FIT_TEST) {
    juml::GaussianNaiveBayes gnb();
    ASSERT_TRUE(false);
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
