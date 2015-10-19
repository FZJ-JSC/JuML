// vim: expandtab:shiftwidth=4:softtabstop=4
#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include "svm/Kernel.h"

TEST (KernelInitialisation, LinearKernel) {
    using juml::svm::Kernel;
    using juml::svm::KernelType;
    arma::Mat<float> x(3,3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            x(i,j) = i + 1;
        }
    }

    Kernel<KernelType::LINEAR, float> kernel(x);

    ASSERT_EQ(1*1 + 2*2 + 3*3, kernel.evaluate_kernel(0,1));
    ASSERT_EQ(1*1 + 2*2 + 3*3, kernel.evaluate_kernel(1,2));
    ASSERT_EQ(1*1 + 2*2 + 3*3, kernel.evaluate_kernel(0,2));

}

TEST (KernelInitialisation, PolyKernel) {
    using juml::svm::Kernel;
    using juml::svm::KernelType;
    arma::Mat<float> x(3,3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            x(i,j) = i + 1;
        }
    }
    int degree = 2;
    double gamma = 1;
    double coef0 = 0;

    Kernel<KernelType::POLY, float> kernel(x, degree, gamma, coef0);

    float result = pow(1*1 + 2*2 + 3*3, 2);

    ASSERT_EQ(result, kernel.evaluate_kernel(0,1));
    ASSERT_EQ(result, kernel.evaluate_kernel(1,2));
    ASSERT_EQ(result, kernel.evaluate_kernel(0,2));

}

TEST (KernelInitialisation, RBFKernel) {
    using juml::svm::Kernel;
    using juml::svm::KernelType;
    arma::Mat<float> x(3,3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            x(i,j) = i + 1;
        }
    }

    Kernel<KernelType::RBF, float> kernel(x, 2);

    float result = exp(-2 * ( 2 * (1 + 4 + 9) - 2 * ( 1 + 4  + 9)));

    ASSERT_EQ(result, kernel.evaluate_kernel(0,1));
    ASSERT_EQ(result, kernel.evaluate_kernel(1,2));
    ASSERT_EQ(result, kernel.evaluate_kernel(0,2));

}

TEST (KernelInitialisation, PrecomputedKernel) {
    using juml::svm::Kernel;
    using juml::svm::KernelType;
    arma::Mat<float> x(3,3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            x(i,j) = i + j;
        }
    }

    Kernel<KernelType::PRECOMPUTED, float> kernel(x);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ASSERT_EQ(i+j, kernel.evaluate_kernel(i,j));
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
