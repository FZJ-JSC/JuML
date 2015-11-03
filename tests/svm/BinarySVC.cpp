#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <armadillo>
#include "svm/BinarySVC.h"
#include "data/Dataset.h"
#include "svm/Kernel.h"

TEST (BinarySVCTest, 4PointExample) {
	using juml::Dataset;
	using juml::svm::BinarySVC;
	//From http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
	//         |
	//         | 2 2
	// --------+--------
	//     1 1?|
	//         |
	arma::Mat<float> X = { {-1, -1}, {-2, -1}, {1, 1}, {2, 1} };
	//Transpose, because we need the samples in columns and the features in rows...
	X = X.t();
	arma::Col<int> Y = {1, 1, 2, 2};
	Dataset<float> X_dataset(X);
	Dataset<int> Y_dataset(Y);
	BinarySVC clf;
	clf.fit(X_dataset, Y_dataset);
	std::cout << "n_support: "<< clf.n_support << " rho: " << clf.rho << " obj_value: " << clf.obj_value << std::endl;
	std::cout << "Support Vectors:" << std::endl << clf.support_vectors << std::endl;
	std::cout << "support:" << std::endl << clf.support << std::endl;
	std::cout << "support_coefs:" << std::endl << clf.support_coefs << std::endl;


	arma::Col<float> unknown = {-0.8, -1};
	Dataset<float> unknown_dataset(unknown);
	Dataset<int> result = clf.predict(unknown_dataset);

	ASSERT_EQ(1, result.data()(0,0));
}

TEST (BinarySVCTest, Linear4PointExample) {
	using juml::Dataset;
	using juml::svm::BinarySVC;
	using juml::svm::KernelType;
	//Example with 4 linear separable points with equal distance
	//        |
	//      1 | 2
	// -------+--------
	//      1 | 2
	//        |
	arma::Mat<float> X = { {-1, -1}, {-1, 1},  {1, -1}, {1, 1} };
	//Transpose, because we need the samples in columns and the features in rows...
	X = X.t();
	arma::Col<int> Y = {1, 1, 2, 2};
	Dataset<float> X_dataset(X);
	Dataset<int> Y_dataset(Y);
	BinarySVC clf(1.0, KernelType::LINEAR);
	clf.fit(X_dataset, Y_dataset);

	std::cout << "n_support: "<< clf.n_support << " rho: " << clf.rho << " obj_value: " << clf.obj_value << std::endl;
	std::cout << "Support Vectors:" << std::endl << clf.support_vectors << std::endl;
	std::cout << "support:" << std::endl << clf.support << std::endl;
	std::cout << "support_coefs:" << std::endl << clf.support_coefs << std::endl;


	arma::Mat<float> unknown = {
		{-2, -1},
		{-0.5, 0},
		{1, 0},
		{0.6, 2}};
	unknown = unknown.t();
	arma::Col<int> expected = {
		1, 1, 2, 2
	};
	Dataset<float> unknown_dataset(unknown);
	Dataset<int> result = clf.predict(unknown_dataset);

	for (int i = 0; i < 4; i++)
		ASSERT_EQ(expected(i), result.data()(0,i));

}

TEST (BinaryLabelTest, CastToInt) {
	using juml::svm::BinaryLabel;
	EXPECT_EQ(-1, (int)BinaryLabel::NEGATIVE);
	EXPECT_EQ(1, (int)BinaryLabel::POSITIVE);
}

TEST(BinaryLabelTest, Multiply) {
	using juml::svm::BinaryLabel;
	EXPECT_EQ(1, BinaryLabel::POSITIVE * BinaryLabel::POSITIVE);
	EXPECT_EQ(1, BinaryLabel::NEGATIVE * BinaryLabel::NEGATIVE);
	EXPECT_EQ(-1, BinaryLabel::POSITIVE * BinaryLabel::NEGATIVE);
	EXPECT_EQ(-1, BinaryLabel::NEGATIVE * BinaryLabel::POSITIVE);
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
