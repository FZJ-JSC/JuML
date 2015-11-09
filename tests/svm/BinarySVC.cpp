#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <armadillo>
#include "svm/BinarySVC.h"
#include "data/Dataset.h"
#include "svm/Kernel.h"
#include "svm/KernelCache.h"
#include "svm/QMatrix.h"

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

TEST(BinaryLabelTest, BooleanOperators) {
	using juml::svm::BinaryLabel;
	EXPECT_EQ(true, BinaryLabel::POSITIVE == BinaryLabel::POSITIVE);
	EXPECT_EQ(false, BinaryLabel::POSITIVE != BinaryLabel::POSITIVE);
	EXPECT_EQ(true, BinaryLabel::NEGATIVE == BinaryLabel::NEGATIVE);
	EXPECT_EQ(false, BinaryLabel::NEGATIVE != BinaryLabel::NEGATIVE);
}

TEST(BinaryLabelTest, Multiply) {
	using juml::svm::BinaryLabel;
	EXPECT_EQ(1, BinaryLabel::POSITIVE * BinaryLabel::POSITIVE);
	EXPECT_EQ(1, BinaryLabel::NEGATIVE * BinaryLabel::NEGATIVE);
	EXPECT_EQ(-1, BinaryLabel::POSITIVE * BinaryLabel::NEGATIVE);
	EXPECT_EQ(-1, BinaryLabel::NEGATIVE * BinaryLabel::POSITIVE);
	EXPECT_EQ(-0.5, BinaryLabel::NEGATIVE * BinaryLabel::POSITIVE * 0.5);
}


TEST(BinarySVCKernelTest, AllPositive) {
	using juml::svm::SVCKernel;
	using juml::svm::Kernel;
	using juml::svm::BinaryLabel;
	using juml::svm::KernelType;
	arma::Mat<float> x(3,3);
   	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			x(i,j) = i + 1;
		}
	}
	std::vector<BinaryLabel> y = {BinaryLabel::POSITIVE, BinaryLabel::POSITIVE, BinaryLabel::POSITIVE};

	Kernel<KernelType::LINEAR> kernel(x);
	SVCKernel<decltype(kernel)> svcKernel(kernel, y);
	for(int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			EXPECT_EQ(kernel.evaluate_kernel(i, j), svcKernel.evaluate_kernel(i, j));
		}
	}
}

TEST(BinarySVCKernelTest, AllNegative) {
	using juml::svm::SVCKernel;
	using juml::svm::Kernel;
	using juml::svm::BinaryLabel;
	using juml::svm::KernelType;
	arma::Mat<float> x(3,3);
   	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			x(i,j) = i + 1;
		}
	}
	std::vector<BinaryLabel> y = {BinaryLabel::NEGATIVE, BinaryLabel::NEGATIVE, BinaryLabel::NEGATIVE};

	Kernel<KernelType::LINEAR> kernel(x);
	SVCKernel<decltype(kernel)> svcKernel(kernel, y);
	for(int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			EXPECT_EQ(kernel.evaluate_kernel(i, j), svcKernel.evaluate_kernel(i, j));
		}
	}
}


TEST(BinarySVCKernelTest, MixedSamples) {
	using juml::svm::SVCKernel;
	using juml::svm::Kernel;
	using juml::svm::BinaryLabel;
	using juml::svm::KernelType;
	arma::Mat<float> x(3,3);
   	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			x(i,j) = 3*j + i;
		}
	}
	std::vector<BinaryLabel> y = {BinaryLabel::POSITIVE, BinaryLabel::NEGATIVE, BinaryLabel::NEGATIVE};

	Kernel<KernelType::RBF> kernel(x, 3.0);
	SVCKernel<decltype(kernel)> svcKernel(kernel, y);
	for(int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if ((i == 0 || j == 0) && i != j) {
				EXPECT_EQ(-kernel.evaluate_kernel(i, j), svcKernel.evaluate_kernel(i, j));
			} else {
				EXPECT_EQ(kernel.evaluate_kernel(i, j), svcKernel.evaluate_kernel(i, j));
			}
		}
	}
}

TEST(BinarySVCKernelTest, CachedKernelMixedSamples) {
	using juml::svm::SVCKernel;
	using juml::svm::Kernel;
	using juml::svm::BinaryLabel;
	using juml::svm::KernelType;
	using juml::svm::KernelCache;
	using juml::svm::QMatrix;
	arma::Mat<float> x(3,3);
   	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			x(i,j) = 3*j + i;
		}
	}
	std::vector<BinaryLabel> y = {BinaryLabel::POSITIVE, BinaryLabel::NEGATIVE, BinaryLabel::NEGATIVE};

	Kernel<KernelType::RBF> kernel(x, 3.0);
	SVCKernel<decltype(kernel)> svcKernel(kernel, y);
	QMatrix *q = new KernelCache<decltype(svcKernel)>(svcKernel, 9 * sizeof(float), 3);
	for(int i = 0; i < 3; i++) {
		const float *Q_i = q->get_col(i);
		for (int j = 0; j < 3; j++) {
			EXPECT_EQ(svcKernel.evaluate_kernel(i, j), Q_i[j]);
		}
	}
	delete q;

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
