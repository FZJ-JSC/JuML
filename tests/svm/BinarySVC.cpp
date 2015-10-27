#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <armadillo>
#include "svm/BinarySVC.h"
#include "data/Dataset.h"

TEST (BinarySVCTest, 4PointExample) {
	using juml::Dataset;
	using juml::svm::BinarySVC;
	//From http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
	arma::Mat<float> X = { {-1, -1}, {-2, -1}, {1, 1}, {2, 1} };
	//Transpose, because we need the samples in columns and the features in rows...
	X = X.t();
	arma::Col<int> Y = {1, 1, 2, 2};
	Dataset<float> X_dataset(X);
	Dataset<int> Y_dataset(Y);
	BinarySVC clf;
	clf.fit(X_dataset, Y_dataset);


	arma::Col<float> unknown = {-0.8, -1};
	Dataset<float> unknown_dataset(unknown);
	Dataset<int> result = clf.predict(unknown_dataset);

	ASSERT_EQ(1, result.data()(0,0));
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
