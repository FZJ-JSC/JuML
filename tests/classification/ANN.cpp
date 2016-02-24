#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <arrayfire.h>
#include "data/Dataset.h"
#include "classification/ANN.h"

TEST(ANN_TEST, TEST_SIMPLE_NETWORK) {
	using juml::Dataset;
	using juml::ann::Layer;
	using juml::ann::SigmoidLayer;
	using juml::SequentialNeuralNet;
	juml::Backend b = AF_BACKEND_CPU;
	float X[] = {
		0, 0, 1,
		1, 1, 1,
		1, 0, 1,
		0, 1, 1};
	float y[] = {0, 1, 1, 0};
	af::array Xarray = af::array(4, 3, X).T();
	//row vector, because the target output vector is in columns, but our output only has 1 value
	af::array yarray = af::array(1, 4, y);
	Dataset Xset(Xarray);
	Dataset yset(yarray);
	std::vector<std::shared_ptr<Layer>> layers;
	layers.push_back(std::shared_ptr<Layer>(new SigmoidLayer(3, 4)));
	layers.push_back(std::shared_ptr<Layer>(new SigmoidLayer(4,1)));
	SequentialNeuralNet net(AF_BACKEND_CPU, layers);
	net.fit(Xset, yset);
	//TODO check result
	//TODO fix network to work with binary problems using 1 output 

}


static const std::string FILE_PATH = "../../../datasets/iris.h5";
static const std::string SAMPLES = "samples";
static const std::string LABELS = "labels";


TEST(ANN_TEST, IRIS_TEST) {
	using juml::ann::Layer;
	using juml::ann::SigmoidLayer;
	std::vector<std::shared_ptr<Layer>> layers;
	layers.push_back(std::shared_ptr<Layer>(new SigmoidLayer(4, 100)));
	layers.push_back(std::shared_ptr<Layer>(new SigmoidLayer(100, 1)));
	juml::SequentialNeuralNet net(AF_BACKEND_CPU, layers);
	juml::Dataset X(FILE_PATH, SAMPLES);
	juml::Dataset y(FILE_PATH, LABELS);

	net.fit(X, y);
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
