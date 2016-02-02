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
	float X[] = {
		0, 0, 1,
		0, 1, 1,
		1, 0, 1,
		1, 1, 1};
	float y[] = {0, 1, 1, 0};
	af::array Xarray = af::array(4, 3, X).T();
	af::array yarray = af::array(4, y);
	Dataset Xset(Xarray);
	Dataset yset(yarray);
	std::vector<Layer*> layers = { new SigmoidLayer(3, 4), new SigmoidLayer(4,2) };
	SequentialNeuralNet net(layers);
	net.fit(Xset, yset);
	//TODO check result
	//TODO fix network to work with binary problems using 1 output 

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
