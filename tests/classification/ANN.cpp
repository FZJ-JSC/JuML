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
	using juml::ann::FunctionLayer;
	using juml::SequentialNeuralNet;
	using juml::ann::Activation;
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
	layers.push_back(std::shared_ptr<Layer>(new FunctionLayer<Activation::Sigmoid>(3, 4)));
	layers.push_back(std::shared_ptr<Layer>(new FunctionLayer<Activation::Sigmoid>(4,1)));
	SequentialNeuralNet net(AF_BACKEND_CPU, layers);
	net.fit(Xset, yset);

	Dataset result = net.predict(Xset);
	af::print("result", result.data());
}

TEST(ANN_TEST, TEST_IDENTITY) {
	using juml::Dataset;
	using juml::ann::Layer;
	using juml::ann::FunctionLayer;
	using juml::ann::Activation;
	using juml::SequentialNeuralNet;
	float X[] = {
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1
	};
	af::array Xarray = af::array(5, 5, X);
	Dataset Xset(Xarray);

	std::vector<std::shared_ptr<Layer>> layers;
	layers.push_back(std::make_shared<FunctionLayer<Activation::Sigmoid>>(5, 5));
	layers.push_back(std::make_shared<FunctionLayer<Activation::Sigmoid>>(5, 5));

	SequentialNeuralNet net(AF_BACKEND_CPU, layers);
	net.fit(Xset, Xset);

	Dataset result = net.predict(Xset);
	af::print("result", result.data());

}

TEST(ANN_TEST, TEST_XOR) {
	using juml::Dataset;
	using juml::ann::Layer;
	using juml::ann::FunctionLayer;
	using juml::SequentialNeuralNet;
	using juml::ann::Activation;
	af::setBackend(AF_BACKEND_CPU);
	af::info();

	float X[] = {
		0, 0,
		0, 1,
		1, 0,
		1, 1
	};
	float y[] = {0, 1, 1, 0};
	af::array Xarray = af::array(2, 4, X);
	af_print(Xarray);
	af::array yarray = af::array(1, 4, y);
	af_print(yarray);
	Dataset Xset(Xarray);
	Dataset yset(yarray);

	std::vector<std::shared_ptr<Layer>> layers;
	layers.push_back(std::make_shared<FunctionLayer<Activation::Sigmoid>>(2, 2));
	layers.push_back(std::make_shared<FunctionLayer<Activation::Linear>>(2, 1));
	SequentialNeuralNet net(AF_BACKEND_CPU, layers);
	net.fit(Xset, yset);

	for(auto it = layers.begin(); it != layers.end(); it++) {
		af::print("weights", (*it)->getWeights());
		af::print("bias", (*it)->getBias());
	}


	Dataset result = net.predict(Xset);
	af::print("result", result.data());
}


static const std::string FILE_PATH = "../../../datasets/iris_ann.h5";
static const std::string SAMPLES = "samples";
static const std::string LABELS = "labels";


TEST(ANN_TEST, IRIS_TEST) {
	af::info();
	af::setBackend(AF_BACKEND_CPU);
	using juml::ann::Layer;
	std::vector<std::shared_ptr<Layer>> layers;
	layers.push_back(std::shared_ptr<Layer>(new juml::ann::FunctionLayer<juml::ann::Activation::Sigmoid>(4, 100)));
	layers.push_back(std::shared_ptr<Layer>(new juml::ann::FunctionLayer<juml::ann::Activation::Sigmoid>(100, 3)));
	juml::SequentialNeuralNet net(AF_BACKEND_CPU, layers);
	juml::Dataset X(FILE_PATH, SAMPLES);
	juml::Dataset y(FILE_PATH, LABELS);

	net.fit(X, y);
}

TEST(ANN_TEST, INCOMPATIBLE_LAYERS) {
	using juml::ann::Layer;
	using juml::ann::FunctionLayer;
	using juml::ann::Activation;
	std::vector<std::shared_ptr<Layer>> layers;
	layers.push_back(std::make_shared<FunctionLayer<Activation::Sigmoid>>(4, 5));
	layers.push_back(std::make_shared<FunctionLayer<Activation::Sigmoid>>(3, 2));
	try {
		juml::SequentialNeuralNet(0, layers);
		FAIL() << "SequentialNeuralNet Constructor did not notice incompatible layers";
	} catch(...) {
		SUCCEED();
		//TODO: Check for explicit Exception type
	}
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
