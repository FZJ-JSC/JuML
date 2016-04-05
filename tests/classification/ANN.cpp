#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <mpi.h>
#include <arrayfire.h>
#include "data/Dataset.h"
#include "classification/ANN.h"
#include "core/Test.h"

TEST_ALL(ANN_TEST, TEST_SIMPLE_NETWORK) {
	using juml::Dataset;
	using juml::ann::Layer;
	using juml::ann::FunctionLayer;
	using juml::SequentialNeuralNet;
	using juml::ann::Activation;
	af::setBackend((af::Backend)BACKEND);
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
	SequentialNeuralNet net(BACKEND);
	net.add(std::shared_ptr<Layer>(new FunctionLayer<Activation::Sigmoid>(3, 4)));
	net.add(std::shared_ptr<Layer>(new FunctionLayer<Activation::Sigmoid>(4, 1)));
	net.fit(Xset, yset);

	Dataset result = net.predict(Xset);
	af::print("result", result.data());
}

TEST_ALL(ANN_TEST, TEST_IDENTITY) {
	using juml::Dataset;
	using juml::ann::Layer;
	using juml::ann::FunctionLayer;
	using juml::ann::Activation;
	using juml::SequentialNeuralNet;
	af::setBackend((af::Backend)BACKEND);
	float X[] = {
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1
	};
	af::array Xarray = af::array(5, 5, X);
	Dataset Xset(Xarray);

	SequentialNeuralNet net(BACKEND);
	net.add(std::make_shared<FunctionLayer<Activation::Sigmoid>>(5, 5));
	net.add(std::make_shared<FunctionLayer<Activation::Sigmoid>>(5, 5));

	net.fit(Xset, Xset);

	Dataset result = net.predict(Xset);
	af::print("result", result.data());

}

TEST_ALL(ANN_TEST, TEST_XOR) {
	using juml::Dataset;
	using juml::ann::Layer;
	using juml::ann::FunctionLayer;
	using juml::SequentialNeuralNet;
	using juml::ann::Activation;
	af::setBackend((af::Backend)BACKEND);
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

	float weights1[] = {0.06, -0.03, 0.2, 0.1};
	float bias1[] = {-0.2, -0.2};
	float weights2[] = {0.2, 0.08};
	float bias2[] = {0.08};


	SequentialNeuralNet net(BACKEND);
	net.add(std::make_shared<FunctionLayer<Activation::Sigmoid>>(
			af::array(2,2, weights1), af::array(2, bias1)));
	net.add(std::make_shared<FunctionLayer<Activation::Linear>>(
			af::array(2, 1, weights2), af::array(1, bias2)));

	for(auto it = net.layers_begin(); it != net.layers_end(); it++) {
		af::print("weights", (*it)->getWeights());
		af::print("bias", (*it)->getBias());
	}

	net.fit(Xset, yset);

	for(auto it = net.layers_begin(); it != net.layers_end(); it++) {
		af::print("weights", (*it)->getWeights());
		af::print("bias", (*it)->getBias());
	}


	Dataset result = net.predict(Xset);
	af::print("result", result.data());

	net.save("xor_trained_net.h5", true);

	SequentialNeuralNet net2(BACKEND);
	net2.load("xor_trained_net.h5");
	for(auto it = net2.layers_begin(); it != net2.layers_end(); it++) {
		af::print("loaded weights", (*it)->getWeights());
		af::print("loaded bias", (*it)->getBias());
	}

	auto it1 = net.layers_begin();
	auto it2 = net2.layers_begin();
	for (;it1 != net.layers_end() && it2 != net2.layers_end(); it1++, it2++) {
		ASSERT_TRUE(af::allTrue<bool>((*it1)->getWeights() == (*it2)->getWeights()));
		ASSERT_TRUE(af::allTrue<bool>((*it1)->getBias() == (*it2)->getBias()));
		//TODO: Compare Type
	}
	if (it1 != net.layers_end() || it2 != net2.layers_end()) {
		FAIL() << "ANN loaded from file does not have the same number of layers as ANN in memory";
	}

}

TEST_ALL(ANN_TEST, SAVE_LOAD_TEST) {
	const char *filename = "save_load_test_file.h5";
	using juml::SequentialNeuralNet;
	using juml::ann::make_SigmoidLayer;

	SequentialNeuralNet net(BACKEND);
	net.add(make_SigmoidLayer(5, 5));
	net.add(make_SigmoidLayer(5, 5));
	net.save(filename, true);

	SequentialNeuralNet net2(BACKEND);
	net2.load(filename);
	auto it1 = net.layers_begin();
	auto it2 = net2.layers_begin();
	for (;it1 != net.layers_end() && it2 != net2.layers_end(); it1++, it2++) {
		ASSERT_TRUE(af::allTrue<bool>((*it1)->getWeights() == (*it2)->getWeights())) << "Failed for " << BACKEND;
		ASSERT_TRUE(af::allTrue<bool>((*it1)->getBias() == (*it2)->getBias())) << "Failed for " << BACKEND;
		//TODO: Compare Type
	}
	if (it1 != net.layers_end() || it2 != net2.layers_end()) {
		FAIL() << "ANN loaded from file does not have the same number of layers as ANN in memory";
	}
}


static const std::string FILE_PATH = "../../../datasets/iris_ann.h5";
static const std::string SAMPLES = "samples";
static const std::string LABELS = "labels";


TEST_ALL(ANN_TEST, IRIS_TEST) {
	af::info();
	af::setBackend((af::Backend)BACKEND);
	using juml::ann::Layer;
	juml::SequentialNeuralNet net(AF_BACKEND_CPU);
	net.add(juml::ann::make_SigmoidLayer(4, 100));
	net.add(juml::ann::make_SigmoidLayer(100, 3));
	juml::Dataset X(FILE_PATH, SAMPLES);
	juml::Dataset y(FILE_PATH, LABELS);

	net.fit(X, y);
}

TEST_ALL(ANN_TEST, INCOMPATIBLE_LAYERS) {
	using juml::ann::Layer;
	using juml::ann::FunctionLayer;
	using juml::ann::Activation;
	juml::SequentialNeuralNet net(BACKEND);
	try {
		net.add(std::make_shared<FunctionLayer<Activation::Sigmoid>>(4, 5));
		net.add(std::make_shared<FunctionLayer<Activation::Sigmoid>>(3, 2));
		FAIL() << "SequentialNeuralNet did not notice incompatible layers";
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
