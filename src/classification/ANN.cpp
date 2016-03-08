/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: ANN.cpp
*
* Description: Implementation of the  Classes used for Artifical Neural Networks
*
* Maintainer: 
*
* Email: 
*/



#include "classification/ANN.h"
#include <stdexcept>
#include <iostream>
namespace juml {

Dataset SequentialNeuralNet::predict(Dataset& X) {
	X.load_equal_chunks();
	this->forward_all(X.data());
	af::array result = this->layers.back()->getLastOutput();
	return Dataset(result, this->comm_);
}

void SequentialNeuralNet::fit(Dataset& X, Dataset& y) {
	X.load_equal_chunks();
	if (X.data().dims(0) != this->layers[0]->input_count) {
		std::stringstream errMsg;
		errMsg << "Number of features (" << X.data().dims(0) 
			<< ") does not match inputs to neural net in first layer ("
			<< this->layers[0]->input_count << ")";
		throw std::runtime_error(errMsg.str());
	}
	y.load_equal_chunks();
	if (X.data().dims(1) != y.data().dims(1)) {
		std::stringstream errMsg;
		errMsg << "Number of samples ("
			<< X.data().dims(1)
			<< ") does not match number of labels ("
			<< y.data().dims(1)
			<< ")";
		throw std::runtime_error(errMsg.str());
	}

	//TODO: Check that number of rows in label set matches number of nodes in the last layer of the network.

	const int max_iterations = 200; //TODO: User-specify this value
	const float learningrate = 1;
	af::array& Xdata = X.data();
	af::array ydata = y.data();
	const int n_samples = Xdata.dims(1);
	const int n_features = Xdata.dims(0);
	const int batchsize = 1;
	const float max_error = 0.001;
	af::array target(this->layers.back()->node_count, batchsize);

	for (int iteration = 0; iteration < max_iterations; iteration++) {
		float error = 0;
		for (int i = 0; i < n_samples; i += batchsize) {
			// (Number of Nodes in last layer=O)xb
			target = ydata(af::span, af::seq(i, i+batchsize - 1));
			// Fxb
			af::array sample = Xdata(af::span, af::seq(i, i+batchsize - 1));
			this->forward_all(sample);
			// Oxb = Oxb - Oxb
			af::array delta = this->layers.back()->getLastOutput() - target;
			this->backwards_all(sample, delta);

			error += af::sum<float>(delta * delta);

			for (auto it = this->layers.begin(); it != this->layers.end(); ++it) {
				(*it)->updateWeights(learningrate, this->comm_);
			}
		}
		MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_FLOAT, MPI_SUM, this->comm_);
		if (this->mpi_rank_ == 0) {
			std::cout << "Iteration " << iteration << " Error: " << (sqrt(error)) << std::endl;
		}
		if (sqrt(error) < max_error) {
			if (this->mpi_rank_ == 0) {
				std::cout << "Iteration " << iteration << " Error: " << (sqrt(error)) << std::endl;
			}
			break;
		}
	}
}

void SequentialNeuralNet::forward_all(const af::array& input) {
	auto itbefore = this->layers.begin();
	//Constructor ensures there is at least one layer
	(*itbefore)->forward(input);
	auto it = this->layers.begin();
	it++;
	for (; it != this->layers.end(); it++, itbefore++) {
		(*it)->forward((*itbefore)->getLastOutput());
	}
}

void SequentialNeuralNet::backwards_all(const af::array& input, const af::array& delta) {
	if (this->layers.size() == 1) {
		this->layers[0]->backwards(input, delta);
	} else {
		int i = this->layers.size() - 1;
		this->layers[i]->backwards(this->layers[i - 1]->getLastOutput(), delta);
		for (i--; i > 0; i--) {
			//LastOutput of layer i+1 now contains the delta
			this->layers[i]->backwards(this->layers[i - 1]->getLastOutput(), this->layers[i + 1]->getLastOutput());
		}
		this->layers[0]->backwards(input, this->layers[1]->getLastOutput());
	}
}

float SequentialNeuralNet::accuracy(Dataset& X, Dataset& y) {
	throw std::runtime_error("not yet implemented");
}



} //end of namespace
