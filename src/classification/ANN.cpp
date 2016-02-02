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
namespace juml {

Dataset SequentialNeuralNet::predict(const Dataset& X) const  {
	throw std::runtime_error("not yet implemented");
}

void SequentialNeuralNet::fit(Dataset& X, Dataset& y) {
	//X.load_equal_chunks();
	if (X.data().dims(0) != this->layers[0]->input_count) {
		throw std::runtime_error("Number of features does not match inputs to neural net in first layer");
	}
	//y.load_equal_chunks();
	if (X.data().dims(1) != y.data().dims(0)) {
		throw std::runtime_error("X and y do not match");
	}

	ClassNormalizer normalizer(this->comm_);
	normalizer.index(y);
	if (normalizer.n_classes() != this->layers.back()->node_count) {
		throw std::runtime_error("Number of nodes in last layer does not match number of classes");
	}

	const int max_iterations = 200; //TODO: User-specify this value
	const float learningrate = 1;
	af::array& Xdata = X.data();
	af::array ydata = y.data();
	const int n_samples = Xdata.dims(1);
	const int n_features = Xdata.dims(0);
	af::array target = af::constant(0.0f, normalizer.n_classes());

	//TODO use matrix-matrix multiply for batches of inputs
	for (int iteration = 0; iteration < max_iterations; iteration++) {
		for (int i = 0; i < n_samples; i++) {
			//TODO: Class labels probably need to be transformed!
			//TODO need to generate target vector
			//target = af::iota(normalizer.n_classes()) == ydata(i);
			af::array sample = Xdata(af::span, i);
			this->forward_all(sample);
			af::array delta = this->layers.back()->getLastOutput() - target;
			this->backwards_all(sample, delta );
		}
		for (auto it = this->layers.begin(); it != this->layers.end(); ++it) {
			(*it)->updateWeights(learningrate);
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

float SequentialNeuralNet::accuracy(const Dataset& X, const Dataset& y) const {
	throw std::runtime_error("not yet implemented");
}



} //end of namespace
