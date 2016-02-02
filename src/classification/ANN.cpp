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
	X.load_equal_chunks();
	if (X.n_features() != this->layers[0]->input_count) {
		throw std::runtime_error("Number of features does not match inputs to neural net in first layer");
	}
	y.load_equal_chunks();
	ClassNormalizer normalizer(this->comm_);
	normalizer.index(y);
	if (normalizer.n_classes() != this->layers.back()->node_count) {
		throw std::runtime_error("Number of nodes in last layer does not match number of classes");
	}
	throw std::runtime_error("not yet implemented");
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
