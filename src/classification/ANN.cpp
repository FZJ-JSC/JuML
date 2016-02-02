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

float SequentialNeuralNet::accuracy(const Dataset& X, const Dataset& y) const {
	throw std::runtime_error("not yet implemented");
}



} //end of namespace
