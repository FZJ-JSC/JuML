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
	throw std::runtime_error("not yet implemented");
}

float SequentialNeuralNet::accuracy(const Dataset& X, const Dataset& y) const {
	throw std::runtime_error("not yet implemented");
}



} //end of namespace
