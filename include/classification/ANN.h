/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: ANN.h
*
* Description: Header File that describes the Classes used for Artifical Neural Networks
*
* Maintainer: 
*
* Email: 
*/




#ifndef JUML_CLASSIFICATION_ANN_H_
#define JUML_CLASSIFICATION_ANN_H_

#include <mpi.h>

#include "classification/BaseClassifier.h"
#include "data/Dataset.h"

namespace juml {
	//TODO: interface class for ANN-layer
	class SequentialNeuralNet : BaseClassifier {
		public: 
			//TODO: Constructor that takes vector of layers
			void fit(Dataset& X, Dataset& y) override;
			Dataset predict(const Dataset& X) const override;
			float accuracy(const Dataset& X, const Dataset& y) const override;
	};
}

#endif
