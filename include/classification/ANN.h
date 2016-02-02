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
#include <vector>
#include "classification/ANNLayers.h"

#include "classification/BaseClassifier.h"
#include "data/Dataset.h"

namespace juml {
	//TODO: interface class for ANN-layer
	class SequentialNeuralNet : BaseClassifier {
		protected:
			std::vector<ann::Layer*> &layers;
			void forward_all(const af::array& input);
			void backwards_all(const af::array& input, const af::array& delta);
		public:
			SequentialNeuralNet(std::vector<ann::Layer*> &layers_, MPI_Comm comm=MPI_COMM_WORLD) :  BaseClassifier(comm), layers(layers_) {
				if (this->layers.size() == 0) {
					throw std::runtime_error("Need at least 1 layer");
				}
			}
			void fit(Dataset& X, Dataset& y) override;
			Dataset predict(const Dataset& X) const override;
			float accuracy(const Dataset& X, const Dataset& y) const override;
	};
}

#endif
