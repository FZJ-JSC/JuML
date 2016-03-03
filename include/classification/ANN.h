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
#include <memory> //For shared_ptr
#include "classification/ANNLayers.h"

#include "classification/BaseClassifier.h"
#include "data/Dataset.h"

namespace juml {
	class SequentialNeuralNet : public BaseClassifier {
		protected:
			std::vector<std::shared_ptr<ann::Layer>> &layers;
			void forward_all(const af::array& input);
			void backwards_all(const af::array& input, const af::array& delta);
		public:
			SequentialNeuralNet(int backend, std::vector<std::shared_ptr<ann::Layer>> &layers_, MPI_Comm comm=MPI_COMM_WORLD) :  BaseClassifier(backend, comm), layers(layers_) {
				if (this->layers.size() == 0) {
					throw std::runtime_error("Need at least 1 layer");
				}
				auto it = layers.begin();
				int before_node_count = (*it)->node_count;
				it++;
				int i = 1;
				for(;it != layers.end();it++) {
					if (before_node_count != (*it)->input_count) {
						std::stringstream errMsg;
						errMsg 
							<< "Layer Mismatch: Layer " 
							<< (i - 1)
							<< " has "
							<< before_node_count
							<< " Nodes, but the next layer expects "
							<< (*it)->input_count 
							<< " inputs";
						throw std::runtime_error(errMsg.str());


					}
					before_node_count = (*it)->node_count;
					i += 1;
				}

			}
			void fit(Dataset& X, Dataset& y) override;
			Dataset predict(Dataset& X) override;
			float accuracy(Dataset& X, Dataset& y) override;
	};
}

#endif
