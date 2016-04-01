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
			std::vector<ann::LayerPtr> layers;
			void forward_all(const af::array& input);
			void backwards_all(const af::array& input, const af::array& delta);
		public:
			SequentialNeuralNet(int backend, MPI_Comm comm=MPI_COMM_WORLD) :
				BaseClassifier(backend, comm) {}
			SequentialNeuralNet& add(ann::LayerPtr layer) {
				if (layers.size() > 0) {
					if (layers.back()->node_count != layer->input_count) {
						std::stringstream errMsg;
						errMsg << "Incompatible Layers. Last Layer has "
							<< layers.back()->node_count
							<< " Nodes, but the proposed Layer expects "
							<< layer->input_count
							<< " Inputs";
						throw std::runtime_error(errMsg.str());
					}
				}
				layers.push_back(layer);
				return *this;
			}
			std::vector<ann::LayerPtr>::iterator layers_begin() {
				return layers.begin();
			}
			std::vector<ann::LayerPtr>::iterator layers_end() {
				return layers.end();
			}
			void fit(Dataset& X, Dataset& y) override;
			float fitBatch(af::array batch, af::array target, float learningrate);
			Dataset predict(Dataset& X) const override;
			af::array predict_array(af::array X) const;

			Dataset classify(Dataset& X) const;
			af::array classify_array(af::array X) const;

			void sync();

			float accuracy(Dataset& X, Dataset& y) const override;
			float classify_accuracy(Dataset& X, Dataset &y) const;
			int classify_accuracy_array(af::array X, af::array y) const;

			void save(std::string filename, bool overwrite);
			void load(std::string filename);
	};
}

#endif
