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

Dataset SequentialNeuralNet::predict(Dataset& X) const {
	if (this->layers.size() == 0) {
		throw std::runtime_error("Need at least 1 layer");
	}
	X.load_equal_chunks();
	const_cast<SequentialNeuralNet*>(this)->forward_all(X.data());
	af::array result = this->layers.back()->getLastOutput();
	return Dataset(result, this->comm_);
}

void SequentialNeuralNet::fit(Dataset& X, Dataset& y) {
	if (this->layers.size() == 0) {
		throw std::runtime_error("Need at least 1 layer");
	}
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

	if (y.data().dims(0) != this->layers.back()->node_count) {
		std::stringstream errMsg;
		errMsg << "Number of rows in label data ("
			<< y.data().dims(0)
			<< ") does not match number of nodes in output layer ("
			<< this->layers.back()->node_count
			<< ")";
		throw std::runtime_error(errMsg.str());
	}

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
			int last_batch_index = std::min(i+batchsize - 1, (int)ydata.dims(1) - 1);
			target = ydata(af::span, af::seq(i, last_batch_index));
			// Fxb
			af::array sample = Xdata(af::span, af::seq(i, last_batch_index));
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

float SequentialNeuralNet::accuracy(Dataset& X, Dataset& y) const {
	throw std::runtime_error("not yet implemented");
}

void write_array_into_hdf5(hid_t id, const char *name, const af::array& array) {
	//This causes warnings because of conversion from long long int to long long unsigned int
	hsize_t dims[2] = {array.dims(1), array.dims(0)};
	hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
	if (dataspace_id < 0) {
		throw std::runtime_error("Can not create dataspace");
	}
	hid_t dataset_id = H5Dcreate2(id, name, H5T_NATIVE_FLOAT, dataspace_id,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dataset_id <= 0) {
		H5Sclose(dataspace_id);
		throw std::runtime_error("Can not create dataset");
	}
	bool devicePtr = af::getBackendId(array) == AF_BACKEND_CPU;
	float *dataptr;
	if (devicePtr) {
		dataptr = array.device<float>();
	} else {
		dataptr = new float[array.elements()];
	}
	H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataptr);
	if (devicePtr) {
		array.unlock();
	} else {
		delete[] dataptr;
	}
	H5Dclose(dataset_id);
	H5Sclose(dataspace_id);
}

void SequentialNeuralNet::save(std::string filename, bool overwrite) {
	//Only the master process saves the network, because its the same on every process anyway.
	if (this->mpi_rank_ != 0) return;
	hid_t file_id = H5Fcreate(filename.c_str(), overwrite ? H5F_ACC_TRUNC : H5F_ACC_EXCL,
			H5P_DEFAULT, H5P_DEFAULT);
	if (file_id >= 0) {
		for (int i = 0; i < this->layers.size(); i++) {
			std::stringstream groupname;
			groupname << i << "_layer";
			hid_t group_id = H5Gcreate(file_id, groupname.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

			af::array& layer = this->layers[i]->getWeights();
			if (group_id >= 0) {
				//TODO: need to catch possible exceptions from write_array_into_hdf5!
				write_array_into_hdf5(group_id, "weights", this->layers[i]->getWeights());
				write_array_into_hdf5(group_id, "bias", this->layers[i]->getBias());
				H5Gclose(group_id);
			} else {
				H5Fclose(file_id);
				std::stringstream errMsg;
				errMsg << "Could not create group " << groupname.str() 
					<< " in " << filename << " HDF5 Error: " << group_id;
				throw std::runtime_error(errMsg.str());
			}

		}
		H5Fclose(file_id);
	} else {
		std::stringstream errMsg;
		errMsg << "Could not save SequentialNeuralNet to " << filename << " HDF5 Error Code: " << file_id;
		throw std::runtime_error(errMsg.str());
	}
}

void SequentialNeuralNet::load(std::string filename) {

}

} //end of namespace
