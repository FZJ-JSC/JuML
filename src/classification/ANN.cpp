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
	X.load_equal_chunks();
	af::array result = this->predict_array(X.data());
	return Dataset(result, this->comm_);
}

af::array SequentialNeuralNet::predict_array(af::array X) const {
	if (this->layers.size() == 0) {
		throw std::runtime_error("Need at least 1 layer");
	}
	const_cast<SequentialNeuralNet*>(this)->forward_all(X);
	return this->layers.back()->getLastOutput();
}

af::array SequentialNeuralNet::classify_array(af::array X) const {
	af::array result = this->predict_array(X);
	af::array values, idxs;
	af::max(values, idxs, result, 0);
	return idxs;
}

Dataset SequentialNeuralNet::classify(Dataset& X) const {
	X.load_equal_chunks();
	af::array result = this->classify_array(X.data());
	return Dataset(result, this->comm_);
}

int SequentialNeuralNet::classify_accuracy_array(const af::array X, const af::array y) const {
	af::array classes = this->classify_array(X);
	int correct = af::count<int>(classes == y);
	MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_INT, MPI_SUM, this->comm_);
	return correct;
}

float SequentialNeuralNet::classify_accuracy(Dataset& X, Dataset& y) const {
	X.load_equal_chunks();
	y.load_equal_chunks();
	int correct = classify_accuracy_array(X.data(), y.data());
	int all = X.data().dims(1);
	MPI_Allreduce(MPI_IN_PLACE, &all, 1, MPI_INT, MPI_SUM, this->comm_);

	return ((float)correct)/all;
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
			error += this->fitBatch(sample, target, learningrate);
		}
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

float SequentialNeuralNet::fitBatch(af::array batch, af::array target, float learningrate) {
	this->forward_all(batch);
	// Oxb = Oxb - Oxb
	af::array delta = this->layers.back()->getLastOutput() - target;
	this->backwards_all(batch, delta);

	float error = af::sum<float>(af::sqrt(af::sum(delta * delta, 0)));
	MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_FLOAT, MPI_SUM, this->comm_);
	int fullbatchsize = batch.dims(1);
	MPI_Allreduce(MPI_IN_PLACE, &fullbatchsize, 1, MPI_FLOAT, MPI_SUM, this->comm_);


	for (auto it = this->layers.begin(); it != this->layers.end(); ++it) {
		(*it)->updateWeights(learningrate, this->comm_);
	}
	return error / fullbatchsize;
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

void write_2d_array_into_hdf5(hid_t id, const char *name, const af::array& array) {
	hsize_t dims[2] = {
		static_cast<hsize_t>(array.dims(1)),
		static_cast<hsize_t>(array.dims(0))
	};
	hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
	if (dataspace_id < 0) {
		throw std::runtime_error("Can not create dataspace");
	}
	hid_t dataset_id = H5Dcreate2(id, name, H5T_NATIVE_FLOAT, dataspace_id,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dataset_id < 0) {
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
	if (file_id < 0) {
		std::stringstream errMsg;
		errMsg << "Could not save SequentialNeuralNet to " << filename << " HDF5 Error Code: " << file_id;
		throw std::runtime_error(errMsg.str());
	}
	for (int i = 0; i < this->layers.size(); i++) {
		std::stringstream groupname;
		groupname << i << "_layer";
		hid_t group_id = H5Gcreate(file_id, groupname.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		if (group_id < 0) {
			std::stringstream errMsg;
			errMsg << "Could not create group " << groupname.str() 
				<< " in " << filename << " HDF5 Error: " << group_id;
			throw std::runtime_error(errMsg.str());
		}

		//TODO: need to catch possible exceptions from write_array_into_hdf5!
		write_2d_array_into_hdf5(group_id, "weights", this->layers[i]->getWeights());
		write_2d_array_into_hdf5(group_id, "bias", this->layers[i]->getBias());
		H5Gclose(group_id);

	}
	H5Fclose(file_id);
}

af::array read_hdf5_into_2d_array(hid_t id, const char *name) {
	hid_t dataset_id = H5Dopen(id, name, H5P_DEFAULT);
	if (dataset_id < 0) {
		throw std::runtime_error("Can not open dataset");
	}
	hid_t dset_space = H5Dget_space(dataset_id);
	if (dset_space < 0) {
		throw std::runtime_error("Could not get Dataspace from dataset");
	}
	int rank = H5Sget_simple_extent_ndims(dset_space);
	if (rank < 0) {
		throw std::runtime_error("Could not determine number of dimensions");
	}
	if (rank > 2) {
		throw std::runtime_error("Can only read 2 dimensional Datasets into af::array for now");
	}
	hsize_t dims[2];
	if (H5Sget_simple_extent_dims(dset_space, dims, NULL) < 0) {
		throw std::runtime_error("Could not read dimensions from Datast");
	}

	if (rank == 1) dims[1] = 1;
	//We transpose the data here, by swapping dimension numbers and by later reading Row-Major into Column-Major
	dim_t afdims[2] = {static_cast<dim_t>(dims[1]), static_cast<dim_t>(dims[0]) };

	af::array out(afdims[0], afdims[1]);
	if (af::getBackendId(out) == AF_BACKEND_CPU) {
		H5Dread(dataset_id, H5T_NATIVE_FLOAT, dset_space, dset_space, H5P_DEFAULT, out.device<float>());
		out.unlock();
	} else {
		size_t size = afdims[0] * afdims[1];
		float* buffer = new float[size];
		H5Dread(dataset_id, H5T_NATIVE_FLOAT, dset_space, dset_space, H5P_DEFAULT, buffer);
		af_write_array(out.get(), buffer, size * sizeof(float), afHost);
		delete[] buffer;
	}
	H5Sclose(dset_space);
	H5Dclose(dataset_id);

	return out;
}

void SequentialNeuralNet::load(std::string filename) {
	if (this->layers.size() != 0) {
		throw std::runtime_error("Network is already populated with layers.");
	}
	hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file_id < 0) {
		throw std::runtime_error("Could not open file to read SequentialNeuralNet");
	}

	int i = 0;
	while (true) {
		std::stringstream groupname;
		groupname << i << "_layer";
		if (!H5Lexists(file_id, groupname.str().c_str(), H5P_DEFAULT)) {
			//The group for the next layer does not exist, so we stop here.
			break;
		}
		hid_t group_id = H5Gopen2(file_id, groupname.str().c_str(), H5P_DEFAULT);
		if (group_id < 0) {
			throw std::runtime_error("Could not open group");
		}
		af::array weights = read_hdf5_into_2d_array(group_id, "weights");
		af::array bias = read_hdf5_into_2d_array(group_id, "bias");

		//TODO create the correct layer type
		this->add(ann::make_SigmoidLayer(weights, bias));

		H5Gclose(group_id);
		i+=1;
	}
	if (i == 0) {
		throw std::runtime_error("Could not load any layers from the file");
	}

	H5Fclose(file_id);
}

} //end of namespace
