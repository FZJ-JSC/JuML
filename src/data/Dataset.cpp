/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Dataset.cpp
*
* Description: Implementation of class Dataset
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#include <exception>
#include <hdf5.h>
#include <math.h>
#include <mpi.h>
#include <sstream>

#include "data/Dataset.h"

#include <iostream>

namespace juml {
    Dataset::Dataset(const std::string& filename, const std::string& dataset, const MPI_Comm comm)
        : filename_(filename), dataset_(dataset), comm_(comm) {
        MPI_Comm_rank(this->comm_, &this->mpi_rank_);
        MPI_Comm_size(this->comm_, &this->mpi_size_);
    };

    void Dataset::load_equal_chunks() {
        // create access list for parallel IO
        hid_t access_plist = H5Pcreate(H5P_FILE_ACCESS);
        if (access_plist < 0) {
            throw std::runtime_error("Could not create file access property list");
        }
        H5Pset_fapl_mpio(access_plist, this->comm_, MPI_INFO_NULL);

        // create file handle 
        const hid_t file_id = H5Fopen(this->filename_.c_str(), H5F_ACC_RDWR, access_plist);
        if (file_id < 0) {
            std::stringstream error;
            error << "Could not open file " << this->filename_;
            throw std::runtime_error(error.str().c_str());
        }

        // read data set
        const hid_t data_id = H5Dopen(file_id, this->dataset_.c_str(), H5P_DEFAULT);
        if (data_id < 0) {
            std::stringstream error;
            error << "Could not open dataset " << this->dataset_ << " in file " << this->filename_;
            throw std::runtime_error(error.str().c_str());
        }
        
        // create file space
        const hid_t file_space_id = H5Dget_space(data_id);
        if (file_space_id < 0) {
            std::stringstream error;
            error << "Could not get file space of file " << this->filename_;
            throw std::runtime_error(error.str().c_str());
        }
        
        // check dimesionality of the dataset
        const int n_dims = H5Sget_simple_extent_ndims(file_space_id);
        if (n_dims > 2) {
            H5Dclose(data_id);
            H5Fclose(file_id);
            H5Pclose(access_plist);
            throw std::domain_error("More than two dimensions are currently not supported.");
        } else if (n_dims < 1) {
            std::stringstream error;
            error << "Got " << n_dims << " in dataset " << this->dataset_ << " in file " << this->filename_;
            throw std::domain_error(error.str().c_str());
        }

        // calculate offsets into the hyperslab
        hsize_t dimensions[n_dims];
        H5Sget_simple_extent_dims(file_space_id, dimensions, NULL);
        hsize_t overlap = dimensions[0] % this->mpi_size_;
        hsize_t offset = 0;
        hsize_t chunk_size = (dimensions[0] / this->mpi_size_);

        if (overlap > this->mpi_rank_) {
            chunk_size += 1;
        } else {
            offset = overlap;
        }

        hsize_t position = offset + this->mpi_rank_ * chunk_size;
        hsize_t chunk_dimensions[n_dims];
        chunk_dimensions[0] = chunk_size;
        hsize_t row_col_offset[n_dims];
        row_col_offset[0] = position;
        for (int i = 1; i < n_dims; ++i) {
            chunk_dimensions[i] = dimensions[i];
            row_col_offset[i] = 0;
        }

        // create memory space
        hid_t mem_space = H5Screate_simple(n_dims, chunk_dimensions, NULL);
        if (mem_space < 0) {
            throw std::runtime_error("Could not create memory space");
        }
        
        // select hyperslab
        herr_t err = H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, row_col_offset, NULL, chunk_dimensions, NULL);
        if (err < 0) {
            std::stringstream error;
            error << "Could not select hyperslabe in file " << this->filename_;
            throw std::runtime_error(error.str().c_str());
        }

        int total_size = chunk_size;
        for (int i = 1; i < n_dims; ++i) {
            total_size *= dimensions[i];
        }

        // read the data
        hid_t data_type = H5Dget_type(data_id);
        H5T_class_t data_class = H5Tget_class(data_type);
        if (data_class != H5T_FLOAT) {
            H5Tclose(data_type);
            H5Sclose(mem_space);
            H5Dclose(data_id);
            H5Fclose(file_id);
            H5Pclose(access_plist);
            throw std::domain_error("Only float supported.");
        }
        float* data_points = new float[total_size];
        H5Dread(data_id, H5T_NATIVE_FLOAT, mem_space, file_space_id, H5P_DEFAULT, data_points);
        this->data_ = arma::fmat(data_points, chunk_size, n_dims < 2 ? 1 : chunk_dimensions[1], false, true);

        // release ressources
        H5Tclose(data_type);
        H5Sclose(mem_space);
        H5Dclose(data_id);
        H5Fclose(file_id);
        H5Pclose(access_plist);
    }
    
    Dataset::~Dataset() {
    }
} // namespace juml
