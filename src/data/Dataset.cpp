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

#include "data/Dataset.h"

#include <iostream>

namespace juml {
    Dataset::Dataset(const std::string& filename, const std::string& data_set, MPI_Comm comm)
        : comm_(comm), filename_(filename_), data_set_(data_set) {

        MPI_Comm_rank(this->comm_, &this->mpi_rank_);
        MPI_Comm_size(this->comm_, &this->mpi_size_);
    }

    Dataset::~Dataset() {
    }

    void Dataset::load_equal_chunks() {
        // create access list for parallel IO
        hid_t access_plist = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(access_plist, this->comm_, MPI_INFO_NULL);

        // create file handle 
        const hid_t file_id = H5Fopen(this->filename_.c_str(), H5F_ACC_RDONLY, access_plist);

        // read data set
        const hid_t data_id = H5Dopen(file_id, this->data_set_.c_str(), access_plist);
        const hid_t file_space_id = H5Dget_space(data_id);
        const int n_dims = H5Sget_simple_extent_ndims(file_space_id);
        if (n_dims > 2) {
            H5Dclose(file_space_id);
            H5Dclose(data_id);
            H5Fclose(file_id);
            H5Pclose(access_plist);

            throw std::domain_error("More than two dimensions are currently not supported.");
        }

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

        hid_t mem_space = H5Screate_simple(n_dims, chunk_dimensions, NULL);
        H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, row_col_offset, NULL, chunk_dimensions, NULL);

        int total_size = chunk_size;
        for (int i = 1; i < n_dims; ++i) {
            total_size *= dimensions[i];
        }

        hid_t data_type = H5Dget_type(data_id);
        if (data_type != H5T_IEEE_F32BE && data_type != H5T_IEEE_F32LE) {
            H5Tclose(data_type);
            H5Sclose(mem_space);
            H5Dclose(file_space_id);
            H5Dclose(data_id);
            H5Fclose(file_id);
            H5Pclose(access_plist);

            throw std::domain_error("Only float supported.");
        }
        float* data_points = new float[total_size];
        H5Dread(data_id, H5T_IEEE_F32LE, mem_space, file_space_id, access_plist, data_points);

        this->data_ = arma::fmat(data_points, chunk_size, n_dims < 2 ? 1 : chunk_dimensions[1], false, true);

        H5Tclose(data_type);
        H5Sclose(mem_space);
        H5Dclose(file_space_id);
        H5Dclose(data_id);
        H5Fclose(file_id);
        H5Pclose(access_plist);
    }
} // namespace juml
