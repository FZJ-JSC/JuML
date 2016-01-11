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

#include <stdexcept>
#include <sstream>

#include "data/Dataset.h"
#include "util/Settings.h"

namespace juml {
    //! Dataset constructor
    Dataset::Dataset(const std::string& filename, const std::string& dataset, const MPI_Comm comm)
        : filename_(filename), dataset_(dataset), comm_(comm) {
        MPI_Comm_rank(this->comm_, &this->mpi_rank_);
        MPI_Comm_size(this->comm_, &this->mpi_size_);
    }

    Dataset::Dataset(af::array& data, MPI_Comm comm)
        : data_(data), comm_(comm) {
        MPI_Comm_rank(this->comm_, &this->mpi_rank_);
        MPI_Comm_size(this->comm_, &this->mpi_size_);    
    }
    
    af::dtype Dataset::h5_to_af(hid_t h5_type) {
             if (H5Tequal(h5_type, H5T_NATIVE_CHAR))   return u8;
        else if (H5Tequal(h5_type, H5T_NATIVE_UCHAR))  return u8;
        else if (H5Tequal(h5_type, H5T_NATIVE_B8))     return b8;
        else if (H5Tequal(h5_type, H5T_NATIVE_SHORT))  return s16;
        else if (H5Tequal(h5_type, H5T_NATIVE_USHORT)) return u16;
        else if (H5Tequal(h5_type, H5T_NATIVE_INT))    return s32;
        else if (H5Tequal(h5_type, H5T_NATIVE_UINT))   return u32;
        else if (H5Tequal(h5_type, H5T_NATIVE_LONG))   return s64;
        else if (H5Tequal(h5_type, H5T_NATIVE_LLONG))   return s64;
        else if (H5Tequal(h5_type, H5T_NATIVE_ULONG))  return u64;
        else if (H5Tequal(h5_type, H5T_NATIVE_ULLONG))  return u64;
        else if (H5Tequal(h5_type, H5T_NATIVE_FLOAT))  return f32;
        else if (H5Tequal(h5_type, H5T_NATIVE_DOUBLE)) return f64;
        else if (H5Tequal(h5_type, H5T_NATIVE_LDOUBLE)) return f64;
        throw std::domain_error("Unsupported HDF5 type");
    }

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
        if (n_dims < 1 || n_dims > 4) {
            std::stringstream error;
            error << "Got " << n_dims << "dimensions in dataset " << this->dataset_ << " in file " << this->filename_ << ". Expected 1 to 4.";
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

        size_t total_size = chunk_size;
        for (int i = 1; i < n_dims; ++i) {
            total_size *= dimensions[i];
        }
        
        // read the data
        hid_t data_type = H5Dget_type(data_id);
        hid_t native_type = H5Tget_native_type(data_type, H5T_DIR_ASCEND);
        af::dtype array_type;
        try {
            array_type = h5_to_af(data_type);
        } catch(const std::domain_error& e) {
            H5Tclose(native_type);
            H5Tclose(data_type);
            H5Sclose(mem_space);
            H5Dclose(data_id);
            H5Fclose(file_id);
            H5Pclose(access_plist);
            throw e;        
        }
        
        // read actual data
        this->data_ = af::array(af::dim4(n_dims, reinterpret_cast<dim_t*>(chunk_dimensions)), array_type);
        if (Settings::getInstance().get_backend() == AF_BACKEND_CPU) {
            H5Dread(data_id, native_type, mem_space, file_space_id, H5P_DEFAULT, this->data_.get());
        } else {
            size_t size = this->data_.bytes();
            char* buffer = new char[size];
            H5Dread(data_id, native_type, mem_space, file_space_id, H5P_DEFAULT, buffer);
            af_write_array(this->data_.get(), buffer, size, afHost); 
            delete[] buffer;	
        }        
        this->data_ = this->data_.T();

        // release ressources
        H5Tclose(native_type);
        H5Tclose(data_type);
        H5Sclose(mem_space);
        H5Dclose(data_id);
        H5Fclose(file_id);
        H5Pclose(access_plist);
    }    
} // namespace juml

