/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: GaussianNaiveBayes.h
*
* Description: Convenience wrappers for HDF5 I/O operations
*
* Maintainer: p.glock
*
* Email: phil.glock@gmail.com
*/

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "core/HDF5.h"

namespace juml {
namespace hdf5 {

    af::dtype h5_to_af(hid_t h5_type) {
        if (H5Tequal(h5_type, H5T_NATIVE_CHAR))    return u8;
        else if (H5Tequal(h5_type, H5T_NATIVE_UCHAR))   return u8;
        else if (H5Tequal(h5_type, H5T_NATIVE_B8))      return b8;
        else if (H5Tequal(h5_type, H5T_NATIVE_SHORT))   return s16;
        else if (H5Tequal(h5_type, H5T_NATIVE_USHORT))  return u16;
        else if (H5Tequal(h5_type, H5T_NATIVE_INT))     return s32;
        else if (H5Tequal(h5_type, H5T_NATIVE_UINT))    return u32;
        else if (H5Tequal(h5_type, H5T_NATIVE_LONG))    return s64;
        else if (H5Tequal(h5_type, H5T_NATIVE_LLONG))   return s64;
        else if (H5Tequal(h5_type, H5T_NATIVE_ULONG))   return u64;
        else if (H5Tequal(h5_type, H5T_NATIVE_ULLONG))  return u64;
        else if (H5Tequal(h5_type, H5T_NATIVE_FLOAT))   return f32;
        else if (H5Tequal(h5_type, H5T_NATIVE_DOUBLE))  return f64;
        else if (H5Tequal(h5_type, H5T_NATIVE_LDOUBLE)) return f64;
        throw std::domain_error("Unsupported HDF5 type");
    }

    hid_t af_to_h5(af::dtype af_type) {
        if  (af_type == u8)     return H5T_NATIVE_CHAR;
        else if (af_type == b8)     return H5T_NATIVE_B8;
        else if (af_type == s16)    return H5T_NATIVE_SHORT;
        else if (af_type == u16)    return H5T_NATIVE_USHORT;
        else if (af_type == s32)    return H5T_NATIVE_INT;
        else if (af_type == u32)    return H5T_NATIVE_UINT;
        else if (af_type == s64)    return H5T_NATIVE_LONG;
        else if (af_type == u64)    return H5T_NATIVE_ULONG;
        else if (af_type == f32)    return H5T_NATIVE_FLOAT;
        else if (af_type == f64)    return H5T_NATIVE_DOUBLE;

        throw std::domain_error("Unsupported af type");
    }

    bool exists(const std::string& filename) {
        std::ifstream file(filename.c_str());
        return file.good();
    }

    hid_t open_file(const std::string &filename) {
        // create a file and close property list identifier
        hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0 ) {
            file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        }

        return file_id;
    }

    void close_file(hid_t& file_id) {
        H5Fclose(file_id);
    }

    void write_array(hid_t file_id, const std::string& dataset, const af::array& data){

        // create dataspace for dataset
        // min dimension has to be 2, else dimension swap causes data loss
        unsigned int min_dim = 2;
        unsigned int dimensions = std::max(data.numdims(), min_dim);
        hsize_t dims[dimensions];

        for (unsigned int i = 0; i < dimensions; ++i) {
            dims[i] = static_cast<hsize_t>(data.dims(i));
        }
        std::reverse(dims, dims + dimensions);
        hid_t filespace = H5Screate_simple(dimensions, dims, NULL);
        // create dataset and close filespace
        hid_t type = af_to_h5(data.type());
        hid_t dset_id = H5Dcreate(file_id, dataset.c_str(), type, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(filespace);

        data.eval();
        if (af::getBackendId(data) == AF_BACKEND_CPU) {
            unsigned char* dump_data = data.device<unsigned char>();
            herr_t status = H5Dwrite(dset_id, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, dump_data);
        } else {
            unsigned char* dump_data = new unsigned char[data.bytes()];
            data.host(dump_data);
            herr_t status = H5Dwrite(dset_id, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, dump_data);
            delete[] dump_data;
        }
        data.unlock();

        H5Dclose(dset_id);
    }

    hid_t popen_file(const std::string &filename, MPI_Comm comm) {
        // create access list for parallel IO
        hid_t access_plist = H5Pcreate(H5P_FILE_ACCESS);
        if (access_plist < 0)
            throw std::runtime_error("Could not create file access property list");
        H5Pset_fapl_mpio(access_plist, comm, MPI_INFO_NULL);

        // create file handle
        const hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, access_plist);
        if (file_id < 0) {
            std::stringstream error;
            error << "Could not open file " << filename;
            throw std::runtime_error(error.str().c_str());
        }

        H5Pclose(access_plist);
        return file_id;
    }

    af::array pread_array(hid_t file_id, const std::string& dataset) {
        hid_t dataset_id = H5Dopen(file_id, dataset.c_str(), H5P_DEFAULT);
        // determine the dataset type
        hid_t native_type = H5Tget_native_type(H5Dget_type(dataset_id), H5T_DIR_ASCEND);

        // create file space
        const hid_t file_space_id = H5Dget_space(dataset_id);

        // get dimensions
        const int n_dims = H5Sget_simple_extent_ndims(file_space_id);
        hsize_t dimensions[n_dims];
        H5Sget_simple_extent_dims(file_space_id, dimensions, NULL);
        // get data type
        af::dtype array_type;
        try {
            array_type = h5_to_af(native_type);
        } catch(const std::domain_error& e) {
            H5Tclose(native_type);
            H5Sclose(file_space_id);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            throw e;
        }

        // initialize the array, swap the row and column dimensions before (HDF5 row-major, AF column-major)
        af::dim4 arrayDim4;
        if (n_dims > 1) {
            std::reverse(dimensions, dimensions + n_dims);
            arrayDim4 = af::dim4(n_dims, reinterpret_cast<dim_t*>(dimensions));
        } else if (n_dims == 1) {
            arrayDim4 = af::dim4(1, dimensions[0]);
        }
        af::array data = af::array(arrayDim4, array_type);
        data.eval();
        // read the actual data
        if (af::getBackendId(af::constant(0, 1)) == AF_BACKEND_CPU) {
            H5Dread(dataset_id, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.device<uint8_t>());
            data.unlock();
        } else {
            size_t size = data.bytes();
            uint8_t* buffer = new uint8_t[size];
            H5Dread(dataset_id, native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);
            af_write_array(data.get(), buffer, size, afHost);
            delete[] buffer;
        }

        H5Tclose(native_type);
        H5Sclose(file_space_id);
        H5Dclose(dataset_id);
        return data;
    }
} // namespace hdf5
} // namespace juml