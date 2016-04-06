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

#include "core/HDF5.h"

namespace juml {
namespace hdf5 {

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

    hid_t create_file(const std::string& filename) {
        // create a file and close property list identifier
        hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        return file_id;

    }

    void close_file(hid_t file_id) {
        H5Fclose(file_id);
    }

    void write_array(hid_t file_id, const std::string& dataset, const af::array& data){

        // create dataspace for dataset
        unsigned int dimensions = data.numdims();
        hsize_t dims[dimensions];

        dims[0] = static_cast<hsize_t>(data.dims(1));
        dims[1] = static_cast<hsize_t>(data.dims(0));
        for (unsigned int i = 2; i < dimensions; ++i) {
            dims[i] = static_cast<hsize_t>(data.dims(i));
        }
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
} // namespace hdf5
} // namespace juml