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

#ifndef JUML_HDF5_H
#define JUML_HDF5_H

#include <arrayfire.h>
#include <hdf5.h>
#include <string>

namespace juml {
namespace hdf5 {
    hid_t af_to_h5(af::dtype af_type);
    hid_t create_file(const std::string& filename);
    void close_file(hid_t file_id);
    void write_array(hid_t file_id, const std::string& dataset, const af::array& data);
} // namespace hdf5
} // namespace juml

#endif //JUML_HDF5_H