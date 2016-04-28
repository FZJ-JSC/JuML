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
    af::dtype h5_to_af(hid_t h5_type);
    hid_t af_to_h5(af::dtype af_type);
    bool exists(const std::string& filename);
    hid_t open_file(const std::string &filename);
    void close_file(hid_t& file_id);
    void write_array(hid_t file_id, const std::string& dataset, const af::array& data);
    hid_t popen_file(const std::string &filename, MPI_Comm comm);
    af::array pread_array(hid_t file_id, const std::string& dataset);
} // namespace hdf5
} // namespace juml

#endif //JUML_HDF5_H