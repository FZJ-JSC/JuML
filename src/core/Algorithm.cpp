/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Algorithm.cpp
*
* Description: Implementation of class Algorithm
*
* Maintainer:
*
* Email:
*/

#include "core/Algorithm.h"
#include "core/HDF5.h"

#include <stdexcept>

namespace juml {
    Algorithm::Algorithm(int backend, MPI_Comm comm) 
      : backend_(backend), comm_(comm) {
        MPI_Comm_rank(this->comm_, &this->mpi_rank_);
        MPI_Comm_size(this->comm_, &this->mpi_size_);
    }

    void Algorithm::save(const std::string &filename, bool override) const {
        if (this->mpi_rank_ == 0) {
            if (!override && hdf5::exists(filename)) {
                throw std::runtime_error("file exists!");
            }
        }
    }
} // namespace juml
