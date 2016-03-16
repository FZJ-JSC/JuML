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

namespace juml {
    Algorithm::Algorithm(int backend, MPI_Comm comm) 
      : backend_(backend), comm_(comm) {
        MPI_Comm_rank(this->comm_, &this->mpi_rank_);
        MPI_Comm_size(this->comm_, &this->mpi_size_);  
    }
} // namespace juml
