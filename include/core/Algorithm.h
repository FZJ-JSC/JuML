/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Algorithm.h
*
* Description: Header of class Algorithm
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "core/Backend.h"

#include <mpi.h>

namespace juml {
    //! Algorithm
    //! TODO: Describe me
    class Algorithm {
    protected:
        Backend backend_;

        MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;
    public:
        Algorithm(int backend = Backend::CPU, MPI_Comm comm = MPI_COMM_WORLD);
    }; // Algorithm
}  // namespace juml
#endif // ALGORITHM_H
