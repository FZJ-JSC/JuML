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

#ifndef JUML_CORE_ALGORITHM_H_
#define JUML_CORE_ALGORITHM_H_

#include <mpi.h>
#include "core/Backend.h"


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
        explicit Algorithm(int backend = Backend::CPU, MPI_Comm comm = MPI_COMM_WORLD);
    }; // Algorithm
}  // namespace juml
#endif // JUML_CORE_ALGORITHM_H_
