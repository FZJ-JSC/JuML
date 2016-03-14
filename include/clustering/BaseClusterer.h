/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BaseClusterer.h
*
* Description: Header of class BaseClusterer
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef BASECLUSTERER_H
#define BASECLUSTERER_H

#include <armadillo>
#include <mpi.h>

#include "data/Dataset.h"

namespace juml {
    class BASECLUSTERER_H {
    protected:
        MPI_Comm comm_;
        int mpi_size_;
        int mpi_rank_;

    public:
        BaseClassifier(MPI_Comm comm=MPI_COMM_WORLD) 
            : comm_(comm) {
            MPI_Comm_size(this->comm_, &this->mpi_size_);
            MPI_Comm_rank(this->comm_, &this->mpi_rank_);
        };

        virtual inline void fit(Dataset<float>& X, Dataset<int>& y) = 0;
        virtual Dataset<int> predict(const Dataset<float>& X) const = 0;
        virtual float accuracy(const Dataset<float>& X, const Dataset<int>& y) const = 0;
    };
} // namespace juml

#endif // BASECLUSTERER_H

