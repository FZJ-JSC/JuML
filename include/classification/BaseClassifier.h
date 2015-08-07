/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BaseClassifier.h
*
* Description: Header of class BaseClassifier
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef BASECLASSIFIER_H
#define BASECLASSIFIER_H

#include <armadillo>
#include <mpi.h>

#include "preprocessing/ClassNormalizer.h"

namespace juml {
    class BaseClassifier {
    protected:
        MPI_Comm comm_;
        ClassNormalizer class_normalizer_;
        int mpi_size_;
        int mpi_rank_;

    public:
        BaseClassifier(MPI_Comm comm=MPI_COMM_WORLD) 
            : comm_(comm), class_normalizer_(comm) {
            MPI_Comm_size(this->comm_, this->mpi_size_);
            MPI_Comm_rank(this->comm_, this->mpi_rank_);
        };

        virtual inline void fit(const Dataset& X, const Dataset& y) {
            this->class_normalizer_.index(y);
        };

        virtual Dataset predict(const Dataset& X) const = 0;
        virtual float accuracy(const Dataset& X, const Dataset& y) const = 0;
    };
} // namespace juml

#endif // BASECLASSIFIER_H
