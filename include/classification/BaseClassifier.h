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

#include <mpi.h>

#include "core/Algorithm.h"
#include "data/Dataset.h"
#include "preprocessing/ClassNormalizer.h"

namespace juml {
    class BaseClassifier : public Algorithm {
    protected:        
        ClassNormalizer class_normalizer_;
        
        MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;

    public:
        BaseClassifier(int backend, MPI_Comm comm=MPI_COMM_WORLD);

        virtual void fit(Dataset& X, Dataset& y);
        virtual Dataset predict(Dataset& X) = 0;
        virtual float accuracy(Dataset& X, Dataset& y) = 0;
    };
} // namespace juml

#endif // BASECLASSIFIER_H

