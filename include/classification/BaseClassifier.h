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

    public:
        BaseClassifier(int backend, MPI_Comm comm=MPI_COMM_WORLD);

        virtual void fit(Dataset& X, Dataset& y);
        virtual Dataset predict(const Dataset& X) const = 0;
        virtual float accuracy(const Dataset& X, const Dataset& y) const = 0;
    };
} // namespace juml

#endif // BASECLASSIFIER_H

