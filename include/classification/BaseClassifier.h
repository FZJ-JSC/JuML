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

    public:
        BaseClassifier(MPI_Comm comm=MPI_COMM_WORLD) :
            comm_(comm)
        {};

        virtual void fit(const arma::Mat<float>& X, const arma::Col<int>& y) {
            this->class_normalizer_.index(y);
        };

        virtual arma::Col<int> predict(const arma::Mat<float>& X) const = 0;
        virtual float accuracy(const arma::Mat<float>& X, const arma::Col<int>& y) const = 0;
    };
} // namespace juml

#endif // BASECLASSIFIER_H
