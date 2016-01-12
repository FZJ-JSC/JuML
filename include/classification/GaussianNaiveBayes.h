/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: GaussianNaiveBayes.h
*
* Description: Header of class GaussianNaiveBayes
*
* Maintainer: p.glock
*
* Email: phil.glock@gmail.com
*/

#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

#include <armadillo>
#include <mpi.h>

#include "classification/BaseClassifier.h"
#include "data/Dataset.h"

namespace juml {
    class GaussianNaiveBayes : public BaseClassifier {
    protected:
        af::array class_counts_;
        af::array prior_;
        af::array theta_;
        af::array stddev_;

    public:
        void fit(Dataset& X, Dataset& y);
        Dataset predict(const Dataset& X) const;
        Dataset predict_probability(const Dataset& X) const;
        float accuracy(const Dataset& X, const Dataset& y) const;

        inline const af::array& class_counts() const {
            return this->class_counts_;
        };
        inline const af::array& prior() const {
            return this->prior_;
        };
        inline const af::array& theta() const {
            return this->theta_;
        };
        inline const af::array& stddev() const {
            return this->stddev_;
        };
    };
} // namespace juml

#endif // GAUSSIANNAIVEBAYES_H
