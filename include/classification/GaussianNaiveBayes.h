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
        arma::Col<float> class_counts_;
        arma::Col<float> prior_;
        arma::Mat<float> theta_;
        arma::Mat<float> stddev_;

    public:
        void fit(const Dataset<float>& X, const Dataset<int>& y);
        Dataset<int> predict(const Dataset<float>& X) const;
        Dataset<float> predict_probability(const Dataset<float>& X) const;
        float accuracy(const Dataset<float>& X, const Dataset<int>& y) const;

        inline const arma::Col<float>& class_counts() const {
            return this->class_counts_;
        };
        inline const arma::Col<float>& prior() const {
            return this->prior_;
        };
        inline const arma::Mat<float>& theta() const {
            return this->theta_;
        };
        inline const arma::Mat<float>& stddev() const {
            return this->stddev_;
        };
    };
} // namespace juml

#endif // GAUSSIANNAIVEBAYES_H

