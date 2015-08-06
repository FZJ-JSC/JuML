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
#include <stdint.h>

#include "classification/BaseClassifier.h"

namespace juml {
    class GaussianNaiveBayes : public BaseClassifier {
    protected:
        arma::fvec class_counts_;
        arma::fvec prior_;
        arma::fmat theta_;
        arma::fmat stddev_;

    public:
        void fit(const arma::fmat& X, const arma::ivec& y);
        arma::ivec predict(const arma::fmat& X) const;
        arma::fmat predict_probability(const arma::fmat& X) const;
        float accuracy(const arma::fmat& X, const arma::ivec& y) const;

        inline const arma::fvec& class_counts() const {
            return this->class_counts_;
        };
        inline const arma::fvec& prior() const {
            return this->prior_;
        };
        inline const arma::fmat& theta() const {
            return this->theta_;
        };
        inline const arma::fmat& stddev() const {
            return this->stddev_;
        };
    };
} // namespace juml

#endif // GAUSSIANNAIVEBAYES_H
