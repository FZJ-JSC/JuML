/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: DataSet.h
*
* Description: Header of class DataSet
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <armadillo>
#include <math.h>

namespace juml {
    template <typename T>
    T gaussian_pdf(T x, T mean, T stddev) {
        const T sigma = std::pow(stddev, 2);
        const T prob = 1.0 / std::sqrt(2.0 * M_PI * sigma);
        const T e = std::exp(-std::pow(x - mean, 2) / (2.0 * sigma));

        return prob * e;
    }

    template <typename T>
    arma::Row<T> gaussian_pdf(const arma::Row<T>& X, const arma::Row<T>& means, const arma::Row<T>& stddevs) {
        arma::Row<T> probs(X.n_elem);

        for (size_t i = 0; i < X.n_elem; ++i) {
            probs(i) = gaussian_pdf(X(i), means(i), stddevs(i));
        }

        return probs;
    }
} // juml

#endif // DISTRIBUTIONS_H
