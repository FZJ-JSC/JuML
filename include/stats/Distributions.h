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

#include <arrayfire.h>

namespace juml {
    template <typename T>
    T gaussian_pdf(const T& X, const T& mean, const T& stddev) {
        af::array result = gaussian_pdf(af::constant(X, 1), af::constant(mean, 1), af::constant(stddev, 1));
        return *(result.host<T>());
    }

    template <>
    af::array gaussian_pdf(const af::array& X, const af::array& means, const af::array& stddevs) {
        af::array sigma = af::pow(stddevs, 2);
        af::array prob = 1.0 / af::sqrt(2.0 * af::Pi * sigma);        
        af::array e = af::exp(af::pow(X - means, 2) / (-2.0 * sigma));

        return prob * e;
    }
    
    template <>
    double gaussian_pdf(const double& X, const double& mean, const double& stddev) {
        af::array result = gaussian_pdf(af::constant(X, 1, f64), af::constant(mean, 1, f64), af::constant(stddev, 1, f64));
        return *(result.host<double>());
    }
} // juml

#endif // DISTRIBUTIONS_H

