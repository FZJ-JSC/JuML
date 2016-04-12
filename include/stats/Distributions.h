/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: DataSet.h
*
* Description: Collection of statistical distributions
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef JUML_STATS_DISTRIBUTIONS_H_
#define JUML_STATS_DISTRIBUTIONS_H_

#include <arrayfire.h>

namespace juml {
    /**
     * gaussian_pdf
     *
     * Calculates the gaussian (normal) probability function
     *
     * @param X      - The locations for which to calculate the probability values
     * @param mean   - The mean values of the gaussian curves
     * @param stddev - The standard deviations of the gaussian curves
     * @returns The probability values
     */
    template <typename T>
    T gaussian_pdf(const T& X, const T& mean, const T& stddev) {
        af::array result = gaussian_pdf(af::constant(X, 1), af::constant(mean, 1), af::constant(stddev, 1));
        return result.scalar<T>();
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
} // namespace juml

#endif // JUML_STATS_DISTRIBUTIONS_H_

