/*
 * Copyright (c) 2015
 * Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
 *
 * This software may be modified and distributed under the terms of BSD-style license.
 *
 * File name: Distances.cpp
 *
 * Description: Implementation of distance metrics
 *
 * Maintainer: m.goetz
 *
 * Email: murxman@gmail.com
 */

#include <stdexcept>

#include "spatial/Distances.h"

namespace juml {
    af::array euclidean(const af::array& from, const af::array& to) {
        if (from.dims(2) > 1 || from.dims(3) > 1 || to.dims(2) > 1 || to.dims(3) > 1) {
            throw std::invalid_argument("euclidean distance only supports two-dimensional input");
        }

        dim_t f = from.dims(0); // features
        dim_t k = to.dims(1);   // number of to destinations
        dim_t n = from.dims(1); // number of samples

        if (f != to.dims(0)) {
            throw std::invalid_argument("from and to must have same number of features");
        }

        // reshape the data to allow vectorization, we basically want to have to volumes (3D), that have the following
        // dimensions: features x samples x k repetitions.
        af::array data_volume = af::tile(from, 1, 1, k);
        af::array centroids_volume = af::tile(af::moddims(to, f, 1, k), 1, n, 1);

        // calculate the actual distance
        return af::moddims(af::sqrt(af::sum(af::pow(centroids_volume - data_volume, 2), 0 /* along features */)), n, k);
    }

    af::array manhattan(const af::array& from, const af::array& to) {
        return af::array();
    }
} // namespace juml
