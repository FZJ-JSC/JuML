/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Distances.h
*
* Description: Collection of spatial distance metrics
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef DISTANCES_H
#define DISTANCES_H

#include <arrayfire.h>

namespace juml {
    typedef af::array (*Distance)(const af::array&, const af::array&);

    /**
     * euclidean
     *
     * Calculates the euclidean distance matrix of two, multi-dimensional point sets.
     *
     * @param   from - the source points, must be a two-dimensional matrix with n x f items
     * @param   to - the destination points, must be a two-dimensional matrix with k x f items
     * @returns A n x k matrix that contains the distance from a singular from point to all to points in each row
     * @throws  invalid_argument, if from or to has more then two dimensions or from and to have varying feature count
     */
    af::array euclidean(const af::array& from, const af::array& to);

    /**
     * manhattan
     *
     * Calculates the manhattan distance matrix of two, multi-dimensional point sets.
     *
     * @param   from - the source points, must be a two-dimensional matrix with n x f items
     * @param   to - the destination points, must be a two-dimensional matrix with k x f items
     * @returns A n x k matrix that contains the distance from a singular from point to all to points in each row
     * @throws  invalid_argument, if from or to has more then two dimensions or from and to have varying feature count
     */
    af::array manhattan(const af::array& from, const af::array& to);
} // juml

#endif // DISTANCES_H
