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

    af::array euclidean(const af::array& from, const af::array& to);
    af::array manhattan(const af::array& from, const af::array& to);
} // juml

#endif // DISTANCES_H

