/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Algorithm.h
*
* Description: Header of class Algorithm
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef ALGORITHM_H
#define ALGORITHM_H

#include "core/Definitions.h"

namespace juml {
    //! Algorithm
    //! TODO: Describe me
    class Algorithm {
    protected:
        Backend backend_;
    public:
        Algorithm(int backend=Backend::CPU);
    }; // Algorithm
}  // juml
#endif // ALGORITHM_H
