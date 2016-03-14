/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BaseClusterer.h
*
* Description: Header of class BaseClusterer
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef BASECLUSTERER_H
#define BASECLUSTERER_H

#include "core/Algorithm.h"
#include "data/Dataset.h"

namespace juml {
    class BaseClusterer : public Algorithm {
    public:
        BaseClusterer(int backend=Backend::CPU, MPI_Comm comm=MPI_COMM_WORLD);

        virtual void fit(Dataset& X) = 0;
        virtual Dataset predict(Dataset& X) const = 0;
    };
} // namespace juml

#endif // BASECLUSTERER_H

