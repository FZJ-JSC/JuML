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
#include "core/Backend.h"
#include "data/Dataset.h"

namespace juml {
    class BaseClusterer : public Algorithm {
    public:
        /**
         * BaseClusterer constructor
         *
         * Stores the clustering processing backend and MPI communicator.
         *
         * @param backend - The processing backend, defaults to CPU
         * @oaram comm - The MPI communicator
         */
        BaseClusterer(int backend=Backend::CPU, MPI_Comm comm=MPI_COMM_WORLD);

        /**
         * (Abstract) fit
         *
         * Fits or clusters the passed training data
         *
         * @param X - The training data
         */
        virtual void fit(Dataset& X) = 0;

        /**
         * (Abstract) predict
         *
         * Predicts to which clusters new, unseen data items belong to, requires that the model was previously @see fit.
         *
         * @returns The predicted cluster assignment
         */
        virtual Dataset predict(Dataset& X) const = 0;
    };
} // namespace juml

#endif // BASECLUSTERER_H

