/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: CART.h
*
* Description: Header of class CART
*
* Maintainer: p.glock
*
* Email: phil.glock@gmail.com
*/

#ifndef CART_H
#define CART_H

#include <arrayfire.h>
#include <mpi.h>

#include "core/Backend.h"
#include "classification/BaseClassifier.h"
#include "data/Dataset.h"

namespace juml {
    class CART : public BaseClassifier {
        void build_tree(af::array& X, af::array& y);

    public:
        CART(int backend=Backend::CPU, MPI_Comm comm=MPI_COMM_WORLD);

        virtual void fit(Dataset& X, Dataset& y);
        virtual Dataset predict(Dataset& X) const;
    };
} // namespace juml

#endif // CART_H
