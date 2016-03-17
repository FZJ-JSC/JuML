/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: CART.cpp
*
* Description: Implementation of class CART
*
* Maintainer: p.glock
*
* Email: phil.glock@gmail.com
*/

#include "core/Backend.h"
#include "core/MPI.h"
#include "classification/CART.h"

namespace juml {
    CART::CART(int backend, MPI_Comm comm)
      : BaseClassifier(backend, comm)
    {}

    void CART::build_tree(af::array& X, af::array& y) {
        for (int rank = 0; rank < this->mpi_size_; ++rank) {
            af::array data;
            if (rank == this->mpi_rank_) {
                data = X;
            }
            mpi::broadcast_inplace(data, rank, this->comm_);
        }

        
    }

    void CART::fit(Dataset& X, Dataset& y) {
        Backend::set(this->backend_.get());
        X.load_equal_chunks();
        y.load_equal_chunks();

        BaseClassifier::fit(X, y);
        af::array& X_ = X.data();
        af::array& y_ = y.data();

        this->build_tree(X_, y_);
    }

    Dataset CART::predict(Dataset& X) const {
        af::array foo;
        return Dataset(foo, this->comm_);
    }

} // namespace juml
