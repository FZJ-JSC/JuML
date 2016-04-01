/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BaseClassifier.cpp
*
* Description: Implementation of class BaseClassifier
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#include "classification/BaseClassifier.h"

namespace juml {
    BaseClassifier::BaseClassifier(int backend, MPI_Comm comm)
      : Algorithm(backend, comm), class_normalizer_(comm) 
    {};

    void BaseClassifier::fit(Dataset& X, Dataset& y) {
        this->class_normalizer_.index(y);
    };

    float BaseClassifier::accuracy(Dataset &X, Dataset &y) const {
        float local_results[2];
        Dataset predictions = this->predict(X);
        // X is loaded in this->predict
        y.load_equal_chunks();

        af::array sum = af::sum(predictions.data() == y.data());
        local_results[0] = (float) sum.scalar<uint>();
        local_results[1] = (float) y.n_samples();
        MPI_Allreduce(MPI_IN_PLACE, local_results, 2, MPI_FLOAT, MPI_SUM, this->comm_);

        return local_results[0] / local_results[1];
    }
} // namespace juml
