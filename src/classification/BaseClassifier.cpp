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
    {}

    void BaseClassifier::fit(Dataset& X, Dataset& y) {
        this->class_normalizer_.index(y);
    }
}  // namespace juml
