/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BaseClassifier.cpp
*
* Description: Implementation of class BaseClusterer
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#include "classification/BaseClusterer.h"

namespace juml {
    BaseClusterer::BaseClusterer(int backend, MPI_Comm comm)
      : Algorithm(backend, comm) 
    {};
} // namespace juml

