/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Dataset.h
*
* Description: Header of class Dataset
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#include <mpi.h>

#include "core/Algorithm.h"
#include "core/Backend.h"
#include "data/Dataset.h"
#include "classification/GaussianNaiveBayes.h"
#include "clustering/KMeans.h"
#include "stats/Distributions.h"

namespace juml {
    int init(int* argc, char*** argv) {
        return MPI_Init(argc, argv);
    }

    int finalize() {
        return MPI_Finalize();
    }
} // namespace juml
