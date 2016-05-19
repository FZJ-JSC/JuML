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
#include "spatial/Distances.h"
#include "stats/Distributions.h"

namespace juml {
    /**
     * init
     *
     * Initializes the JuML execution environment. Must be called before using any other JuML library call.
     *
     * @param argc - a pointer to the arguments count of main()
     * @param argv - a pointer to the actual arguments of main()
     */
    int init(int* argc, char*** argv) {
        return MPI_Init(argc, argv);
    }

    /**
     * finalize
     *
     * Terminates the JuML execution environment. Must be called prior to exiting main(). All subsequent calls to the
     * JuML library stack yield undefined behavior.
     */
    int finalize() {
        return MPI_Finalize();
    }
} // namespace juml
