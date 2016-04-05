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

#include "core/Backend.h"

#include <mpi.h>

namespace juml {
    /**
     * Algorithm
     *
     * Abstract class, that is subclassed by all algorithms. It provides a basic interface.
     */
    class Algorithm {
    protected:
        /**
         * @var backend_
         * @brief execution backend. Holds the backend on which the algorithm is running. (@see Backend)
         */
        Backend backend_;

        /**
         * @var comm_
         * @brief mpi communicator.
         */
        MPI_Comm comm_;

        /**
         * @var mpi_rank_
         * @brief mpi rank of the process. An integer with the processes rank. Different for each mpi process.
         */
        int mpi_rank_;

        /**
         * @var mpi_size_
         * @brief mpi size of the communicator. Number of processes in the communicator. Same for each mpi process.
         */
        int mpi_size_;
    public:
        /**
         * Algorithm constructor
         * @param backend The execution backend, defaults to CPU. (@see Backend)
         * @param comm The MPI communicator for the execution, defaults to world communicator.
         */
        Algorithm(int backend=Backend::CPU, MPI_Comm comm=MPI_COMM_WORLD);

        /**
         * Abstract interface to load an algorithm from a hdf5 file.
         * @param filename The filename of the loaded file.
         */
        void load(const std::string& filename) = 0;

        /**
         * Abstract interface to save an algorithm to a hdf5 file.
         * @param filename The filename of the saved file.
         */
        void save(const std::string& filename) = 0;
    }; // Algorithm
}  // juml
#endif // ALGORITHM_H
