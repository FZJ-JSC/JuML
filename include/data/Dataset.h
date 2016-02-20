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

#ifndef DATASET_H
#define DATASET_H

#include <arrayfire.h>
#include <hdf5.h>
#include <limits>
#include <mpi.h>
#include <string>
#include <sys/stat.h>

namespace juml {
    //! Dataset
    //! TODO: Describe me
    class Dataset {
    protected:
        af::array data_;
        const std::string filename_;
        const std::string dataset_;
        time_t loading_time_ = std::numeric_limits<time_t>::min();
        const MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;
        
        af::dtype h5_to_af(hid_t h5_type);

    public:
        //! Dataset constructor
        Dataset(const std::string& filename, const std::string& dataset, const MPI_Comm comm=MPI_COMM_WORLD);        
        Dataset(af::array& data, MPI_Comm comm=MPI_COMM_WORLD);

        time_t modified_time();
        int load_equal_chunks();

        virtual af::array& data();
        virtual const af::array& data() const;
        virtual dim_t n_samples() const;
        virtual dim_t n_features() const;
    }; // Dataset
}  // juml
#endif // DATASET_H

