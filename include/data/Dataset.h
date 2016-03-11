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
#include <mpi.h>
#include <string>

namespace juml {
    //! Dataset
    //! TODO: Describe me
    class Dataset {
    protected:
        af::array data_;

        const std::string filename_;
        const std::string dataset_;

        const MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;

        dim_t global_items_;
        dim_t global_offset_;
        
        af::dtype h5_to_af(hid_t h5_type);

    public:
        //! Dataset constructor
        Dataset(const std::string& filename, const std::string& dataset, const MPI_Comm comm=MPI_COMM_WORLD);        
        Dataset(af::array& data, MPI_Comm comm=MPI_COMM_WORLD);
       
        void load_equal_chunks();

        virtual af::array& data();
        virtual const af::array& data() const;
        virtual dim_t n_samples() const;
        virtual dim_t n_features() const;
        virtual dim_t global_items() const;
        virtual dim_t global_offset() const;
    }; // Dataset
}  // juml
#endif // DATASET_H

