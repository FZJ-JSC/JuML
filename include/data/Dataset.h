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

#ifndef JUML_DATA_DATASET_H_
#define JUML_DATA_DATASET_H_

#include <arrayfire.h>
#include <hdf5.h>
#include <mpi.h>
#include <sys/stat.h>
#include <string>

namespace juml {
    //! Dataset
    //! TODO: Describe me
    class Dataset {
     protected:
        af::array data_;

        const std::string filename_;
        const std::string dataset_;
        time_t loading_time_ = 0;

        const MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;

        dim_t sample_dim_;
        dim_t global_n_samples_;
        dim_t global_offset_;

        af::dtype h5_to_af(hid_t h5_type);
        hid_t af_to_h5(af::dtype af_type);

     public:
        //! Dataset constructor
        Dataset(const std::string& filename, const std::string& dataset, const MPI_Comm comm = MPI_COMM_WORLD);
        explicit Dataset(const af::array& data, MPI_Comm comm = MPI_COMM_WORLD);

        time_t modified_time() const;
        void normalize(float min = 0, float max = 1, bool independent_features = false,
                       const af::array& selected_features = af::array());
        void normalize_stddev(float x_std = 1, bool independent_features = false,
                              const af::array &selected_features = af::array());
        af::array mean(bool total = false) const;
        time_t loading_time() const;
        void load_equal_chunks(bool force = false);
        void dump_equal_chunks(const std::string& filename, const std::string& dataset);
        af::array stddev(bool total = false) const;

        virtual af::array& data();
        virtual const af::array& data() const;
        virtual dim_t n_samples() const;
        virtual dim_t n_features() const;
        virtual dim_t global_n_samples() const;
        virtual dim_t global_offset() const;
        virtual dim_t sample_dim() const;
    }; // Dataset
}  // namespace juml
#endif // JUML_DATA_DATASET_H_
