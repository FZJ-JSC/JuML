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

#include <armadillo>

#include <string>

#include "classification/BaseClassifier.h"

namespace juml {
    //! Dataset
    //! TODO: Describe me
    class Dataset {
    friend BaseClassifier;

    protected:
        arma::Mat<float> data_;
        std::string filename_;
        std::string data_set_;
        MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;

        virtual void load_equal_chunks();


    public:
        //! Dataset constructor
        Dataset(const std::string& filename, const std::string& data_set, MPI_Comm comm=MPI_COMM_WORLD);
        ~Dataset();

        virtual inline arma::Mat<float>& data() { return this->data_; }
        virtual inline size_t n_features() const { return this->data_.n_cols; }
        virtual inline size_t n_samples() const { return this->data_.n_rows; }
    }; // Dataset
}  // juml
#endif // DATASET_H
