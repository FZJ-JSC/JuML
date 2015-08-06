/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: DataSet.h
*
* Description: Header of class DataSet
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
    //! DataSet
    //! TODO: Describe me
    class DataSet {
    friend BaseClassifier;

    protected:
        arma::fmat data_;
        std::string filename_;
        std::string data_set_;
        MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;

        virtual void load_equal_chunks();


    public:
        //! DataSet constructor
        DataSet(const std::string& filename, const std::string& data_set, MPI_Comm comm=MPI_COMM_WORLD);
        ~DataSet();

        virtual inline arma::fmat& data() { return this->data_; }
        virtual inline size_t n_features() const { return this->data_.n_cols; }
        virtual inline size_t n_samples() const { return this->data_.n_rows; }
    }; // DataSet
}  // juml
#endif // DATASET_H
