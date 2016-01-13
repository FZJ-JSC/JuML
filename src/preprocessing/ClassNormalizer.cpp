/*
 * Copyright (c) 2015
 * Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
 *
 * This software may be modified and distributed under the terms of BSD-style license.
 *
 * File name: ClassNormalizer.cpp
 *
 * Description: Implementation of class ClassNormalizer
 *
 * Maintainer: p.glock
 *
 * Email: phil.glock@gmail.com
 */

#include "preprocessing/ClassNormalizer.h"

namespace juml {
    ClassNormalizer::ClassNormalizer(MPI_Comm comm) : comm_(comm) {
        MPI_Comm_rank(comm, &this->mpi_rank_);
        MPI_Comm_size(comm, &this->mpi_size_);
    }

    void ClassNormalizer::index(const Dataset& y) {
        af::Backend currentBackend = af::getBackendId(af::constant(0, 1));
        
        this->class_mapping_.clear();        
        af::array local_class_labels = af::setUnique(y.data().as(s64));   
        // send the local number of classes to all processes
        int n_classes = local_class_labels.elements();
        int* n_classes_per_processor = new int[this->mpi_size_];
        MPI_Allgather(&n_classes, 1, MPI_INT, n_classes_per_processor, 1, MPI_INT, this->comm_);

        // calculate displacements
        int* displacements = new int[this->mpi_size_];
        displacements[0] = 0;
        int total_n_classes = 0;
        
        for (int i = 1; i < this->mpi_size_; ++i) {
            total_n_classes += n_classes_per_processor[i - 1];
            displacements[i] = total_n_classes;
        }
        total_n_classes += n_classes_per_processor[this->mpi_size_ - 1];

        // exchange class labels
        intl* total_classes = new intl[total_n_classes];
        if(currentBackend == AF_BACKEND_CPU) {
            MPI_Allgatherv(local_class_labels.device<intl>(), n_classes, MPI_LONG_LONG, total_classes, n_classes_per_processor, displacements, MPI_LONG_LONG, this->comm_);
            local_class_labels.unlock();
        }
        else {
            dim_t* buffer = local_class_labels.host<intl>();         
            MPI_Allgatherv(buffer, n_classes, MPI_LONG_LONG, total_classes, n_classes_per_processor, displacements, MPI_LONG_LONG, this->comm_);            
        }

        // compute global unique classes
        af::array global_classes(total_n_classes, total_classes);
        
        this->class_labels_ = af::setUnique(global_classes);

        // release mpi buffers
        delete[] total_classes;
        delete[] displacements;
        delete[] n_classes_per_processor;

        // map original classes to normalized ones
        for (int label = 0; label < this->class_labels_.elements(); ++label) {
            auto original_class = this->class_labels_(label).scalar<intl>();
            this->class_mapping_[original_class] = label;
        }
    }
} // namespace juml

