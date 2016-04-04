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

#include <stdexcept>

#include "core/Backend.h"
#include "core/MPI.h"
#include "preprocessing/ClassNormalizer.h"

namespace juml {
    ClassNormalizer::ClassNormalizer(MPI_Comm comm)
      : comm_(comm) {
        MPI_Comm_rank(comm, &this->mpi_rank_);
        MPI_Comm_size(comm, &this->mpi_size_);
    }

    const af::array& ClassNormalizer::classes() const {
        return this->class_labels_;
    }

    void ClassNormalizer::index(const Dataset& y) {
        const af::array& data = y.data();
        dim_t n = data.dims(1);

        // check dimensionality
        if (data.dims(0) > 1 || data.dims(2) > 1 || data.dims(3) > 1) {
            throw std::invalid_argument("Class labels must be a row-vector");
        }

        // setup backend, local classes and collect globally
        Backend::set(af::getBackendId(data));
        af::array class_labels = af::setUnique(af::moddims(data, n));
        mpi::allgatherv(class_labels, this->comm_, 0);

        // compute global unique classes
        class_labels = af::setUnique(class_labels);
        this->class_labels_ = af::moddims(class_labels, 1, class_labels.dims(0)).as(s64);
    }

    af::array ClassNormalizer::invert(const af::array& transformed_labels) const  {
        if (transformed_labels.dims(0) > 1 || transformed_labels.dims(2) > 1 || transformed_labels.dims(3) > 1) {
            throw std::invalid_argument("Transformed labels needs to be a row-vector");
        }

        dim_t labels = transformed_labels.elements();
        if (af::anyTrue<bool>(transformed_labels < 0 || transformed_labels >= labels)) {
            throw std::invalid_argument("Could not invert transformed labels - out of bounds");
        }
        return this->class_labels_(transformed_labels);
    }

    dim_t ClassNormalizer::n_classes() const {
        return this->class_labels_.elements();
    }

    af::array ClassNormalizer::transform(const af::array& original_labels) const {
        if (original_labels.dims(0) > 1 || original_labels.dims(2) > 1 || original_labels.dims(3) > 1) {
            throw std::invalid_argument("Original labels needs to be a row-vector");
        }

        dim_t classes = this->n_classes();
        dim_t labels = original_labels.elements();

        // create volumes and compare them against each other
        af::array class_volume = af::tile(af::moddims(this->class_labels_, classes), 1, static_cast<unsigned int>(labels));
        af::array label_volume = af::tile(original_labels, classes);

        // find the indexes where they are equal and clamp them using the modulo operation
        af::array transformed = af::where(class_volume == label_volume);
        if (transformed.elements() != labels) {
            throw std::invalid_argument("Could not convert all the labels");
        }
        return af::moddims(transformed % classes, 1, labels);
    }
} // namespace juml
