/*
 * Copyright (c) 2015
 * Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
 *
 * This software may be modified and distributed under the terms of BSD-style license.
 *
 * File name: ClassNormalizer.h
 *
 * Description: Header of class ClassNormalizer
 *
 * Maintainer: p.glock
 *
 * Email: phil.glock@gmail.com
 */

#ifndef CLASS_NORMALIZER_H
#define CLASS_NORMALIZER_H

#include <arrayfire.h>
#include <cstdint>
#include <mpi.h>
#include <stdexcept>
#include <sstream>

#include "data/Dataset.h"

namespace juml {
    //! ClassNormalizer
    //! TODO: Describe me
    class ClassNormalizer {
    protected:
        af::array* class_labels_;
        MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;
        
    public:
        ClassNormalizer(MPI_Comm comm=MPI_COMM_WORLD);

        const af::array& classes() const;
        void index(const Dataset& y);
        
        template <typename T>
        T invert(const intl transformed_label) const {
            if (transformed_label < 0 || transformed_label > this->n_classes()) {
                std::stringstream message;
                message << "Class " << transformed_label << " not found";
                throw std::invalid_argument(message.str().c_str());
            }
            af::array::array_proxy index = (*this->class_labels_)(transformed_label);
            
            return static_cast<T>(index.scalar<intl>());
        }
        af::array invert(const af::array& transformed_labels) const;
        
        dim_t n_classes() const;

        template <typename T>
        intl transform(const T class_label) const {
            af::array indices = af::where((*this->class_labels_) == class_label);
            if (indices.elements() != 1) {
                std::stringstream message;
                message << "Class " << class_label << " not found";
                throw std::invalid_argument(message.str().c_str());
            }
            
            return static_cast<intl>(indices.scalar<unsigned int>());
        }
        af::array transform(const af::array& class_labels) const;
        
        ~ClassNormalizer();
    }; // ClassNormalizer
}  // juml
#endif // CLASS_NORMALIZER_H

