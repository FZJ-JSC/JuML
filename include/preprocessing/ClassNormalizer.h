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
#include <map>
#include <mpi.h>
#include <stdexcept>
#include <sstream>

#include "data/Dataset.h"

namespace juml {
    //! ClassNormalizer
    //! TODO: Describe me
    class ClassNormalizer {
    protected:
        af::array class_labels_;
        std::map<intl, intl> class_mapping_;
        MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;
    public:
        ClassNormalizer(MPI_Comm comm=MPI_COMM_WORLD);

        void index(const Dataset& y);

        inline dim_t n_classes() const {
            return this->class_labels_.elements();
        }

        template <typename T>
        intl transform(T class_label) const {
            auto found = this->class_mapping_.find(class_label);
            if (found == this->class_mapping_.end()) {
                std::stringstream message;
                message << "Class " << class_label << " not found";
                throw std::invalid_argument(message.str().c_str());
            }
            return found->second;
        }

        inline intl invert(int transformed_label) const {
            if (transformed_label < 0 || transformed_label > this->n_classes()) {
                std::stringstream message;
                message << "Class " << transformed_label << " not found";
                throw std::invalid_argument(message.str().c_str());
            }
            return this->class_labels_(transformed_label).scalar<intl>();
        }

        inline const af::array& classes() const {
            return this->class_labels_;
        }
    }; // ClassNormalizer
}  // juml
#endif // CLASS_NORMALIZER_H

