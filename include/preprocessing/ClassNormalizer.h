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

#include <armadillo>
#include <map>
#include <mpi.h>

#include "data/Dataset.h"

namespace juml {
    //! ClassNormalizer
    //! TODO: Describe me
    class ClassNormalizer {
    protected:
        arma::Col<int> class_labels_;
        std::map<int, int> class_mapping_;
        MPI_Comm comm_;
        int mpi_rank_;
        int mpi_size_;
    public:
        ClassNormalizer(MPI_Comm comm=MPI_COMM_WORLD);

        void index(const Dataset<int>& y);

        inline int n_classes() const {
            return this->class_labels_.n_elem;
        }

        int transform(int class_label) const;

        inline int invert(int transformed_label) const {
            return this->class_labels_(transformed_label);
        }

        inline const arma::Col<int>& classes() const {
            return this->class_labels_;
        }
    }; // ClassNormalizer
}  // juml
#endif // CLASS_NORMALIZER_H
