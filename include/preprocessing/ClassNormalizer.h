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
    /**
     * ClassNormalizer
     *
     * This utility class allows the normalization of arbitrary input class labels (e.g. -10, -3.4, 5, 100, ...) into
     * the range [0, #classes) across all nodes. It is used in various classifiers to simply their implementation.
     */
    class ClassNormalizer {
    protected:
        /**
         * @var   class_labels_
         * @brief A 1xn row vector, where each original label is placed at the normalized label it will be converted to
         */
        af::array class_labels_;
        /**
         * @var   comm_
         * @brief The MPI communicator pointing to all involved hosts
         */
        MPI_Comm comm_;
        /**
         * @var   mpi_rank_
         * @brief The MPI rank of this node in comm_
         */
        int mpi_rank_;
        /**
         * @var   mpi_size_
         * @brief The number of nodes in comm_
         */
        int mpi_size_;
        
    public:
        /**
         * ClassNormalizer constructor
         *
         * @params comm The MPI communicator for the execution
         */
        ClassNormalizer(MPI_Comm comm=MPI_COMM_WORLD);

        const af::array& classes() const;

        /**
         * index
         * Indexes and normalizes the class labels contained in the passed dataset to the range [0, #classes)
         * @param y The class labels to normalize
         */
        void index(const Dataset& y);

        /**
         * invert
         * Denormalizes or transforms a single, integral label back to its original value in the dataset
         * @param transformed_label The label to denormalize
         * @returns The original label in the dataset
         * @throws invalid_argument If input cannot be inverted
         */
        template <typename T>
        T invert(const intl transformed_label) const {
            if (transformed_label < 0 || transformed_label > this->n_classes()) {
                std::stringstream message;
                message << "Class " << transformed_label << " not found";
                throw std::invalid_argument(message.str().c_str());
            }
            af::array::array_proxy index = this->class_labels_(transformed_label);
            
            return static_cast<T>(index.scalar<intl>());
        }
        /**
         * invert
         * Vectorized version of denormalization
         * @param transformed_labels A 1xn row-vector to be denormalized
         * @returns A 1xn row-vector containing te denormalized labels
         * @throws invalid_argument If input cannot be inverted or input is not a row-vector
         */
        af::array invert(const af::array& transformed_labels) const;

        /**
         * n_classes
         * @returns The number of indexed classes or zero if not yet indexed
         */
        dim_t n_classes() const;

        /**
         * transform
         * Normalizes a single, integral class-label
         * @param class_label The label to be transformed
         * @returns The transformed label
         * @throws invalid_arguments If label cannot be transformed
         */
        template <typename T>
        intl transform(const T class_label) const {
            af::array indices = af::where(this->class_labels_ == class_label);
            if (indices.elements() != 1) {
                std::stringstream message;
                message << "Class " << class_label << " not found";
                throw std::invalid_argument(message.str().c_str());
            }
            
            return static_cast<intl>(indices.scalar<unsigned int>());
        }

        /**
         * transform
         * Vectorized label transformation
         * @param A 1xn row-vector containing the labels to be normalized
         * @returns A 1xn row-vector containing the normalized labels
         * @throws invalid_argument If labels cannot be transformed or input is not a row-vector
         */
        af::array transform(const af::array& original_labels) const;
    }; // ClassNormalizer
}  // juml
#endif // CLASS_NORMALIZER_H
