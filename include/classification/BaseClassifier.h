/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BaseClassifier.h
*
* Description: Header of class BaseClassifier
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef JUML_CLASSIFICATION_BASECLASSIFIER_H_
#define JUML_CLASSIFICATION_BASECLASSIFIER_H_

#include <mpi.h>

#include "core/Algorithm.h"
#include "data/Dataset.h"
#include "preprocessing/ClassNormalizer.h"

namespace juml {
    class BaseClassifier : public Algorithm {
     protected:
        /**
         * @var   class_normalizer_
         * @brief A ClassNormalizer instance, normalizes class labels on calling fit
         */
        ClassNormalizer class_normalizer_;

     public:
        /**
         * BaseClassifier constructor
         *
         * Stores the processing backend and the MPI communicator for the classification process
         *
         * @param backend - The backend, defaults to CPU
         * @param comm - The MPI communicator
         */
        explicit BaseClassifier(int backend = Backend::CPU, MPI_Comm comm = MPI_COMM_WORLD);

        /**
         * (Abstract) fit
         *
         * Trains the classifier
         *
         * @param X - The training data
         * @param y - The training labels
         */
        virtual void fit(Dataset& X, Dataset& y);

        /**
         * (Abstract) predict
         *
         * Classifies a dataset according to the previously trained model.
         *
         * @param X - The test data to classify
         * @returns The predicted labels
         */
        virtual Dataset predict(Dataset& X) const = 0;

        /**
         * (Abstract) accuracy
         *
         * Fits the classification models and returns the prediction accuracy.
         *
         * @param X - The training data
         * @param y - The training labels
         * @returns The training prediction accuracy between 0 and 1.
         */
        virtual float accuracy(Dataset& X, Dataset& y) const = 0;
    };
} // namespace juml

#endif // JUML_CLASSIFICATION_BASECLASSIFIER_H_

