/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: GaussianNaiveBayes.h
*
* Description: Header of class GaussianNaiveBayes
*
* Maintainer: p.glock
* 
* Email: phil.glock@gmail.com
*/

#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

#include <arrayfire.h>
#include <mpi.h>

#include "core/Backend.h"
#include "classification/BaseClassifier.h"
#include "data/Dataset.h"

namespace juml {
    /**
     * Gaussian Naive Bayes (or GNB)
     *
     * Simple probabilistic classifier based on Bayes' theorem that assumes independence of the feature variables. In
     * this model the likelihood of the features is assumed to be normal/gaussian distributed.
     *
     * Example:
     *
     * @code
     * #include <JuML.h>
     * using namespace juml;
     *
     * Dataset X("train.h5", "data");
     * Dataset y("train.h5", "labels");
     * Dataset C("test.h5", "data"):
     *
     * GaussianNaiveBayes gnb();
     * gnb.fit(X, y);
     * Dataset predictions = gnb.predict(c);
     * @endcode
     */
    class GaussianNaiveBayes : public BaseClassifier {
    protected:
        /**
         * @var   class_counts_
         * @brief A 1 x n arrayfire row-vector. Each column contains the global occurrence of the normalized class.
         */
        af::array class_counts_;
        /**
         * @var   prior_
         * @brief A 1 x n arrayfire row-vector. Each column represents the relative global occurrence of the respective
         *        normalized class.
         */
        af::array prior_;
        /**
         * @var   stddev_
         * @brief A f x n arrayfire matrix. Each column represents the standard deviation of each feature of this class.
         */
        af::array stddev_;
        /**
         * @var   theta_
         * @brief A f x n arrayfire matrix. Each column represents the mean of the feature of each feature of this class.
         */
        af::array theta_;

    public:
        /**
         * GaussianNaiveBayes constructor
         * @param backend The execution backend
         * @param comm The MPI communicator for the execution
         */
        GaussianNaiveBayes(int backend=Backend::CPU, MPI_Comm comm=MPI_COMM_WORLD);

        /**
         * Fits GNB classifier based on the passed training data and labels
         * @param X The training dataset
         * @param y The label dataset to train on
         */
        virtual void fit(Dataset& X, Dataset& y) override;

        /**
         * Classifies the passed test data based on the previously fit model.
         * @param X The test dataset
         * @returns The predicted labels as dataset
         */
        virtual Dataset predict(Dataset& X) const override;

        /**
         * Predicts the probability estimates of the passed test data based on the previously fit model.
         * @param X The test dataset
         * @return The predicted probabilities for each class (f x n) as dataset
         */
        virtual Dataset predict_probability(Dataset& X) const;
        virtual float accuracy(Dataset& X, Dataset& y) const override;

        const af::array& class_counts() const;
        const af::array& prior() const;
        const af::array& stddev() const;
        const af::array& theta() const;
    };
} // namespace juml

#endif // GAUSSIANNAIVEBAYES_H
