/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: GaussianNaiveBayes.cpp
*
* Description: Implementation of class GaussianNaiveBayes
*
* Maintainer: p.glock
*
* Email: phil.glock@gmail.com
*/

#include <algorithm>

#include "classification/GaussianNaiveBayes.h"
#include "stats/Distributions.h"

namespace juml {
    void GaussianNaiveBayes::fit(Dataset& X, Dataset& y) {
        X.load_equal_chunks();
        y.load_equal_chunks();
        BaseClassifier::fit(X, y);

        const af::array& X_ = X.data();
        const af::array& y_ = y.data();

        const int n_classes = this->class_normalizer_.n_classes();
        this->class_counts_ = af::constant(0.0f, n_classes);
        this->prior_ = af::constant(0.0f, n_classes);
        this->theta_ = af::constant(0.0f, n_classes, X_.dims(1));
        this->stddev_ = af::constant(0.0f, n_classes, X_.dims(1));

        #pragma omp parallel
        {
            af::array class_counts = af::constant(0.0f, n_classes);
            af::array priors = af::constant(0.0f, n_classes);
            af::array theta = af::constant(0.0f, n_classes, X_.dims(1));

            #pragma omp for nowait
            for (size_t row = 0; row < X_.dims(0) ; ++row) {
                const size_t class_index = this->class_normalizer_.transform(y_(row));

                class_counts(class_index) += 1;
                priors(class_index) += 1;
                theta.row(class_index) += X_.row(row);
            }

            #pragma omp critical
            {
                this->class_counts_ += class_counts;
                this->prior_ += priors;
                this->theta_ += theta;
            }
        }

        // copy variables into one array and use only one mpi call
        const size_t n_floats = n_classes * (2 + X_.dims(1)) + 1;
        float* message = new float[n_floats];

        float* _p_class_counts = this->class_counts_.host<float>();
        float* _p_prior = this->prior_.host<float>();
        float* _p_theta = this->theta_.host<float>();
        // create message
        std::copy(_p_class_counts, _p_class_counts + n_classes, message);
        std::copy(_p_prior, _p_prior + n_classes, message + n_classes);
        std::copy(_p_theta, _p_theta + this->theta_.elements(), message + n_classes * 2);
        message[n_floats - 1] = (float)y_.elements();

        // reduce to obtain global view
        MPI_Allreduce(MPI_IN_PLACE, message, n_floats, MPI_FLOAT, MPI_SUM, this->comm_);

        // copy the reduced elements back
        std::copy(message, message + n_classes, this->class_counts_.get());
        std::copy(message + n_classes, message + 2 * n_classes, this->prior_.get());
        std::copy(message + 2 * n_classes, message + 2 * n_classes + this->theta_.elements(), this->theta_.get());

        // extract reduced class counts, prior and theta from message

        const size_t total_n_y = (size_t)message[n_floats - 1];
        this->prior_ /= total_n_y;
        this->theta_.cols(0, af::end) /= this->class_counts_;

        // calculate standard deviation for each feature
        #pragma omp parallel
        {
            af::array stddev = af::constant(0.0f, n_classes, X_.dims(1));
            #pragma omp for nowait
            for (size_t row = 0; row < X_.dims(0); ++row) {
                const size_t class_index = this->class_normalizer_.transform(y_(row));
                af::array deviation = X_.row(row) - this->theta_.row(class_index);
                stddev.row(class_index) += af::pow(deviation, 2);
            }

            #pragma omp critical
            {
                this->stddev_ += stddev;
            }
        }

        // exchange the standard deviations values
        const size_t n_stddev = n_classes * X_.dims(1);
        float* _p_stddev = this->stddev_.host<float>();
        std::copy(_p_stddev, _p_stddev + n_stddev, message);
        MPI_Allreduce(MPI_IN_PLACE, message, n_stddev, MPI_FLOAT, MPI_SUM, this->comm_);
        std::copy(message, message + n_stddev, _p_stddev);

        // normalize standard deviation by class counts and calculate root (not variance)
        this->stddev_.cols(0, af::end) /= this->class_counts_;
        this->stddev_ = af::sqrt(this->stddev_);

        // release message buffer
        delete[] message;
    }

    Dataset GaussianNaiveBayes::predict_probability(const Dataset& X) const {
        const af::array& X_ = X.data();
        af::array probabilities = af::constant(1.0f, X_.dims(0), this->class_normalizer_.n_classes());

        for (size_t i = 0; i < this->prior_.elements(); ++i) {
            const float prior = this->prior_(i).scalar<float>();
            probabilities.col(i) *= prior;

            #pragma omp parallel for
            for (size_t row = 0; row < X_.dims(0); ++row) {
                const af::array& mean = this->theta_.row(i);
                const af::array& stddev = this->stddev_.row(i);
                af::array features_probs = gaussian_pdf(X_.row(row), mean, stddev);
                probabilities(row, i) *= af::product(features_probs);
            }
        }
        return Dataset(probabilities, this->comm_);
    }

    Dataset GaussianNaiveBayes::predict(const Dataset& X) const {
        const af::array& X_ = X.data();

        Dataset probabilities = this->predict_probability(X);
        af::array predictions(X_.dims(0));
        af::array max_index = af::constant(0, X_.dims(0));
        af::array max_val = af::constant(0.0f, X_.dims(0));
        af::max(max_val, max_index, probabilities.data(), 1);

        for (size_t i = 0; i < max_index.elements(); ++i) {
            predictions(i) = this->class_normalizer_.invert(max_index(i).scalar<int>());
        }
        Dataset preds(predictions, this->comm_);
        return preds;
    }

    float GaussianNaiveBayes::accuracy(const Dataset& X, const Dataset& y) const {
        const af::array& y_ = y.data();
        Dataset predictions = this->predict(X);

        float local_sum = af::sum(predictions.data() == y_);
        float message[2] = {local_sum, (float)y_.elements()};
        MPI_Allreduce(MPI_IN_PLACE, message, 2, MPI_FLOAT, MPI_SUM, this->comm_);
        const float total_sum = message[0];
        const float total_n_samples = message[1];

        return total_sum / total_n_samples;
    }
} // namespace juml
