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

#include "classification/GaussianNaiveBayes.h"
#include "stats/Distributions.h"
#include "utils/operations.h"

namespace juml {
    void GaussianNaiveBayes::fit(const arma::Mat<float>& X, const arma::Col<int>& y) {
        BaseClassifier::fit(X, y);
        const int n_classes = this->class_normalizer_.n_classes();
        this->class_counts_.zeros(n_classes);
        this->prior_.zeros(n_classes);
        this->theta_.zeros(n_classes, X.n_cols);
        this->stddev_.zeros(n_classes, X.n_cols);

        #pragma omp parallel for
        for (size_t row = 0; row < X.n_rows ; ++row) {
            const size_t class_index = this->class_normalizer_.transform(y(row));
            ++class_counts_(class_index);
            ++this->prior_(class_index);
            this->theta_.row(class_index) += X.row(row);
        }
        this->prior_ /= y.n_elem;
        this->theta_.each_col() /= class_counts_;

        #pragma omp parallel for
        for (size_t row = 0; row < X.n_rows; ++row) {
            const size_t class_index = this->class_normalizer_.transform(y(row));
            arma::frowvec deviation = X.row(row) - this->theta_.row(class_index);
            this->stddev_.row(class_index) += arma::pow(deviation, 2);
        }

        this->stddev_.each_col() /= class_counts_;
        this->stddev_ = arma::sqrt(this->stddev_);
    }

    arma::Mat<float> GaussianNaiveBayes::predict_probability(const arma::Mat<float>& X) const {
        arma::Mat<float> probabilities = arma::ones<arma::Mat<float>>(X.n_rows, this->class_normalizer_.n_classes());

        for (size_t i = 0; i < this->prior_.n_elem; ++i) {
            const float prior = this->prior_(i);
            probabilities.col(i) *= prior;

            #pragma omp parallel for
            for (size_t row = 0; row < X.n_rows; ++row) {
                const arma::frowvec& mean = this->theta_.row(i);
                const arma::frowvec& stddev = this->stddev_.row(i);
                arma::frowvec features_probs = gaussian_pdf<float>(X.row(row), mean, stddev);
                probabilities(row, i) *= arma::prod(features_probs);
            }
        }

        return probabilities;
    }

    arma::Col<int> GaussianNaiveBayes::predict(const arma::Mat<float>& X) const {
        arma::Mat<float> probabilities = this->predict_probability(X);
        arma::Col<int> predictions(X.n_rows);
        arma::Col<unsigned int> max_index = argmax(probabilities, 1);

        for (size_t i = 0; i < max_index.n_elem; ++i) {
            predictions(i) = this->class_normalizer_.invert(max_index(i));
        }

        return predictions;
    }

    float GaussianNaiveBayes::accuracy(const arma::Mat<float>& X, const arma::Col<int>& y) const {
        arma::Col<int> predictions = this->predict(X);
        return (float)arma::sum(predictions == y) / (float)y.n_elem;
    }
} // namespace juml
