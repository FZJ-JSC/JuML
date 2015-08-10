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
#include "utils/operations.h"

namespace juml {
    void GaussianNaiveBayes::fit(const Dataset<float>& X, const Dataset<int>& y) {
        BaseClassifier::fit(X, y);
        
        const arma::Mat<float>& X_ = X.data();
        const arma::Mat<int>& y_ = y.data();
        
        const int n_classes = this->class_normalizer_.n_classes();
        this->class_counts_.zeros(n_classes);
        this->prior_.zeros(n_classes);
        this->theta_.zeros(n_classes, X_.n_cols);
        this->stddev_.zeros(n_classes, X_.n_cols);

        #pragma omp parallel
        {
            arma::Col<float> class_counts(n_classes, arma::fill::zeros);
            arma::Col<float> priors(n_classes, arma::fill::zeros);
            arma::Mat<float> theta(n_classes, X_.n_cols, arma::fill::zeros);
            
            #pragma omp for nowait
            for (size_t row = 0; row < X_.n_rows ; ++row) {
                const size_t class_index = this->class_normalizer_.transform(y_(row));
                ++class_counts(class_index);
                ++priors(class_index);
                theta.row(class_index) += X.data().row(row);
            }
            
            #pragma omp critical
            {
                this->class_counts_ += class_counts;
                this->prior_ += priors;
                this->theta_ += theta;
            }
        }
        
        // copy variables into one array and use only one mpi call
        const size_t n_floats = n_classes * ( 2 + X_.n_cols) + 1;
        float* message = new float[n_floats];
        
        std::copy(this->class_counts_.memptr(), this->class_counts_.memptr() + this->class_counts_.n_elem, message);
        std::copy(this->prior_.memptr(), this->prior_.memptr() + this->prior_.n_elem, message + n_classes);
        std::copy(this->theta_.memptr(), this->theta_.memptr() + this->theta_.n_elem, message + n_classes * 2);
        message[n_floats - 1] = y_.n_elem;
        
        MPI_Allreduce(MPI_IN_PLACE, message, n_floats, MPI_FLOAT, MPI_SUM, this->comm_);
        
        // extract reduced class counts, prior and theta from message
        for (size_t i = 0; i < n_classes; ++i) {
            this->class_counts_(i) = message[i];
            this->prior_(i) = message[i + n_classes];
            
            for (size_t j = 0; j < X_.n_cols; ++j) {
                this->theta_(i, j) = message[i * X_.n_cols + j + 2 * n_classes];
            }
        }
        
        const size_t total_n_y = message[n_floats - 1];
        this->prior_ /= total_n_y;
        this->theta_.each_col() /= this->class_counts_;
        
        // calculate standard deviation for each feature
        #pragma omp parallel
        {
            arma::Mat<float> stddev(n_classes, X_.n_cols, arma::fill::zeros);
            #pragma omp for nowait
            for (size_t row = 0; row < X_.n_rows; ++row) {
                const size_t class_index = this->class_normalizer_.transform(y_(row));
                arma::Row<float> deviation = X_.row(row) - this->theta_.row(class_index);
                stddev.row(class_index) += arma::pow(deviation, 2);
            }
            
            #pragma omp critical
            {
                this->stddev_ += stddev;
            }
        }
        
        const size_t n_stddev = n_classes * X_.n_cols;
        std::copy(this->stddev_.memptr(), this->stddev_.memptr() + n_stddev, message);
        MPI_Allreduce(MPI_IN_PLACE, message, n_stddev, MPI_FLOAT, MPI_SUM, this->comm_);
        
        for (size_t row = 0; row < n_classes; ++row) {
            for (size_t col = 0; col < X_.n_elem; ++col) {
                this->stddev_(row, col) = message[row * n_classes + col];
            }
        }
        
        this->stddev_.each_col() /= this->class_counts_;
        this->stddev_ = arma::sqrt(this->stddev_);        
        
        // release message buffer
        delete[] message;
    }

    Dataset<float> GaussianNaiveBayes::predict_probability(const Dataset<float>& X) const {
        const arma::Mat<float>& X_ = X.data();
        arma::Mat<float> probabilities = arma::ones<arma::Mat<float>>(X_.n_rows, this->class_normalizer_.n_classes());

        for (size_t i = 0; i < this->prior_.n_elem; ++i) {
            const float prior = this->prior_(i);
            probabilities.col(i) *= prior;

            #pragma omp parallel for
            for (size_t row = 0; row < X_.n_rows; ++row) {
                const arma::Row<float>& mean = this->theta_.row(i);
                const arma::Row<float>& stddev = this->stddev_.row(i);
                arma::Row<float> features_probs = gaussian_pdf<float>(X_.row(row), mean, stddev);
                probabilities(row, i) *= arma::prod(features_probs);
            }
        }
        return Dataset<float>(probabilities, this->comm_);
    }

    Dataset<int> GaussianNaiveBayes::predict(const Dataset<float>& X) const {
        const arma::Mat<float>& X_ = X.data();
        
        Dataset<float> probabilities = this->predict_probability(X);
        arma::Col<int> predictions(X_.n_rows);
        arma::Col<unsigned int> max_index = argmax(probabilities.data(), 1);

        for (size_t i = 0; i < max_index.n_elem; ++i) {
            predictions(i) = this->class_normalizer_.invert(max_index(i));
        }
        Dataset<int> preds(predictions, this->comm_);
        return preds;
    }

    float GaussianNaiveBayes::accuracy(const Dataset<float>& X, const Dataset<int>& y) const {
        const arma::Mat<int>& y_ = y.data();
        Dataset<int> predictions = this->predict(X);
        
        float local_sum = arma::accu(predictions.data() == y_);
        
        return 0.0f;
    }
} // namespace juml

