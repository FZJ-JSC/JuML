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
#include <iostream>

#include "classification/GaussianNaiveBayes.h"
#include "stats/Distributions.h"

namespace juml {
    GaussianNaiveBayes::GaussianNaiveBayes(int backend, MPI_Comm comm)
      : BaseClassifier(backend, comm)
    {}
    
    void GaussianNaiveBayes::fit(Dataset& X, Dataset& y) {
        af::setBackend(static_cast<af::Backend>(this->backend_.get()));
        
        X.load_equal_chunks();
        y.load_equal_chunks();
        BaseClassifier::fit(X, y);        
        
        const af::array& X_ = X.data();
        const af::array& y_ = y.data();
        
        const dim_t n_classes = this->class_normalizer_.n_classes();
        this->class_counts_ = af::constant(0.0f, 1, n_classes);
        this->prior_ = af::constant(0.0f, 1, n_classes);
        this->theta_ = af::constant(0.0f, X.n_features(), n_classes);
        this->stddev_ = af::constant(0.0f, X.n_features(), n_classes);
        
        af::array transformed_labels = this->class_normalizer_.transform(y_);
        for (int label = 0; label < n_classes; ++label) {
            af::array class_index = (transformed_labels == label);
            this->class_counts_(label) = af::sum<int>(class_index);
            if (!af::anyTrue<bool>(this->class_counts_(label)))
                continue;
            af::array accumulated = af::sum(X_(af::span, class_index), 1);
            this->theta_.col(label) = accumulated;
        }
        this->prior_ = this->class_counts_;
        
        // copy variables into one array and use only one mpi call
        const dim_t n_floats = n_classes * (2 + X.n_features()) + 1;
        float* message = new float[n_floats];
        
        // create message
        this->class_counts_.host(reinterpret_cast<void*>(message));
        this->prior_.host(reinterpret_cast<void*>(message + n_classes));
        this->theta_.host(reinterpret_cast<void*>(message + n_classes * 2));
        message[n_floats - 1] = (float)y.n_samples();
        
        // reduce to obtain global view
        MPI_Allreduce(MPI_IN_PLACE, message, n_floats, MPI_FLOAT, MPI_SUM, this->comm_);
        
        // copy the reduced elements back        
        this->class_counts_.write(message, n_classes * sizeof(float));
        this->prior_.write(message + n_classes, n_classes * sizeof(float));
        this->theta_.write(message + n_classes * 2, this->theta_.elements() * sizeof(float));
        
        // extract reduced class counts, prior and theta from message
        const intl total_n_y = (intl) message[n_floats - 1];
        this->prior_ /= total_n_y;        
        gfor (af::seq row, X.n_features()) {
            this->theta_(row, af::span) /= this->class_counts_;
        }
        
        // calculate standard deviation for each feature
        for (int label = 0; label < n_classes; ++label) {
            af::array class_index = (transformed_labels == label);
            if (!af::anyTrue<bool>(class_index))
                continue;
            af::array samples = X_(af::span, class_index);
            af::gforSet(true);
                af::array deviations = af::pow(samples - this->theta_(af::span, label), 2);
            af::gforSet(false);
            this->stddev_(af::span, label) = af::sum(deviations, 1);
        }
        
        // exchange the standard deviations values
        const size_t n_stddev = n_classes * X_.dims(0);
        this->stddev_.host(reinterpret_cast<void*>(message));
        MPI_Allreduce(MPI_IN_PLACE, message, n_stddev, MPI_FLOAT, MPI_SUM, this->comm_);
        this->stddev_.write(message, n_stddev * sizeof(float));
        
        // normalize standard deviation by class counts and calculate root (not variance)
        gfor (af::seq row, X.n_features()) {
            this->stddev_(row, af::span) /= this->class_counts_;
        }
        this->stddev_ = af::sqrt(this->stddev_);
        
        // release message buffer
        delete[] message;
    }

    Dataset GaussianNaiveBayes::predict_probability(Dataset& X) const {
        const dim_t n_classes = this->class_normalizer_.n_classes();
        af::setBackend(static_cast<af::Backend>(this->backend_.get()));
        X.load_equal_chunks();
        af::array probabilities = af::constant(1.0f, n_classes, X.n_samples());
        
        const af::array& X_ = X.data();
        gfor (af::seq sample, X.n_samples()) {
            probabilities(af::span, sample) *= this->prior_.T();
            
            for (int label = 0; label < n_classes; ++label) {
                af::array mean = this->theta_(af::span, label);
                af::array stddev = this->stddev_(af::span, label);
                af::array X_row = X_(af::span, sample);
                af::array class_probability = gaussian_pdf(X_row, mean, stddev);
                probabilities(label, sample) = af::product(class_probability, 0);
            }
        }
        
        return Dataset(probabilities, this->comm_);
    }

    Dataset GaussianNaiveBayes::predict(Dataset& X) {
        // X is loaded in this->predict_probability
        Dataset probabilities = this->predict_probability(X);
        
        af::array values(X.n_samples());
        af::array locations(X.n_samples());
        
        af::max(values, locations, probabilities.data(), 0);
        af::array locations_orig = this->class_normalizer_.invert(locations);
                
        return Dataset(locations_orig, this->comm_);
    }

    float GaussianNaiveBayes::accuracy(Dataset& X, Dataset& y) {
        float local_results[2];
        Dataset predictions = this->predict(X);
        // X is loaded in this->predict
        y.load_equal_chunks();

        af::array sum = af::sum(predictions.data() == y.data());
        local_results[0] = (float)sum.scalar<uint>();
        local_results[1] = (float)y.n_samples();
        MPI_Allreduce(MPI_IN_PLACE, local_results, 2, MPI_FLOAT, MPI_SUM, this->comm_);
        
        return local_results[0] / local_results[1];
    }
    
    const af::array& GaussianNaiveBayes::class_counts() const {
        return this->class_counts_;
    }
    
    const af::array& GaussianNaiveBayes::prior() const {
        return this->prior_;
    }
    
    const af::array& GaussianNaiveBayes::stddev() const {
        return this->stddev_;
    }
    
    const af::array& GaussianNaiveBayes::theta() const {
        return this->theta_;
    }
} // namespace juml

