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
#include <limits>
#include <core/HDF5.h>

#include "core/Backend.h"
#include "core/MPI.h"
#include "classification/GaussianNaiveBayes.h"
#include "stats/Distributions.h"

namespace juml {
    GaussianNaiveBayes::GaussianNaiveBayes(int backend, MPI_Comm comm)
      : BaseClassifier(backend, comm)
    {}
    
    void GaussianNaiveBayes::fit(Dataset& X, Dataset& y) {
        Backend::set(this->backend_.get());
        
        X.load_equal_chunks();
        y.load_equal_chunks();
        BaseClassifier::fit(X, y);        
        
        const af::array& X_ = X.data();
        const af::array& y_ = y.data();
        
        const dim_t n_classes = this->class_normalizer_.n_classes();
        this->class_counts_ = af::constant(0.0f, 1, n_classes);
        this->prior_ = af::constant(0.0f, 1, n_classes);
        this->stddev_ = af::constant(0.0f, X.n_features(), n_classes);
        this->theta_ = af::constant(0.0f, X.n_features(), n_classes);
        
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
        if (n_floats > std::numeric_limits<int>::max())
            throw std::domain_error("Message too large");
        float message[n_floats];
        
        // create message
        this->class_counts_.host(reinterpret_cast<void*>(message));
        this->prior_.host(reinterpret_cast<void*>(message + n_classes));
        this->theta_.host(reinterpret_cast<void*>(message + n_classes * 2));
        message[n_floats - 1] = (float)y.n_samples();
        
        // reduce to obtain global view
        MPI_Allreduce(MPI_IN_PLACE, message, static_cast<int>(n_floats), MPI_FLOAT, MPI_SUM, this->comm_);
        
        // copy the reduced elements back        
        this->class_counts_.write(message, n_classes * sizeof(float));
        this->prior_.write(message + n_classes, n_classes * sizeof(float));
        this->theta_.write(message + n_classes * 2, this->theta_.elements() * sizeof(float));
        
        // extract reduced class counts, prior and theta from message
        const intl total_n_y = static_cast<intl>(message[n_floats - 1]);
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
        mpi::allreduce_inplace(this->stddev_, MPI_SUM, this->comm_);
        
        // normalize standard deviation by class counts and calculate root (not variance)
        gfor (af::seq row, X.n_features()) {
            this->stddev_(row, af::span) /= this->class_counts_;
        }
        this->stddev_ = af::sqrt(this->stddev_);
    }

    Dataset GaussianNaiveBayes::predict_probability(Dataset& X) const {
        const dim_t n_classes = this->class_normalizer_.n_classes();
        Backend::set(this->backend_.get());
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

    Dataset GaussianNaiveBayes::predict(Dataset& X) const {
        // X is loaded in this->predict_probability
        Dataset probabilities = this->predict_probability(X);
        
        af::array values(X.n_samples());
        af::array locations(X.n_samples());
        
        af::max(values, locations, probabilities.data(), 0);
        af::array locations_orig = this->class_normalizer_.invert(locations);
                
        return Dataset(locations_orig, this->comm_);
    }

    float GaussianNaiveBayes::accuracy(Dataset& X, Dataset& y) const {
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

    void GaussianNaiveBayes::save(const std::string& filename) const {
        if (this->mpi_rank_ == 0) {
            hid_t file_id = hdf5::open_file(filename.c_str());
            hdf5::write_array(file_id, "class_counts", this->class_counts_);
            hdf5::write_array(file_id, "prior", this->prior_);
            hdf5::write_array(file_id, "stddev", this->stddev_);
            hdf5::write_array(file_id, "theta", this->theta_);
            hdf5::close_file(file_id);
        }
    }

    void GaussianNaiveBayes::load(const std::string &filename) {
        hid_t plist;
        //hid_t file_id = hdf5::popen_file(filename, plist,this->comm_);
        hid_t access_plist;

        access_plist = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(access_plist, MPI_COMM_WORLD, MPI_INFO_NULL);
        hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, access_plist);
        this->class_counts_ = hdf5::pread_array(file_id, "class_counts");
        this->prior_ = hdf5::pread_array(file_id, "prior");
        this->stddev_ = hdf5::pread_array(file_id, "stddev");
        this->theta_ = hdf5::pread_array(file_id, "theta");
        hdf5::close_file(file_id);
        H5Pclose(access_plist);
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
