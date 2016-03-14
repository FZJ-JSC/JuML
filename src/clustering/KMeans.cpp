/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: KMeans.cpp
*
* Description: Implementation of class KMeans
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#include <random>
#include <set>
#include <stdexcept>

#include "core/MPI.h"
#include "clustering/KMeans.h"

namespace juml {
    KMeans::KMeans(
           uintl    k,
           uintl    max_iter,
           Method   initialization,
           Distance distance,
           float    tolerance,
           uintl    seed,
           int      backend,
           MPI_Comm comm)
      : BaseClusterer(backend, comm),
        k_(k),
        max_iter_(max_iter),
        initialization_(initialization),
        distance_(distance),
        tolerance_(tolerance),
        seed_(seed) {
        if (k < 2)
            throw std::invalid_argument("The number of k must be at least 2");
        if (tolerance < 0.0f)
            throw std::invalid_argument("The tolerance must be larger then 0");
    }

    void KMeans::initialize_random_centroids(const Dataset& dataset) {
        const af::array& data = dataset.data();

        // initialize random generator
        std::mt19937 random_state(this->seed_);
        std::uniform_int_distribution<intl> index_selector(0, dataset.global_items() - 1);

        // create a centroid array that holds the locally chosen centroids
        this->centroids_ = af::constant(0, data.dims(0), static_cast<dim_t>(this->k_), data.type());

        // randomly roll the centroids
        for (uint i = 0; i < this->k_; ++i) {
            intl index = index_selector(random_state);
            if (dataset.global_offset() <= index && index <= dataset.global_offset() + data.dims(1)) {
                this->centroids_(af::span, i) = data(af::span, index - dataset.global_offset());
            }
        }
        mpi::allreduce_inplace(this->centroids_, MPI_SUM, this->comm_);
    }
    
    void KMeans::initialize_kpp_centroids(const Dataset& data) {
        throw "Not implemented yet...genius!";
    }

    af::array KMeans::closest_centroids(const af::array& data) const {
        // calculate the distances using a euclidean distance
        af::array distances = this->distance_(this->centroids_, data);

        // get the locations where the distance is minimal
        af::array minimum_values, locations;
        af::min(minimum_values, locations, distances, 0 /* along the first - sample - dimension */);

        return af::moddims(locations, 1, data.dims(1));
    }

    void KMeans::cluster(const Dataset& dataset) {
        const af::array& data = dataset.data();

        // dimensions
        dim_t f = data.dims(0); // features
        dim_t k = static_cast<dim_t>(this->k_); // centroid count
        dim_t n = data.dims(1); // number of sample

        // calculate the convergence threshold globally
        uintl threshold = static_cast<uintl>(std::floor(this->tolerance_ * dataset.global_items()));

        // remember the cluster assignments in order to break on changes
        af::array previous_assignments = af::constant(-1, 1, n);

        // perform actual clustering
        for (uint i = 0; i < this->max_iter_; ++i) {
            af::array locations = this->closest_centroids(data);
            af::array changes = af::tile(af::sum(previous_assignments != locations), f);

            // update the centroids
            af::array centroid_update = af::constant(0, f, 2 * k + 1, data.type());
            gfor (af::seq j, k) {
                af::array closeness_volume = af::tile((locations == j), f);
                centroid_update(af::span, j) = af::sum(data * closeness_volume, 1 /* along the samples */);
                af::seq second_half(af::seq(k, 2 * k - 1), true);
                centroid_update(af::span, second_half) = af::sum(closeness_volume, 1);
            }
            centroid_update(af::span, 2 * k) = changes;

            // exchange and average update
            mpi::allreduce_inplace(centroid_update, MPI_SUM, this->comm_);
            for (uintl j = 0; j < k; ++j) {
                centroid_update(af::span, j) /= centroid_update(af::span, j + (double)k);
            }
            this->centroids_ = centroid_update(af::span, af::seq(0, k - 1));

            // count the number of cluster assignment changes and leave the loop if below threshold
            if ((centroid_update(0, 2 * k) < threshold).as(u8).scalar<unsigned char>()) break;
            previous_assignments = locations;
        }
    }
    
    void KMeans::fit(Dataset& X) {
        // initialize backend and load data
        af::setBackend(static_cast<af::Backend>(this->backend_.get()));
        X.load_equal_chunks();

        // dimensionality checks
        af::array& data = X.data();
        if (data.dims(2) > 1 || data.dims(3) > 1) {
            throw std::logic_error("K-Means clustering is only defined for two-dimensional input data");
        }

        // initialize centroids
        switch (this->initialization_) {
            case Method::RANDOM:
                this->initialize_random_centroids(X);
                break;
            case Method::KMEANS_PLUS_PLUS:
                this->initialize_kpp_centroids(X);
                break;
            default:
                throw std::invalid_argument("Unsupported initialization method");
        }

        // actually cluster the data
        this->cluster(X);
    }

    Dataset KMeans::predict(Dataset& X) const {
        af::setBackend(static_cast<af::Backend>(this->backend_.get()));
        X.load_equal_chunks();

        af::array locations = this->closest_centroids(X.data());
        return Dataset(locations, this->comm_);
    }

    const af::array& KMeans::centroids() const {
        return this->centroids_;
    }

    const uintl KMeans::k() const {
        return this->k_;
    }

    const uintl KMeans::max_iter() const {
        return this->max_iter_;
    }

    const KMeans::Method KMeans::initialization() const {
        return this->initialization_;
    }

    const Distance KMeans::distance() const {
        return this->distance_;
    }

    const uintl KMeans::seed() const {
        return this->seed_;
    }

    const float KMeans::tolerance() const {
        return this->tolerance_;
    }
} // namespace juml
