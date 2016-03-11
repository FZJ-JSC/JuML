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
           ulong    k,
           ulong    max_iter,
           Method   initialization,
           float    tolerance,
           ulong    seed,
           int      backend,
           MPI_Comm comm)
      : BaseClusterer(backend, comm), 
        k_(k), 
        max_iter_(max_iter), 
        initialization_(initialization), 
        tolerance_(tolerance),
        seed_(seed)
    {}    

    void KMeans::initialize_random_centroids(const af::array& data) {
        std::mt19937 random_state(this->seed_);
        std::uniform_int_distribution<int> rank_selector(0, this->mpi_size_ - 1);
        std::uniform_int_distribution<uint> index_selector(0, static_cast<uint>(data.dims(1)) - 1);

        // create a centroid array that holds the locally chosen centroids
        this->centroids_ = af::constant(0, data.dims(0), this->k_, data.type());

        // randomly roll the centroids
        for (uint i = 0; i < this->k_; ++i) {
            // roll the mpi and index rank where we pick the initial centroid from
            // note: all nodes have to roll here both values, so that the random state advances consistently
            int rank = rank_selector(random_state);
            uint index = index_selector(random_state);

            if (rank != this->mpi_rank_) continue;
            this->centroids_(af::span, i) = data(af::span, index);
        }

        mpi::allreduce_inplace(this->centroids_, MPI_SUM, this->comm_);
    }
    
    void KMeans::initialize_kpp_centroids(const af::array& data) {
        throw "Not implemented yet...genius!";
    }

    af::array KMeans::closest_centroids(const af::array& data) const {
        dim_t f = data.dims(0); // features
        dim_t k = static_cast<dim_t>(this->k_); // centroid count
        dim_t n = data.dims(1); // number of sample

        // reshape the data to allow vectorization, we basically want to have to volumes (3D), that have the following
        // dimensions: features x samples x k repetitions. do the same for the centroids inside the loop
        af::array data_volume = af::tile(data, 1, 1, k);
        af::array centroids_volume = af::tile(af::moddims(this->centroids_, f, 1, k), 1, n, 1);

        // calculate the distances using a euclidean distance
        af::array distances = af::sqrt(af::sum(af::pow(centroids_volume - data_volume, 2), 0 /* along features */));

        // get the locations where the distance is minimal
        af::array minimum_values, locations;
        af::min(minimum_values, locations, distances, 2 /* along the third, or centroid, dimension */);

        return locations;
    }

    void KMeans::cluster(const af::array& data) {
        dim_t f = data.dims(0); // features
        dim_t k = static_cast<dim_t>(this->k_); // centroid count
        dim_t n = data.dims(1); // number of sample

        // calculate the convergence threshold globally
        uintl threshold = static_cast<uintl>(n);
        MPI_Allreduce(MPI_IN_PLACE, &threshold, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm_);
        threshold = static_cast<uintl>(std::floor(this->tolerance_ * threshold));

        // remember the cluster assignments in order to break on changes
        af::array previous_assignments = af::constant(-1, 1, n);

        // perform actual clustering
        for (uint i = 0; i < this->max_iter_; ++i) {
            af::array locations = this->closest_centroids(data);

            // update the centroids
            af::array centroid_update = af::constant(0, f, 2 * k, data.type());

            gfor (af::seq j, k) {
                af::array closeness_volume = af::tile((locations == j), f);
                centroid_update(af::span, j) = af::sum(data * closeness_volume, 1 /* along the samples */);
                af::seq second_half(af::seq(k, 2 * k - 1), true);
                centroid_update(af::span, second_half) = af::sum(closeness_volume, 1);
            }
            mpi::allreduce_inplace(centroid_update, MPI_SUM, this->comm_);
            for (uintl j = 0; j < k; ++j) {
                centroid_update(af::span, j) /= centroid_update(af::span, j + (double)k);
            }
            this->centroids_ = centroid_update(af::span, af::seq(0, k - 1));

            // count the number of cluster assignment changes and leave the loop if below threshold
            uintl changes = af::sum<uintl>(previous_assignments != locations);
            if (changes < threshold) break;
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
                this->initialize_random_centroids(data);
                break;
            case Method::KMEANS_PLUS_PLUS:
                this->initialize_kpp_centroids(data);
                break;
            default:
                throw std::domain_error("Unsupported initialization method");
        }

        // actually cluster the data
        this->cluster(data);
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
} // namespace juml
