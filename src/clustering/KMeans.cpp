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
#include <core/HDF5.h>

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

    KMeans::KMeans(
            uintl      k,
            af::array& centroids,
            uintl      max_iter,
            Distance   distance,
            float      tolerance,
            uintl      seed,
            int        backend,
            MPI_Comm   comm)
      : KMeans(k, max_iter, static_cast<Method>(-1), distance, tolerance, seed, backend, comm) {
        this->centroids_ = centroids;
        if (this->centroids_.dims(1) != this->k_)
            throw std::invalid_argument("Mismatch between centroid count and number of clusters k");
    }

    void KMeans::initialize_random_centroids(const Dataset& dataset) {
        const af::array& data = dataset.data();

        // initialize random generator
        std::mt19937 random_state(this->seed_);
        std::uniform_int_distribution<intl> index_selector(0, dataset.global_n_samples() - 1);

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
    
    void KMeans::initialize_kpp_centroids(const Dataset& dataset) {
        const af::array& data = dataset.data();

        dim_t f = data.dims(0);
        dim_t n = data.dims(1);

        // initialize random generator
        std::mt19937 random_state(this->seed_);
        std::uniform_int_distribution<intl> index_selector(0, dataset.global_n_samples() - 1);

        // choose first centroid randomly and broadcast it
        af::array centroids = af::constant(0, f, data.type());
        intl index = index_selector(random_state);
        if (dataset.global_offset() <= index && index <= dataset.global_offset() + n)
            centroids += data(af::span, index - dataset.global_offset());
        mpi::allreduce_inplace(centroids, MPI_SUM, this->comm_);

        // calculate the total distance psi and the number of initialization steps
        af::array total_distance = af::sum(this->distance_(centroids, data), 1 /* along samples */);
        mpi::allreduce_inplace(total_distance, MPI_SUM, this->comm_);
        uintl initialization_steps = af::ceil(af::log(total_distance)).as(u64).scalar<uintl>();

        // pick centroid candidates for initilization_steps time (log(psi))
        for (uintl i = 0; i < initialization_steps; ++i) {
            // select closest distance
            af::array distances = af::min(this->distance_(centroids, data), 0);
            // calculate probability
            af::array probabilities = (2.0 * distances) / af::tile(af::sum(distances, 1), 1, n);
            af::array boundary = af::randu(1, n);

            // pick samples and exchange globally
            af::array candidates = data(af::span, probabilities > boundary);

            // obtain global number of candidates and own offset
            intl offset = candidates.dims(1);
            intl items  = offset;
            MPI_Exscan(MPI_IN_PLACE, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &items, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            if (items == 0) continue;
            if (this->mpi_rank_ == 0)
                offset = 0;

            // prepare local candidates for transmission in offset candidate mask
            af::array global_candidates = af::constant(0, f, items, candidates.type());
            if (candidates.dims(1) > 0) {
                af::seq insert_location = af::seq(af::seq(offset, offset + candidates.dims(1) - 1), true);
                global_candidates(af::span, insert_location) = candidates;
            }

            // obtain global candidates and append to local candidates
            mpi::allreduce_inplace(global_candidates, MPI_SUM, this->comm_);
            centroids = af::join(1, centroids, global_candidates);
        }

        // initialize the final centroids to be empty
        this->centroids_ = af::constant(0, f, this->k_, data.type());

        // select k points from the candidates
        dim_t number_of_candidates = centroids.dims(1);
        af::array locations = this->closest_centroids(centroids, data);
        af::array weights = af::constant(0, number_of_candidates, s64);
        gfor (af::seq j, number_of_candidates) {
            weights(j) = af::sum(locations == j);
        }
        mpi::allreduce_inplace(weights, MPI_SUM, this->comm_);

        // initialize the pick arrays
        af::array upper_bounds = af::accum(weights);
        af::array lower_bounds = upper_bounds - weights;
        intl pick_boundary = dataset.global_n_samples() - 1;

        // actually pick from the array
        for (uintl i = 0; i < this->k_; ++i) {
            uintl pick = std::uniform_int_distribution<uintl>(0, pick_boundary)(random_state);
            af::array index = lower_bounds <= pick & pick < upper_bounds;

            // assign the picked centroid
            this->centroids_(af::span, i) = centroids(af::span, index);
            intl weight = weights(index).scalar<intl>();

            // update the pick ranges
            lower_bounds(index) = -1;
            upper_bounds(index) = -1;
            af::array update_index = pick < upper_bounds;
            lower_bounds(update_index) -= weight;
            upper_bounds(update_index) -= weight;
            pick_boundary -= weight;
        }
    }

    af::array KMeans::closest_centroids(const af::array& centroids, const af::array& data) const {
        // calculate the distances using a euclidean distance
        af::array distances = this->distance_(centroids, data);

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
        uintl threshold = static_cast<uintl>(std::floor(this->tolerance_ * dataset.global_n_samples()));

        // remember the cluster assignments in order to break on changes
        af::array previous_assignments = af::constant(-1, 1, n);

        // perform actual clustering
        for (uint i = 0; i < this->max_iter_; ++i) {
            af::array locations = this->closest_centroids(this->centroids_, data);
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
        Backend::set(this->backend_.get());
        X.load_equal_chunks();

        // dimensionality checks
        af::array& data = X.data();
        if (data.dims(2) > 1 || data.dims(3) > 1) {
            throw std::invalid_argument("K-Means clustering is only defined for two-dimensional input data");
        }

        // initialize centroids
        if (this->initialization_ == static_cast<Method>(-1)) {}// centroids given
        else if (this->initialization_ == Method::RANDOM)
            this->initialize_random_centroids(X);
        else if (this->initialization_ == Method::KMEANS_PLUS_PLUS)
            this->initialize_kpp_centroids(X);
        else
            throw std::invalid_argument("Unsupported initialization method");

        // actually cluster the data
        this->cluster(X);
    }

    Dataset KMeans::predict(Dataset& X) const {
        Backend::set(this->backend_.get());
        X.load_equal_chunks();

        af::array locations = this->closest_centroids(this->centroids_, X.data());
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

    void KMeans::save(const std::string& filename, bool override) const {
        Algorithm::save(filename, override);
        if (this->mpi_rank_ == 0) {
            hid_t file_id = juml::hdf5::open_file(filename);
            juml::hdf5::write_array(file_id, "centroids", this->centroids_);
            juml::hdf5::close_file(file_id);
        }
        MPI_Barrier(this->comm_);
    }

    void KMeans::load(const std::string& filename) {
        hid_t file_id = juml::hdf5::popen_file(filename, this->comm_);
        this->centroids_ = juml::hdf5::pread_array(file_id, "centroids");
        this->k_ = this->centroids_.dims(0);
        juml::hdf5::close_file(file_id);
    }
} // namespace juml
