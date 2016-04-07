/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: KMeans.h
*
* Description: Header of class KMeans
*
* Maintainer: m.goetz
*
* Email: murxman@gmail.com
*/

#ifndef KMEANS_H
#define KMEANS_H

#include <arrayfire.h>
#include <mpi.h>

#include "core/Backend.h"
#include "clustering/BaseClusterer.h"
#include "data/Dataset.h"
#include "spatial/Distances.h"

namespace juml {
    /**
     * K-Means
     *
     * K-Means is popular unsupervised clustering algorithm that assigns data points to predefined number (k) different
     * clusters. The average run-time complexity of this implementation using Lloyd's algorithm is O(k*n*max_iterations)
     * and is therefore compared to other clustering algorithms fast. K-Means tends to fall into local minima and should
     * be executed a number of times with different starting centroids or seeds.
     *
     * Example:
     *
     * @code
     * #include <JuML.h>
     *
     * using namespace juml;
     *
     * Dataset X("train.h5", "data");
     * juml::KMeans kmeans(3); // k=3
     * kmeans.fit(X);
     * @endcode
     */
    class KMeans : public BaseClusterer {
    public:
        /**
         * Method
         *
         * K-Means initialization method symbols
         */
        enum Method {
            RANDOM,          /** Choose initial centroids randomly */
            KMEANS_PLUS_PLUS /** Use a smart heuristic to guess the initial centroids */
        };
        
    protected:
        /**
         * @var   centroids_
         * @brief The centroids of the K-Means clustering, defaults to empty array before fit
         */
        af::array centroids_;
        /**
         * @var   k_
         * @brief The number of clusters (k) to detect
         */
        uintl     k_;
        /**
         * @var   max_iter_
         * @brief The maximum number of Lloyd iterations to perform despite no convergence
         */
        uintl     max_iter_;
        /**
         * @var   initialization_
         * @brief The selected centroid initialization method Method
         */
        Method    initialization_;
        /**
         * @var   distance_
         * @brief The selected distance function (e.g. euclidean)
         */
        Distance  distance_;
        /**
         * @var   seed_
         * @brief The random seed for centroid selection, only applicable for Method::RANDOM
         */
        uintl     seed_;
        /**
         * @var   tolerance_
         * @brief Fraction of moving points (rounded up) below which the clustering is considered to be converged
         */
        float     tolerance_;

        /**
         * initialize_random_centroids
         *
         * Initializes the centroids randomly in a distributed environment
         *
         * @param data - The training dataset passed to fit
         */
        void initialize_random_centroids(const Dataset& data);

        /**
         * initialize_kpp_centroids
         *
         * Initializes the centroids using a smart heuristic called scalable K-Means++ by Bahmani et al. The original
         * publication can be found here http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf. Works more
         * efficient in distributed environments compared to the original K-Means++ by breaking sequential computation
         * steps, but converges to the same result.
         *
         * @param data - The training dataset passed to fit
         */
        void initialize_kpp_centroids(const Dataset& data);

        /**
         * cluster
         *
         * Internal implementation meat that performs the Lloyd iterations on the data. Performs convergence and
         * maximum iteration checks. The centroids are synchronized across all nodes.
         *
         * @param data - The training dataset passed to fit
         */
        void cluster(const Dataset& data);

        /** closest_centroids
         *
         * Utility function that computes for each data item the closest centroid from a set of given centroids. The
         * function is used both in cluster and predict.
         *
         * @param centroids - A f x k matrix modeling the centroids
         * @param data      - A f x n matrix modeling the cluster points
         * @returns A 1 x n row-vector with the index of the closest centroid
         */
        af::array closest_centroids(const af::array& centroids, const af::array& data) const;
        
    public:
        /**
         * KMeans constructor
         *
         * This constructor allows to initialize a K-Means clusterer with a chosen initialization Method.
         *
         * @param k              - The number of clusters k to cluster the data into
         * @param max_iter       - The maximum number of Lloyd iteration to perform, defaults to 100
         * @param initialization - Method how to initialize the initial centroids, defaults to K-Means++
         * @param distance       - The distance metric for the clustering, defaults to euclidean
         * @param tolerance      - Fraction of moved points (rounded up) below which the found clustering is considered
         *                         to have converged, defaults to 1e-3.
         * @param seed           - Random seed for centroids initialization, is only relevant for Method::RANDOM.
         * @param backend        - The computation backend for the clustering, defaults to CPU
         * @param comm           - The MPI communicator used for the processing
         * @throws invalid_argument if k is less than two or tolerance is less than 0.
         */
        KMeans(uintl      k,
               uintl      max_iter=100,
               Method     initialization=KMEANS_PLUS_PLUS,
               Distance   distance=euclidean,
               float      tolerance=1e-3,
               uintl      seed=42L,
               int        backend=Backend::CPU,
               MPI_Comm   comm=MPI_COMM_WORLD);

        /**
         * KMeans constructor
         *
         * This constructor allows to initialize a K-Means clusterer with a given set of centroids.
         *
         * @param k              - The number of clusters k to cluster the data into
         * @param centroids      - The initial set of centroids
         * @param max_iter       - The maximum number of Lloyd iteration to perform, defaults to 100
         * @param distance       - The distance metric for the clustering, defaults to euclidean
         * @param tolerance      - Fraction of moved points (rounded up) below which the found clustering is considered
         *                         to have converged, defaults to 1e-3.
         * @param seed           - Random seed for centroids initialization, is only relevant for Method::RANDOM.
         * @param backend        - The computation backend for the clustering, defaults to CPU
         * @param comm           - The MPI communicator used for the processing
         * @throws invalid_argument if k is less than two, tolerance is less than 0 or k #centroids mismatch.
         */
        KMeans(uintl      k,
               af::array& centroids,
               uintl      max_iter=100,
               Distance   distance=euclidean,
               float      tolerance=1e-3,
               uintl      seed=42L,
               int        backend=Backend::CPU,
               MPI_Comm   comm=MPI_COMM_WORLD);

        /**
         * fit
         *
         * Performs the clustering on dataset and computes the centroids of the k clusters to detect.
         *
         * @param X - The training data
         * @throws invalid_argument if X has more then two dimensions or the initialization method is not applicable
         */
        virtual void fit(Dataset& X) override;

        /**
         * predict
         *
         * Assigns each of the passed test data points to cluster/centroid.
         *
         * @param X - The data to be predicted
         * @returns A dataset with 1xn row-vector with the predicted labels/centroid indices
         */
        virtual Dataset predict(Dataset& X) const override;

        const af::array& centroids() const;
        const uintl k() const;
        const uintl max_iter() const;
        const Method initialization() const;
        const Distance distance() const;
        const uintl seed() const;
        const float tolerance() const;
    }; // KMeans
}  // juml

#endif // KMEANS_H
