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
    //! KMeans
    //! TODO: Describe me
    class KMeans : public BaseClusterer {
    public:
        enum Method {
            RANDOM,
            KMEANS_PLUS_PLUS
        };
        
    protected:
        af::array centroids_;
        uintl     k_;
        uintl     max_iter_;
        Method    initialization_;
        Distance  distance_;
        uintl     seed_;
        float     tolerance_;
        
        void initialize_random_centroids(const Dataset& data);
        void initialize_kpp_centroids(const Dataset& data);
        void cluster(const Dataset& data);
        af::array closest_centroids(const af::array& centroids, const af::array& data) const;
        
    public:
        //! KMeans constructor
        //!
        //!
        KMeans(uintl      k,
               uintl      max_iter=100,
               Method     initialization=KMEANS_PLUS_PLUS,
               Distance   distance=euclidean,
               float      tolerance=1e-3,
               uintl      seed=42L,
               int        backend=Backend::CPU,
               MPI_Comm   comm=MPI_COMM_WORLD);

        KMeans(uintl      k,
               af::array& centroids,
               uintl      max_iter=100,
               Distance   distance=euclidean,
               float      tolerance=1e-3,
               uintl      seed=42L,
               int        backend=Backend::CPU,
               MPI_Comm   comm=MPI_COMM_WORLD);

        virtual void fit(Dataset& X) override;
        virtual Dataset predict(Dataset& X) const override;

        const af::array& centroids() const;
        const uintl k() const;
        const uintl max_iter() const;
        const Method initialization() const;
        const Distance distance() const;
        const uintl seed() const;
        const float tolerance() const;

        virtual void load(const std::string& filename);
        virtual void save(const std::string& filename, bool override=true) const;
    }; // KMeans
}  // juml

#endif // KMEANS_H
