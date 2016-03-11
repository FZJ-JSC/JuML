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

#include "core/Definitions.h"
#include "clustering/BaseClusterer.h"
#include "data/Dataset.h"

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

        ulong  k_;
        ulong  max_iter_;
        Method initialization_;
        ulong  seed_;
        float  tolerance_;
        
        void initialize_random_centroids(const af::array& data);
        void initialize_kpp_centroids(const af::array& data);
        void cluster(const af::array& data);
        af::array closest_centroids(const af::array& data) const;
        
    public:
        //! KMeans constructor
        //!
        //!
        KMeans(ulong    k,
               ulong    max_iter=100,
               Method   initialization=RANDOM,
               float    tolerance=1e-3,
               ulong    seed=42L,
               int      backend=Backend::CPU,
               MPI_Comm comm=MPI_COMM_WORLD);        
        virtual void fit(Dataset& X);
        virtual Dataset predict(Dataset& X) const;

        const af::array& centroids() const;
    }; // KMeans
}  // juml
#endif // KMEANS_H
