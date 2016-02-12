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
* Maintainer:
*
* Email:
*/

#include <random>
#include <set>
#include <stdexcept>

#include "clustering/KMeans.h"

namespace juml {
    KMeans(unsigned int k, 
           unsigned int max_iter, 
           Method initialization, 
           float tolerance,
           int backend,
           MPI_Comm comm)
      : BaseClusterer(backend, comm), 
        k_(k), 
        max_iter_(max_iter), 
        initialization_(initialization), 
        tolerance_(tolerance)
    {}    

    void initialize_random_centroids(const af::array& data) {
        // determine how many of the centroids we have to pick locally
        uint picks = this->k_ / this->mpi_size_ + (this->k_ % this->mpi_size_ < this->mpi_rank_ ? 1 : 0);
        
        
    }
    
    void initialize_kpp_centroids(const af::array& data) {
        throw "Not implemented yet...genius!";
    }
    
    void KMeans::fit(Dataset& X) {
        af::setBackend(static_cast<af::Backend>(this->backend_.get()));      
        
        X.load_equal_chunks();
        
        switch (this->initialization_) {
        case Method::RANDOM:
            this->initialize_random_centroids(X);
            break;
        case Method::KMEANS_PLUS_PLUS:
            this->initialize_kpp_centroids();
            break;
        default:
            throw std::domain_error("Unsupported initialization method");
        }
    }
    
    Dataset KMeans::predict(Dataset& X) const {
    }
} // namespace juml
