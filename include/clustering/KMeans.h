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

#include <armadillo>

#include "clustering/BaseClassifier.h"

namespace juml {
    //! KMeans
    //! TODO: Describe me
    class KMeans : public BaseClassifier {
    public:
        enum Method {
            RANDOM,
            KMEANS_PLUS_PLUS,
            GIVEN
        };
        //! KMeans constructor
        //!
        //!
        KMeans(unsigned int k, 
               unsigned int max_iter=100, 
               unsigned n_init=1, 
               Method init=RANDOM, 
               float tolerance=1e-3, 
               int random_seed, 
               const arma::Mat<float>& centroids);
        
        
    }; // KMeans
}  // juml
#endif // KMEANS_H
