/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: GaussianNaiveBayes.h
*
* Description: Header of class GaussianNaiveBayes
*
* Maintainer: p.glock
* 
* Email: phil.glock@gmail.com
*/

#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

#include <arrayfire.h>
#include <mpi.h>

#include "core/Backend.h"
#include "classification/BaseClassifier.h"
#include "data/Dataset.h"

namespace juml {
    class GaussianNaiveBayes : public BaseClassifier {
    protected:
        af::array class_counts_;
        af::array prior_;
        af::array stddev_;
        af::array theta_;

    public:
        GaussianNaiveBayes(int backend=Backend::CPU, MPI_Comm comm=MPI_COMM_WORLD);
    
        virtual void fit(Dataset& X, Dataset& y);
        virtual Dataset predict(Dataset& X) const override;
        virtual Dataset predict_probability(Dataset& X) const;
        virtual float accuracy(Dataset& X, Dataset& y) const override;

        const af::array& class_counts() const;
        const af::array& prior() const;
        const af::array& stddev() const;
        const af::array& theta() const;
    };
} // namespace juml

#endif // GAUSSIANNAIVEBAYES_H
