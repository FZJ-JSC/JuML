#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

#include <armadillo>
#include <stdint.h>

#include "classification/BaseClassifier.h"

namespace juml {
namespace classification {

    class GaussianNaiveBayes : public BaseClassifier {
    protected:
        arma::fvec class_counts_;
        arma::fvec prior_;
        arma::fmat theta_;
        arma::fmat stddev_;

    public:
        void fit(const arma::fmat& X, const arma::ivec& y);
        arma::ivec predict(const arma::fmat& X) const;
        arma::fmat predict_probability(const arma::fmat& X) const;
        float accuracy(const arma::fmat& X, const arma::ivec& y) const;

        inline const arma::fvec& class_counts() const {
            return this->class_counts_;
        };
        inline const arma::fvec& prior() const {
            return this->prior_;
        };
        inline const arma::fmat& theta() const {
            return this->theta_;
        };
        inline const arma::fmat& stddev() const {
            return this->stddev_;
        };
    };
} // namespace classification
} // namespace juml

#endif // GAUSSIANNAIVEBAYES_H
