#include <stdint.h>

#include "stats/Distributions.h"

namespace juml {
namespace stats {
    float gaussian_pdf(float x, float mean, float stddev) {
        const float sigma = std::pow(stddev, 2);
        const float prob = 1.0f / std::sqrt(2.0f * M_PI * sigma);
        const float e = std::exp(-std::pow(x - mean, 2) / (2.0f * sigma));

        return prob * e;
    }

    arma::fvec gaussian_pdf(arma::fvec& X, arma::fvec& means, arma::fvec& stddevs) {
        arma::fvec probs(X.n_elem);

        for (uint64_t i = 0; i < X.n_elem; ++i) {
            probs[i] = gaussian_pdf(X[i], means[i], stddevs[i]);
        }

        return probs;
    }
} // stats
} // juml
