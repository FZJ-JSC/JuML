#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <armadillo>
#include <math.h>

namespace juml {
namespace stats {
    float gaussian_pdf(float x, float mean, float stddev);
    arma::frowvec gaussian_pdf(const arma::frowvec& X, const arma::frowvec& means, const arma::frowvec& stddevs);
} // stats
} // juml

#endif // DISTRIBUTIONS_H
