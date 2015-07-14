#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <armadillo>
#include <math.h>

namespace juml {
namespace stats {
    float gaussian_pdf(float x, float mean, float stddev);
    arma::fvec gaussian_pdf(const arma::fvec& X, const arma::fvec& means, const arma::fvec& stddevs);
} // stats
} // juml

#endif // DISTRIBUTIONS_H
