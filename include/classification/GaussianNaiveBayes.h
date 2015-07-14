#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

#include <armadillo>
#include <map>
#include <mpi.h>
#include <stdint.h>

namespace juml {
namespace classification {

    class GaussianNaiveBayes {
    protected:
        arma::ivec classes_;
        std::map<int, uint64_t> class_mapping_;
        std::map<uint64_t, int> inverse_mapping_;
        arma::fvec class_counts_;
        arma::fvec prior_;
        arma::fmat theta_;
        arma::fmat stddev_;
        MPI_Comm comm_;

        std::map<int, uint64_t> get_class_mapping(arma::ivec& classes) const;
        std::map<uint64_t, int> invert_mapping(std::map<int, uint64_t>& mapping) const;

    public:
        GaussianNaiveBayes(MPI_Comm comm=MPI_COMM_WORLD);

        void fit(arma::fmat& X, arma::ivec& y);
        arma::ivec predict(arma::fmat& X) const;
        arma::fmat predict_probability(arma::fmat& X) const;
        float score(arma::fmat& X, arma::ivec& y) const;
        arma::ivec argmax(arma::fmat& X, int dim) const;

        inline const arma::fvec& get_class_counts() const { return this->class_counts_; };
        inline const arma::fvec& get_prior() const { return this->prior_; };
        inline const arma::fmat& get_theta() const { return this->theta_; };
        inline const arma::fmat& get_stddev() const { return this->stddev_; };
    };
} // namespace classification
} // namespace juml

#endif // GAUSSIANNAIVEBAYES_H
