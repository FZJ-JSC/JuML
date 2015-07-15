#ifndef BASECLASSIFIER_H
#define BASECLASSIFIER_H

#include <armadillo>
#include <mpi.h>

namespace juml {
namespace classification {
    class BaseClassifier {
    protected:
        MPI_Comm comm_;
        
    public:
        BaseClassifier(MPI_Comm comm=MPI_COMM_WORLD) :
            comm_(comm)
        {};

        virtual void fit(arma::fmat& X, arma::ivec& y) = 0;
        virtual arma::ivec predict(arma::fmat& X) const = 0;
        virtual float accuracy(arma::fmat& X, arma::ivec& y) const = 0;
    };
} // namespace classification
} // namespace juml

#endif // BASECLASSIFIER_H
