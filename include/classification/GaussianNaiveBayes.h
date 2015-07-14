#ifndef GAUSSIANNAIVEBAYES_H
#define GAUSSIANNAIVEBAYES_H

#include <armadillo>

namespace juml {
namespace classification {

    class GaussianNaiveBayes {
    protected:
        arma::vec<float> prior_;
        arma::Mat<float> theta_;
        arma::Mat<float> stddev_;
    public:
        void fit(arma::Mat<float>& X, arma::vec<int>& y);
        arma::vec<int> predict(arma::Mat<float>& X);
        arma::Mat<float> predictProbability(arma::Mat<float>& X);
        float score(arma::Mat<float>& X, arma::vec<int>& y);
    };

} // namespace classification
} // namespace juml

#endif //GAUSSIANNAIVEBAYES_H
