#include "classification/GaussianNaiveBayes.h"
#include "stats/Distributions.h"
#include "utils/operations.h"

namespace juml {
namespace classification {

    void GaussianNaiveBayes::fit(const arma::fmat& X, const arma::ivec& y) {
        BaseClassifier::fit(X, y);
        const int32_t n_classes = this->class_normalizer_.n_classes();
        this->class_counts_.zeros(n_classes);
        this->prior_.zeros(n_classes);
        this->theta_.zeros(n_classes, X.n_cols);
        this->stddev_.zeros(n_classes, X.n_cols);

        #pragma omp parallel for
        for (uint64_t row = 0; row < X.n_rows ; ++row) {
            const uint64_t class_index = this->class_normalizer_.transform(y(row));
            ++class_counts_(class_index);
            ++this->prior_(class_index);
            this->theta_.row(class_index) += X.row(row);
        }
        this->prior_ /= y.n_elem;
        this->theta_.each_col() /= class_counts_;

        #pragma omp parallel for
        for (uint64_t row = 0; row < X.n_rows; ++row) {
            const uint64_t class_index = this->class_normalizer_.transform(y(row));
            arma::frowvec deviation = X.row(row) - this->theta_.row(class_index);
            this->stddev_.row(class_index) += arma::pow(deviation, 2);
        }

        this->stddev_.each_col() /= class_counts_;
        this->stddev_ = arma::sqrt(this->stddev_);
    }

    arma::fmat GaussianNaiveBayes::predict_probability(const arma::fmat& X) const {
        arma::fmat probabilities = arma::ones<arma::fmat>(X.n_rows, this->class_normalizer_.n_classes());

        for (uint64_t i = 0; i < this->prior_.n_elem; ++i) {
            const float prior = this->prior_(i);
            probabilities.col(i) *= prior;

            #pragma omp parallel for
            for (uint64_t row = 0; row < X.n_rows; ++row) {
                const arma::frowvec& mean = this->theta_.row(i);
                const arma::frowvec& stddev = this->stddev_.row(i);
                arma::frowvec features_probs = juml::stats::gaussian_pdf(X.row(row), mean, stddev);
                probabilities(row, i) *= arma::prod(features_probs);
            }
        }

        return probabilities;
    }

    arma::ivec GaussianNaiveBayes::predict(const arma::fmat& X) const {
        arma::fmat probabilities = this->predict_probability(X);
        arma::ivec predictions(X.n_rows);
        arma::uvec max_index = juml::utils::argmax(probabilities, 1);

        for (uint64_t i = 0; i < max_index.n_elem; ++i) {
            predictions(i) = this->class_normalizer_.invert(max_index(i));
        }

        return predictions;
    }

    float GaussianNaiveBayes::accuracy(const arma::fmat& X, const arma::ivec& y) const {
        arma::ivec predictions = this->predict(X);
        return (float)arma::sum(predictions == y) / (float)y.n_elem;
    }
} // namespace classification
} // namespace juml
