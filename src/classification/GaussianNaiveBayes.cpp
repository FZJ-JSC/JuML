#include "classification/GaussianNaiveBayes.h"
#include "stats/Distributions.h"
#include "utils/operations.h"

namespace juml {
namespace classification {

    void GaussianNaiveBayes::fit(arma::fmat& X, arma::ivec& y) {
        this->classes_ = arma::unique(y);
        this->class_mapping_ = this->get_class_mapping(this->classes_);
        this->inverse_mapping_ = this->invert_mapping(this->class_mapping_);

        this->class_counts_.zeros(this->classes_.n_elem);
        this->prior_.zeros(this->classes_.n_elem);
        this->theta_.zeros(this->classes_.n_elem, X.n_cols);
        this->stddev_.zeros(this->classes_.n_elem, X.n_cols);

        #pragma omp parallel for
        for (uint64_t row = 0; row < X.n_rows ; ++row) {
            const uint64_t class_index = this->class_mapping_[y(row)];
            ++class_counts_(class_index);
            ++this->prior_(class_index);
            this->theta_.row(class_index) += X.row(row);
        }
        this->prior_ /= y.n_elem;
        this->theta_.each_col() /= class_counts_;

        #pragma omp parallel for
        for (uint64_t row = 0; row < X.n_rows; ++row) {
            const uint64_t class_index = this->class_mapping_[y(row)];
            arma::frowvec deviation = X.row(row) - this->theta_.row(class_index);
            this->stddev_.row(class_index) += arma::pow(deviation, 2);
        }

        this->stddev_.each_col() /= class_counts_;
        this->stddev_ = arma::sqrt(this->stddev_);
    }

    arma::fmat GaussianNaiveBayes::predict_probability(arma::fmat& X) const {
        arma::fmat probabilities = arma::ones<arma::fmat>(X.n_rows, this->classes_.n_elem);

        for (const auto& mapping : this->class_mapping_) {
            const float prior = this->prior_(mapping.second);
            probabilities.col(mapping.second) *= prior;

            #pragma omp parallel for
            for (uint64_t row = 0; row < X.n_rows; ++row) {
                const arma::frowvec& mean = this->theta_.row(mapping.second);
                const arma::frowvec& stddev = this->stddev_.row(mapping.second);
                arma::frowvec features_probs = juml::stats::gaussian_pdf(X.row(row), mean, stddev);
                probabilities(row, mapping.second) *= arma::prod(features_probs);
            }
        }

        return probabilities;
    }

    arma::ivec GaussianNaiveBayes::predict(arma::fmat& X) const {
        arma::fmat probabilities = this->predict_probability(X);
        arma::ivec predictions(X.n_rows);
        arma::uvec max_index = juml::utils::argmax(probabilities, 1);

        for (uint64_t i = 0; i < max_index.n_elem; ++i) {
            predictions(i) = this->inverse_mapping_.find(max_index(i))->second;
        }

        return predictions;
    }

    float GaussianNaiveBayes::accuracy(arma::fmat& X, arma::ivec& y) const {
        arma::ivec predictions = this->predict(X);
        return (float)arma::sum(predictions == y) / (float)y.n_elem;
    }

    std::map<int, uint64_t> GaussianNaiveBayes::get_class_mapping(arma::ivec& classes) const {
        std::map<int, uint64_t> mapping;
        for (uint64_t i = 0; i < classes.n_elem; ++i) {
            mapping[classes(i)] = i;
        }

        return mapping;
    }

    std::map<uint64_t, int> GaussianNaiveBayes::invert_mapping(std::map<int, uint64_t>& mapping) const {
        std::map<uint64_t, int> inverse_mapping;
        for (const auto& pair : mapping) {
            inverse_mapping[pair.second] = pair.first;
        }

        return inverse_mapping;
    }
} // namespace classification
} // namespace juml
