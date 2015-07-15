#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <armadillo>
#include <limits>
#include <stdint.h>

namespace juml {
namespace utils {
    template <typename T>
    arma::uvec argmax_cols(const arma::Mat<T>& X) {
        arma::uvec indices(X.n_cols);

        for (uint64_t col = 0; col < X.n_cols; ++col) {
            T compare = std::numeric_limits<T>::lowest();
            for (uint64_t row = 0; row < X.n_rows; ++row) {
                const auto element = X(row, col);

                if (element > compare) {
                    compare = element;
                    indices(col) = row;
                }
            }
        }
    }

    template <typename T>
    arma::uvec argmin_cols(const arma::Mat<T>& X) {
        arma::uvec indices(X.n_cols);

        for (uint64_t col = 0; col < X.n_cols; ++col) {
            T compare = std::numeric_limits<T>::max();
            for (uint64_t row = 0; row < X.n_rows; ++row) {
                const auto element = X(row, col);

                if (element < compare) {
                    compare = element;
                    indices(col) = row;
                }
            }
        }

        return indices;
    }

    template <typename T>
    arma::uvec argmax_rows(const arma::Mat<T>& X) {
        arma::uvec indices(X.n_rows);

        for (uint64_t row = 0; row < X.n_rows; ++row) {
            T compare = std::numeric_limits<T>::lowest();
            for (uint64_t col = 0; col < X.n_cols; ++col) {
                const auto element = X(row, col);

                if (element > compare) {
                    compare = element;
                    indices(row) = col;
                }
            }
        }

        return indices;
    }

    template <typename T>
    arma::uvec argmin_rows(const arma::Mat<T>& X) {
        arma::uvec indices(X.n_rows);

        for (uint64_t row = 0; row < X.n_rows; ++row) {
            T compare = std::numeric_limits<T>::max();
            for (uint64_t col = 0; col < X.n_cols; ++col) {
                const auto element = X(row, col);

                if (element < compare) {
                    compare = element;
                    indices(row) = col;
                }
            }
        }

        return indices;
    }

    template <typename T>
    arma::uvec argmax(const arma::Mat<T>& X, int dim=0) {
        if (dim == 0) {
            return argmax_cols(X);
        }
        return argmax_rows(X);
    }

    template <typename T>
    arma::uvec argmin(const arma::Mat<T>& X, int dim=0) {
        if (dim == 0) {
            return argmin_cols(X);
        }
        return argmin_rows(X);

    }
} // namespace utils
} // namespace juml

#endif // OPERATIONS_H
