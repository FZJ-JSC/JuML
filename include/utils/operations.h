/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: DataSet.h
*
* Description: Header of class DataSet
*
* Maintainer: p.glock
*
* Email: phil.glock@gmail.com
*/

#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <armadillo>
#include <limits>

namespace juml {
    template <typename T>
    arma::uvec argmax_cols(const arma::Mat<T>& X) {
        arma::uvec indices(X.n_cols);

        for (size_t col = 0; col < X.n_cols; ++col) {
            T compare = std::numeric_limits<T>::lowest();
            for (size_t row = 0; row < X.n_rows; ++row) {
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

        for (size_t col = 0; col < X.n_cols; ++col) {
            T compare = std::numeric_limits<T>::max();
            for (size_t row = 0; row < X.n_rows; ++row) {
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

        for (size_t row = 0; row < X.n_rows; ++row) {
            T compare = std::numeric_limits<T>::lowest();
            for (size_t col = 0; col < X.n_cols; ++col) {
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

        for (size_t row = 0; row < X.n_rows; ++row) {
            T compare = std::numeric_limits<T>::max();
            for (size_t col = 0; col < X.n_cols; ++col) {
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
} // namespace juml

#endif // OPERATIONS_H
