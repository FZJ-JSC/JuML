// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Kernel.h
*
* Description: Header of class Kernel
*
* Maintainer:
*
* Email:
*/

#ifndef JUML_SVM_KERNEL_H_
#define JUML_SVM_KERNEL_H_

#include <armadillo>
#include <math.h>

namespace juml {

    namespace svm {

        enum class KernelType {LINEAR, POLY, RBF, PRECOMPUTED};

        //! Kernel
        //! TODO: Describe me
        template <KernelType kernel_type, typename kernel_t>
        class Kernel {
        private:
            //! Kernel constructor
            Kernel();
        public:
            inline kernel_t evaluate_kernel(int i, int j) const {
                throw std::logic_error("Invalid KernelType");
            }
        }; // Kernel


        //! Kernel<KernelType::LINEAR, kernel_t>
        //! TODO: Describe me
        template <typename kernel_t>
        class Kernel <KernelType::LINEAR, kernel_t> {
            const arma::Mat<kernel_t>& x;
            const arma::Col<int>& y;
        public:
            Kernel(arma::Mat<kernel_t>& x_, arma::Col<int>& y_) : x(x_), y(y_) {
                assert(x.n_rows == y.n_rows);
            }

            inline kernel_t evaluate_kernel(int i, int j) const {
                return arma::dot(this->x.col(i), this->x.col(j));
            }
        }; // Kernel<KernelType::LINEAR, kernel_t>


        //! Kernel<KernelType::POLY, kernel_t>
        //! TODO: Describe me
        template <typename kernel_t>
        class Kernel <KernelType::POLY, kernel_t> {
            const arma::Mat<kernel_t>& x;
            const arma::Col<int>& y;
            const int degree;
            const double gamma;
            const double coef0;
        public:
            Kernel(arma::Mat<kernel_t>& x_, arma::Col<int>& y_, int degree_, double gamma_, double coef0_)
                : x(x_), y(y_), degree(degree_), gamma(gamma_), coef0(coef0_) {
            }

            inline kernel_t evaluate_kernel(int i, int j) const {
                return pow(gamma * arma::dot(x.col(i), x.col(j)) + coef0, degree);
            }
        }; // Kernel<KernelType::POLY, kernel_t>


        //! Kernel<KernelType::RBF, kernel_t>
        //! TODO: Describe me
        template <typename kernel_t>
        class Kernel <KernelType::RBF, kernel_t> {
            const arma::Mat<kernel_t>& x;
            const arma::Col<int>& y;
            const double gamma;
            arma::Col<kernel_t> x_square;
        public:
            Kernel(arma::Mat<kernel_t>& x_, arma::Col<int>& y_, double gamma_)
                : x(x_), y(y_), gamma(gamma_), x_square(x_.n_rows) {
                for (int i = 0; i < x.n_rows; i++) {
                    x_square[i] = arma::dot(x.col(i), x.col(i));
                }
            }

            inline kernel_t evaluate_kernel(int i, int j) const {
                return exp(
                        - gamma * (
                            x_square[i] + x_square[j] - 2 * arma::dot(x.col(i), x.col(j))
                            )
                        );
            }
        }; // Kernel<KernelType::RBF, kernel_t>


        //! Kernel<KernelType::PRECOMPUTED, kernel_t>
        //! TODO: Describe me
        template <typename kernel_t>
        class Kernel <KernelType::PRECOMPUTED, kernel_t> {
        public:
            const arma::Mat<kernel_t>& precomputed_kernel;
            Kernel(arma::Mat<kernel_t>& kernel)
                : precomputed_kernel(kernel) {
            }

            inline kernel_t evaluate_kernel(int i, int j) const {
                return precomputed_kernel(i,j);
            }
        }; // Kernel<KernelType::PRECOMPUTED, kernel_t>

    } // juml::svm
}  // juml
#endif // JUML_SVM_KERNEL_H_
