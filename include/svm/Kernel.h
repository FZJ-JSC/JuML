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
#include "QMatrix.h"

namespace juml {

    namespace svm {

        enum class KernelType {LINEAR, POLY, RBF, PRECOMPUTED};
        
        template <KernelType kernel_type>
        kernel_t evaluate_kernel(const arma::subview_col<kernel_t>& a, const arma::subview_col<kernel_t>& b, double degree, double gamma, double coef0) {
            throw std::logic_error("Invalid Kerneltype");
        }

        template <>
        inline kernel_t evaluate_kernel <KernelType::LINEAR>(const arma::subview_col<kernel_t>& a, const arma::subview_col<kernel_t>& b, double degree, double gamma, double coef0) {
            return arma::dot(a, b);
        }

        template <>
        inline kernel_t evaluate_kernel <KernelType::POLY>(const arma::subview_col<kernel_t>& a, const arma::subview_col<kernel_t>& b, double degree, double gamma, double coef0) {
            return  pow(gamma * arma::dot(a, b) + coef0, degree);
        }

         template <>
         inline kernel_t evaluate_kernel <KernelType::RBF>(const arma::subview_col<kernel_t>& a, const arma::subview_col<kernel_t>& b, double degree, double gamma, double coef0) {
            return exp(
                        - gamma * (
                            arma::dot(a,a) + arma::dot(b,b) - 2 * arma::dot(a, b)
                            )
                        );

        }





        //! Kernel
        //! TODO: Describe me
        template <KernelType kernel_type>
        class Kernel {
        private:
            //! Kernel constructor
            Kernel();
        public:
            inline kernel_t evaluate_kernel(int i, int j) const {
                throw std::logic_error("Invalid KernelType");
            }
        }; // Kernel


        //! Kernel<KernelType::LINEAR>
        //! TODO: Describe me
        template <>
        class Kernel <KernelType::LINEAR> {
            const arma::Mat<kernel_t>& x;
        public:
            Kernel(const arma::Mat<kernel_t>& x_) : x(x_) {
            }

            inline kernel_t evaluate_kernel(int i, int j) const {
                return arma::dot(this->x.col(i), this->x.col(j));
            }
        }; // Kernel<KernelType::LINEAR>


        //! Kernel<KernelType::POLY>
        //! TODO: Describe me
        template <>
        class Kernel <KernelType::POLY> {
            const arma::Mat<kernel_t>& x_;
            const int degree_;
            const double gamma_;
            const double coef0_;
        public:
            Kernel(const arma::Mat<kernel_t>& x, int degree, double gamma, double coef0)
                : x_(x), degree_(degree), gamma_(gamma), coef0_(coef0) {
            }

            inline kernel_t evaluate_kernel(int i, int j) const {
                return pow(gamma_ * arma::dot(x_.col(i), x_.col(j)) + coef0_, degree_);
            }
        }; // Kernel<KernelType::POLY>


        //! Kernel<KernelType::RBF>
        //! TODO: Describe me
        template <>
        class Kernel <KernelType::RBF> {
            const arma::Mat<kernel_t>& x;
            const double gamma;
            arma::Col<kernel_t> x_square;
        public:
            Kernel(const arma::Mat<kernel_t>& x_, double gamma_)
                : x(x_), gamma(isnan(gamma_) ? 1.0/x_.n_rows : gamma_), x_square(x_.n_cols) {
                for (int i = 0; i < x.n_cols; i++) {
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
        }; // Kernel<KernelType::RBF>


        //! Kernel<KernelType::PRECOMPUTED>
        //! TODO: Describe me
        template <>
        class Kernel <KernelType::PRECOMPUTED> {
        public:
            const arma::Mat<kernel_t>& precomputed_kernel;
            Kernel(const arma::Mat<kernel_t>& kernel)
                : precomputed_kernel(kernel) {
            }

            inline kernel_t evaluate_kernel(int i, int j) const {
                return precomputed_kernel(i,j);
            }
        }; // Kernel<KernelType::PRECOMPUTED>

    } // juml::svm
}  // juml
#endif // JUML_SVM_KERNEL_H_
