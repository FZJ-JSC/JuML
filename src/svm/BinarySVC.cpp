// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BinarySVC.cpp
*
* Description: Implementation of class BinarySVC, which implements a BaseClassifier for
* Binary Classification Problems using a Support Vector Machine.
*
* Maintainer:
*
* Email:
*/

#include "svm/BinarySVC.h"
#include "svm/Solver.h"
#include "svm/KernelCache.h"

namespace juml {
    namespace svm {



        void BinarySVC::fit(Dataset<float>& X, Dataset<int>& y) {
            this->class_normalizer_.index(y);
            if (this->class_normalizer_.n_classes() != 2) {
                throw std::invalid_argument("BinarySVC only supports binary classification problems");
            }

            //TODO: All training data needs to be loaded on all processes!
            const arma::Mat<float> Xdata = X.data();
            const arma::Mat<int> Ydata = y.data();

            if (Xdata.n_cols != Ydata.n_cols * Ydata.n_rows) {
                throw std::invalid_argument("X needs to have as many columns as Y has entries!");
            }

            //Inspired by libsvm's `solve_c_svc`-Function. See LIBSVM-COPYRIGHT.txt

            int l = Xdata.n_cols;
            arma::Col<double> minus_ones(l);
            arma::Col<double> alpha = arma::Col<double>(l, arma::fill::zeros);

            arma::Col<int> yVector = arma::vectorise(Ydata);
            std::vector<BinaryLabel> y_(l);
            for (int i = 0; i < l; i++) {
                minus_ones(i) = -1;
                y_.at(i) = this->class_normalizer_.transform(yVector(i)) == 0 ? BinaryLabel::POSITIVE : BinaryLabel::NEGATIVE;
            }

            QMatrix *Q;

            if (this->kernelType_ == KernelType::LINEAR) {
                Kernel<KernelType::LINEAR> kernel(Xdata);
                Q = new KernelCache<Kernel<KernelType::LINEAR>>(kernel, this->cache_size_, l);
            } else if (this->kernelType_ == KernelType::POLY) {
                Kernel<KernelType::POLY> kernel(Xdata, this->degree_, this->gamma_, this->coef0_);
                Q = new KernelCache<Kernel<KernelType::POLY>>(kernel, this->cache_size_, l);
            } else if (this->kernelType_ == KernelType::RBF) {
                Kernel<KernelType::RBF> kernel(Xdata, this->gamma_);
                Q = new KernelCache<Kernel<KernelType::RBF>>(kernel, this->cache_size_, l);
            }
            //XXX: Is 'kernel' still availble for Q or is it deconstructed?
            //Probably need to create it with 'new' and delete it later.

            //TODO: epsilon parameter needs to be set using constructor!
            SMOSolver s;
            s.Solve(l, *Q, minus_ones, y_, this->weight_positive_, this->weight_negative_, 1e-3,
                    alpha, &this->rho, &this->obj_value);

            double sum_alpha = arma::sum(alpha);
            std::vector<arma::uword> sv_indices;
            for (int i = 0; i < l; i++) {
                if (fabs(alpha(i)) > 0) {
                    sv_indices.push_back(i);
                }
                alpha(i) *= (int)y_.at(i);
            }
            this->n_support = sv_indices.size();
            this->support = arma::uvec(sv_indices);
            this->support_vectors = Xdata.cols(this->support);
            this->support_coefs = alpha.elem(this->support);

            delete Q;
        }

        Dataset<int> BinarySVC::predict(const Dataset<float>& X) const {
            const arma::Mat<float> &data = X.data();
            arma::Mat<int> results = arma::Mat<int>(1, data.n_cols);
            for (int i = 0; i < data.n_cols; i++) {
                BinaryLabel label = this->predict_single(data.col(i));
                results(i) = this->class_normalizer_.invert(label == BinaryLabel::POSITIVE ? 0 : 1);
            }
            Dataset<int> preds(results);
            return preds;
        }

        BinaryLabel BinarySVC::predict_single(const arma::subview_col<float>& x) const {
            arma::Col<double> kvalue(this->n_support);
#define EVALKERNELCASE(kType) case kType:\
            v = evaluate_kernel<kType>(this->support_vectors.col(i), x, this->degree_, this->gamma_, this->coef0_);break
            for (int i = 0; i < this->n_support; i++) {
                double v;
                switch(this->kernelType_) {
                    EVALKERNELCASE(KernelType::LINEAR);
                    EVALKERNELCASE(KernelType::POLY);
                    EVALKERNELCASE(KernelType::RBF);
                    default: throw std::logic_error("Unsupported kernel type");
                }
                kvalue(i) = v;
            }
#undef EVALKERNELCASE

            //TODO implement prediction based on kvalue
            return BinaryLabel::POSITIVE;
        }

        BinarySVC::~BinarySVC() {
        }
    } // svm

} // juml
