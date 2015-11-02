// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BinarySVC.h
*
* Description: Header of class BinarySVC
*
* Maintainer:
*
* Email:
*/

#ifndef JUML_SVM_SVCKERNEL_H_
#define JUML_SVM_SVCKERNEL_H_

#include <mpi.h>
//TODO We don't want to include Kernel.h here. It probably should not be part of the public API
#include "Kernel.h"
#include "Solver.h"
#include "classification/BaseClassifier.h"

namespace juml {
	namespace svm {
		template <class InnerKernel>
		class SVCKernel {
			InnerKernel &inner;
			std::vector<BinaryLabel> &labels;
			public:
			SVCKernel(InnerKernel& inner_, std::vector<BinaryLabel>& labels_)
			       	: inner(inner_), labels(labels_) {
			}

			inline kernel_t evaluate_kernel(int i, int j) {
				return labels[i] * labels[j] * inner.evaluate_kernel(i, j);
			}
		}; // SVCKernel

                //! TODO: Describe Me
                class BinarySVC : public BaseClassifier {
                protected:
                    const double C_, degree_, gamma_, coef0_;
                    const size_t cache_size_;
                    const KernelType kernelType_;
                    const double weight_positive_, weight_negative_;

                    BinaryLabel predict_single(const arma::subview_col<float>& col) const;

                public:
                    BinarySVC(
                            double C = 1.0, KernelType kernel = KernelType::RBF,
                            double degree = 3, double gamma = 0.0, double coef0 =0.0,
                            size_t cache_size = 200*(2<<10),
                            double weight_positive = NAN, double weight_negative = NAN,
                            MPI_Comm comm = MPI_COMM_WORLD)
                        : C_(C), degree_(degree), gamma_(gamma), coef0_(coef0),
                     cache_size_(cache_size),
                     //TODO Specify weights as mutliplies of C?
                     weight_positive_(weight_positive == NAN ? C : weight_positive),
                     weight_negative_(weight_negative == NAN ? C : weight_negative),
                     kernelType_(kernel), BaseClassifier(comm) {

                    }

                    virtual void fit(Dataset<float>& X, Dataset<int>& y) override;
                    virtual Dataset<int> predict(const Dataset<float>& X) const override;
                    virtual float accuracy(const Dataset<float>& X, const Dataset<int>& y) const override {
                        //TODO implement in BaseClassifier using predict?
                        return 0;
                    }

                    //! Number of Support Vectors
                    int n_support = 0;
                    //! Indices of support Vectors
                    arma::uvec support;
                    //! Support Vectors
                    arma::Mat<kernel_t> support_vectors;
                    
                    arma::Col<double> support_coefs;

                    double rho, obj_value;

                    virtual ~BinarySVC();

                }; // BinarySVC
	} // svm
} // juml

#endif // JUML_SVM_SVCKERNEL_H_
