// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: BinarySVC.h
*
* Description: Header of class Kernel
*
* Maintainer:
*
* Email:
*/

#ifndef JUML_SVM_SVCKERNEL_H_
#define JUML_SVM_SVCKERNEL_H_

#include <mpi.h>
#include "Kernel.h"
#include "classification/BaseClassifier.h"

namespace juml {
	namespace svm {
		//TODO move to appropiate header file
		enum class BinaryLabel {POSITIVE = 1, NEGATIVE = -1};
		inline int operator*(BinaryLabel a, BinaryLabel b) {
			return (int)a * (int)b;
		}

		template <class InnerKernel>
		class SVCKernel {
			InnerKernel &inner;
			std::vector<BinaryLabel> &labels;
			typedef decltype(inner.evaluate_kernel(0,0)) kernel_t;
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
                public:
                    BinarySVC(
                            double C = 1.0, KernelType kernel = KernelType::RBF,
                            double degree = 3, double gamma = 0.0, double coef0 =0.0,
                            size_t cache_size = 200*(2<<10),
                            double weight_positive = NAN, double weight_negative = NAN,
                            MPI_Comm comm = MPI_COMM_WORLD)
                   : C_(C), degree_(degree), gamma_(gamma), coef0_(coef0),
                     cache_size_(cache_size),
                     weight_positive_(weight_positive == NAN ? C : weight_positive),
                     weight_negative_(weight_negative == NAN ? C : weight_negative),
                     kernelType_(kernel), BaseClassifier(comm) {

                    }

                    virtual void fit(Dataset<float>& X, Dataset<int>& y);
                    virtual Dataset<int> predict(const Dataset<float>& X);
                    virtual float accuracy(const Dataset<float>& X, const Dataset<int>& y) {
                        //TODO implement in BaseClassifier using predict?
                        return 0;
                    }

                };
	} // svm
} // juml

#endif // JUML_SVM_SVCKERNEL_H_
