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

#include "Kernel.h"

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
	} // svm
} // juml

#endif // JUML_SVM_SVCKERNEL_H_
