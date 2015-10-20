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

namespace juml {
	namespace svm {
		void BinarySVC::fit(Dataset<float>& X, Dataset<int>& y) {
			this->class_normalizer_.index(y);
			if (this->class_normalizer_.n_classes() != 2) {
				throw std::runtime_error("BinarySVC only supports binary classification problems");
			}
                        //TODO implement fit
		}
		Dataset<int> BinarySVC::predict(const Dataset<float>& X) const {
                    //TODO implement predict
		}
	} // svm
} // juml
