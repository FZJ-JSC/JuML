// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: QMatrix.h
*
* Description: Header of class QMatrix
*
* Maintainer:
*
* Email:
*/

#ifndef JUML_SVM_QMATRIX_H_
#define JUML_SVM_QMATRIX_H_

namespace juml {
    namespace svm {
        typedef float kernel_t;

        class QMatrix {
            public: 
            virtual const kernel_t *get_col(int col) = 0;
            virtual const kernel_t* get_col(int col, std::vector<unsigned int>) = 0;
            virtual bool is_cached(int col) = 0;
            virtual kernel_t evaluate_kernel(int i, int j) = 0;
            virtual ~QMatrix() {
            }
        };
    }
}

#endif // JUML_SVM_QMATRIX_H_
