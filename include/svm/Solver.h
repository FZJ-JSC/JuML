// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Solver.h
*
* Description: Header of interface Solver and class SMOSolver
*
* Maintainer:
*
* Email:
*/

#ifndef JUML_SVM_SOLVER_H_
#define JUML_SVM_SOLVER_H_

#include <armadillo>
#include "QMatrix.h"

namespace juml {
    namespace svm {
        enum class BinaryLabel {POSITIVE = 1, NEGATIVE = -1};
        inline int operator*(BinaryLabel a, BinaryLabel b) {
            return (int)a * (int)b;
        }


        //! Solver for the SVM-Problem
        class Solver {
        public:
            //! Solve the SVM-Problem.
            //!
            //! \param[in] l size of Q, p, y, alpha
            //! \param[in] Q The Kernel Matrix. Might be cached.
            //! \param[in] p
            //! \param[in] y Specifies if the i'th sample is a positive or negative sample
            //! \param[out] alpha the resulting alpha values will be stored there
            //! \param[in] weight_positive weight for positive samples
            //! \param[in] weight_negative weight for negative samples
            //! \param[in] eps stopping tolerance
            virtual void Solve(int l, QMatrix& Q, const arma::Col<double>& p, const std::vector<BinaryLabel>& y,
                    arma::Col<double>& alpha, double weight_positive, double weight_negative, double eps) const = 0;

            virtual ~Solver() {
            }
        }; // Solver

        //! Solver for SVM using a simplified version of the LibSVM Solver

        //! See LIBSVM-COPYRIGHT.txt
        class SMOSolver : Solver {
            public:
            virtual void Solve(int l, QMatrix& Q, const arma::Col<double>& p, const std::vector<BinaryLabel>& y,
                    arma::Col<double>& alpha, double weight_positive, double weight_negative, double eps) const override;
        }; // SMOSolver
    } // svm
}  // juml
#endif // JUML_SVM_SOLVER_H_
