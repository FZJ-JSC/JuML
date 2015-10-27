// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Solver.cpp
*
* Description: Implementation of class Solver
*
* Maintainer:
*
* Email:
*/

#include "svm/Solver.h"

namespace juml {
    namespace svm {
        void SMOSolver::Solve(int l, QMatrix& Q, const arma::Col<double>& p, const std::vector<BinaryLabel>& y,
                    arma::Col<double>& alpha, double weight_positive, double weight_negative, double eps) const {
        }

    }
} // namespace juml
