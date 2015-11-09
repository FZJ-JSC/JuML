// vim: expandtab:shiftwidth=4:softtabstop=4
/*
* Copyright (c) 2015
* Forschungszentrum Juelich GmbH, Juelich Supercomputing Center
*
* This software may be modified and distributed under the terms of BSD-style license.
*
* File name: Solver.cpp
*
* Description: Implementation of class SMOSolver
*
* Maintainer:
*
* Email:
*/

#include "svm/Solver.h"

namespace juml {
    namespace svm {
        const double INF = HUGE_VAL;
        const double TAU = 1e12;

        enum class ALPHA_STATUS {UPPER_BOUND, LOWER_BOUND, FREE};


        //SMO-Solver from LibSVM. Modified to not use shrinking and use our own data structures.
        //See LIBSVM-COPYRIGHT.txt

        inline double get_C(BinaryLabel y, double Cpositive, double Cnegative) {
            return y == BinaryLabel::POSITIVE ? Cpositive : Cnegative;
        }
        inline ALPHA_STATUS get_alpha_status(double a, double C) {
            if (a >= C) {
                return ALPHA_STATUS::UPPER_BOUND;
            } else if (a <= 0) {
                return ALPHA_STATUS::LOWER_BOUND;
            } else {
                return ALPHA_STATUS::FREE;
            }
        }
        inline bool is_lower_bound(ALPHA_STATUS status) {
            return status == ALPHA_STATUS::LOWER_BOUND;
        }
        inline bool is_upper_bound(ALPHA_STATUS status) {
            return status == ALPHA_STATUS::UPPER_BOUND;
        }
        inline bool is_free(ALPHA_STATUS status) {
            return status == ALPHA_STATUS::FREE;
        }


        double calculate_rho(const std::vector<BinaryLabel>& y, const arma::Col<double> &G, const ALPHA_STATUS* alpha_status) {
            double r;
            int nr_free = 0;
            double ub = INF, lb = -INF, sum_free = 0;
            int l = G.n_rows;

            for (int i = 0; i < l; i++) {
                double yG = ((int)y[i]) * G(i);
                if (is_upper_bound(alpha_status[i])) {
                    if (y[i] == BinaryLabel::NEGATIVE) {
                        ub = std::min(ub, yG);
                    } else {
                        lb = std::max(lb, yG);
                    }
                } else if (is_lower_bound(alpha_status[i])) {
                    if (y[i] == BinaryLabel::POSITIVE) {
                        ub = std::min(ub, yG);
                    } else {
                        lb = std::max(lb, yG);
                    }
                } else {
                    ++nr_free;
                    sum_free += yG;
                }
            }
            
            if (nr_free > 0) {
                r = sum_free / nr_free;
            } else {
                r = (ub + lb) / 2;
            }
            return r;
        }

        // Return true if already optimal, return 0 otherwise
        bool select_working_set(QMatrix &Q, const std::vector<BinaryLabel> y, const arma::Col<double> &G,
                const arma::Col<kernel_t> &QD, const ALPHA_STATUS* alpha_status, double eps, int* out_i, int* out_j) {
            double Gmax = -INF;
            double Gmax2 = -INF;
            int Gmax_idx = -1;
            int Gmin_idx = -1;
            double obj_diff_min = INF;

            int l = G.n_rows;

            for (int t = 0; t < l; t++) {
                if (y[t] == BinaryLabel::POSITIVE) {
                    if (!is_upper_bound(alpha_status[t])) {
                        if (-G[t] >= Gmax) {
                            Gmax = -G[t];
                            Gmax_idx = t;
                        }
                    }
                } else {
                    //y[t] == BinaryLabel::NEGATIVE
                    if (!is_lower_bound(alpha_status[t])) {
                        if (G[t] >= Gmax) {
                            Gmax = G[t];
                            Gmax_idx = t;
                        }
                    }
                }
            }

            int i = Gmax_idx;
            const kernel_t *Q_i = NULL;
            if (i != -1) {
                Q_i = Q.get_col(i);
            }
            //If i == -1: Gmax = -INF and Q_i will not be accessed by the following loop

            for (int j = 0; j < l; j++) {
                if (y[j] == BinaryLabel::POSITIVE) {
                    if (!is_lower_bound(alpha_status[j])) {
                        double grad_diff = Gmax + G(j);
                        if (G(j) >= Gmax2) {
                            Gmax2 = G(j);
                        }
                        if (grad_diff > 0) {
                            double obj_diff;
                            double quad_coef = QD(i) + QD(j) - 2.0 * ((int)y[i]) * Q_i[j];
                            if (quad_coef > 0) {
                                obj_diff = - (grad_diff * grad_diff) / quad_coef;
                            } else {
                                obj_diff = - (grad_diff * grad_diff) / TAU;
                            }

                            if (obj_diff <= obj_diff_min) {
                                Gmin_idx = j;
                                obj_diff_min = obj_diff;
                            }
                        }
                    }
                } else {
                    //y[j] == BinaryLabel::NEGATIVE
                    if (!is_upper_bound(alpha_status[j])) {
                        double grad_diff = Gmax-G[j];
                        if (-G[j] >= Gmax2) {
                            Gmax2 = -G[j];
                        }
                        if (grad_diff > 0) {
                            double obj_diff;
                            double quad_coef = QD(i) + QD(j) + 2.0 * ((int)y[i]) * Q_i[j];
                            if (quad_coef > 0) {
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            } else {
                                obj_diff = - (grad_diff * grad_diff) / TAU;
                            }

                            if (obj_diff <= obj_diff_min) {
                                Gmin_idx = j;
                                obj_diff_min = obj_diff;
                            }
                        }
                    }
                }
            } // end for loop

            if (Gmax + Gmax2 < eps) {
                return true;
            }

            *out_i = Gmax_idx;
            *out_j = Gmin_idx;
            return false;
        }

        //! SMO Solver inspired by LibSVM's Solver::Solve
        void SMOSolver::Solve(int l, QMatrix& Q, const arma::Col<double>& p, const std::vector<BinaryLabel>& y,
                     double Cpositive, double Cnegative, double eps,
                     arma::Col<double>& alpha, double *rho, double *obj_value) const {
            arma::Col<kernel_t> QD(l);
            //Calculate Diagonal of kernel matrix
            for (int i = 0; i < l; i++) {
                QD(i) = Q.evaluate_kernel(i,i);
            }

            //LibSVMs Solve-Function copys p, y and alpha, so it can swap indices. We don't do that.

            ALPHA_STATUS *alpha_status = new ALPHA_STATUS[l];

            for (int i = 0; i < l; i++) {
                //update_alpha_status
                alpha_status[i] = get_alpha_status(alpha(i), get_C(y[i], Cpositive, Cnegative));
            }

            //We do not need to keep track of the 'active_set' because we don't do shrinking.

            //Initialize the gradient
            arma::Col<double> G(l);
            //Gradient, if we threat free variables as 0 ... Only used when shrinking
            //arma::Col<double> G_bar(l, arma::fill::zeros);
            for (int i = 0; i < l; i++) {
                G(i) = p(i);
            }

            for (int i = 0; i < l; i++) {
                if (!is_lower_bound(alpha_status[i])) {
                    const kernel_t *Q_i = Q.get_col(i);
                    double alpha_i = alpha(i);
                    for (int j = 0; j < l; j++) {
                        G(j) += alpha_i * Q_i[j];
                    }
                    //G_bar is unused
                    /*if (is_upper_bound(alpha_status[i]) {
                        for (int j = 0; j < l; j++) {
                            G_bar(j) += get_C(y[i], Cpositive, Cnegative) * Q_i[j];
                        }
                    }*/
                }
            }


            //optimization step
            int iter = 0;
            int max_iter = std::max(10000000, l > INT_MAX/100 ? INT_MAX : 100 * l);

            while (iter < max_iter) {
                int i, j;
                if (select_working_set(Q, y, G, QD, alpha_status, eps, &i, &j)) {
                    break;
                }
                iter++;

                const kernel_t *Q_i = Q.get_col(i);
                const kernel_t *Q_j = Q.get_col(j);

                double C_i = get_C(y[i], Cpositive, Cnegative);
                double C_j = get_C(y[j], Cpositive, Cnegative);

                double old_alpha_i = alpha(i);
                double old_alpha_j = alpha(j);

                if (y[i] != y[j]) {
                    double quad_coef = QD[i] + QD[j] + 2 * Q_i[j];
                    if (quad_coef <= 0) {
                        quad_coef = TAU;
                    }
                    double delta = (-G(i) - G(j)) / quad_coef;
                    double diff = alpha(i) - alpha(j);
                    alpha(i) += delta;
                    alpha(j) += delta;

                    if (diff > 0) {
                        if (alpha(j) < 0) {
                            alpha(j) = 0;
                            alpha(i) = diff;
                        }
                    } else {
                        if (alpha(i) < 0) {
                            alpha(i) = 0;
                            alpha(j) = -diff;
                        }
                    }
                    if (diff > C_i - C_j) {
                        if (alpha(i) > C_i) {
                            alpha(i) = C_i;
                            alpha(j) = C_i - diff;
                        }
                    } else {
                        if (alpha(j) > C_j) {
                            alpha(j) = C_j;
                            alpha(i) = C_j + diff;
                        }
                    }
                } else { //y[i] == y[j]
                    double quad_coef = QD[i] + QD[j] - 2 * Q_i[j];
                    if (quad_coef <= 0) {
                        quad_coef = TAU;
                    }
                    double delta = (G[i] - G[j]) / quad_coef;
                    double sum = alpha[i] + alpha[j];

                    alpha(i) -= delta;
                    alpha(j) += delta;

                    if (sum > C_i) {
                        if(alpha(i) > C_i) {
                            alpha(i) = C_i;
                            alpha(j) = sum - C_i;
                        }
                    } else {
                        if (alpha(j) < 0) {
                            alpha(j) = 0;
                            alpha(i) = sum;
                        }
                    }
                    if (sum > C_j) {
                        if (alpha(j) > C_j) {
                            alpha(j) = C_j;
                            alpha(i) = sum - C_j;
                        }
                    } else {
                        if (alpha(i) < 0) {
                            alpha(i) = 0;
                            alpha(j) = sum;
                        }
                    }
                }


                // update G

                double delta_alpha_i = alpha(i) - old_alpha_i;
                double delta_alpha_j = alpha(j) - old_alpha_j;

                for (int k = 0; k < l; k++) {
                    G(k) += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
                }

                // update alpha status and G_bar
                bool was_upper_bound_i = is_upper_bound(alpha_status[i]);
                bool was_upper_bound_j = is_upper_bound(alpha_status[j]);

                //update_alpha_status
                alpha_status[i] = get_alpha_status(alpha[i], get_C(y[i], Cpositive, Cnegative));
                alpha_status[j] = get_alpha_status(alpha[j], get_C(y[j], Cpositive, Cnegative));

                //G_bar is unused
                /*if (was_upper_bound_i != is_upper_bound(alpha_status[i])) {
                    if (was_upper_bound_i) {
                        for (int k=0; k <l; k++) {
                            G_bar(k) -= C_i * Q_i[k];
                        }
                    } else {
                        for (int k = 0; k < l; k++) {
                            G_bar[k] += C_i * Q_i[k];
                        }
                    }
                }

                if (was_upper_bound_j != is_upper_bound(alpha_status[j])) {
                    if (was_upper_bound_i) {
                        for (int k = 0; k < l; k++) {
                            G_bar(k) -= C_j * Q_j[k];
                        }
                    } else {
                        for (int k = 0; k < l; k++) {
                            G_bar(k) += C_j * Q_j[k];
                        }
                    }
                }*/

            } // end of iteration loop

            if (iter >= max_iter) {
                //TODO Warning
            }

            *rho = calculate_rho(y, G, alpha_status);

            //calculate obj value
            double v = 0;
            for (int i = 0; i < l; i++) {
                v += alpha(i) * (G(i) + p(i));
            }
            *obj_value =  v / 2;

            delete[] alpha_status;
        }
    }
} // namespace juml
