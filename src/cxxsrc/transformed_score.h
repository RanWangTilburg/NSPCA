#pragma once

#include <cstdlib>
#include <cmath>
#include "../custat/cuStat.h"
#include "util.h"
#include "Constants.h"

using cuStat::ViewXd;
namespace NSPCA {
    namespace internal {
        struct sv {
            double a0;
            double a1;
            double a2;
            double v;

            sv(double a0, double a1, double a2, double v) : a0(a0), a1(a1), a2(a2), v(v) {}

            sv() : a0(0.0), a1(0.0), a2(0.0), v(0.0) {}
        };

        sv
        get_sol_from_a2(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper,
                        double a2);

        sv get_sol_from_a2_upper(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                                 double upper, double a2);

        sv &max_sv(sv &x, sv &b);

        double a2_1(double sqrt_n, size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                    double upper);

        double a2_2(double sqrt_n, size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                    double upper);

        double a2_3(double sqrt_n, size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                    double upper);

        double a2_4(double sqrt_n, size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                    double upper);

        sv
        solve_transformed_score_j(ViewXi& incidence_count, ViewXd& incident_count_score, double scale, Dim &dim,
                                  double square_n, size_t col);

        struct solve_transformed_score_cols {
            ViewXi incidence_count_view;
            ViewXd incidence_count_score_view;
            ViewXd transformed_score_solution_view;
            double scale;
            double squared_n;
            Dim & dim_ref;
            int col_start;
            int col_end;

            solve_transformed_score_cols(int *incidence_count, double *incident_count_score,
                                         double *transformed_score_solution,
                                         double scale, double square_n, int col_start, int col_end, Dim &dim)
                    : incidence_count_view(incidence_count, 3, dim._nVar),
                      incidence_count_score_view(incident_count_score, 3, dim._nVar),
                      transformed_score_solution_view(transformed_score_solution, 3, dim._nVar),
                      scale(scale),
                      squared_n(square_n),
                      dim_ref(dim),
                      col_start(col_start),
                      col_end(col_end) {}

            void operator()() {
                for (int col = col_start; col <col_end; col++) {
//                    sv
//                    solve_transformed_score_j(ViewXi& incidence_count, ViewXd& incident_count_score, double scale, dim &dim,
//                                              double square_n, size_t col);
                    sv result = solve_transformed_score_j(incidence_count_view, incidence_count_score_view, scale,
                                                          dim_ref, squared_n, col);
                    transformed_score_solution_view(0, col) = result.a0;
                    transformed_score_solution_view(1, col) = result.a1;
                    transformed_score_solution_view(2, col) = result.a2 + result.a0;

                }
                return;
            }
        };


        sv
        solve_transformed_score_j(ViewXi &incidence_count_view, ViewXd &incidence_count_score_view, double scale, Dim &dim,
                                  double square_n, size_t col) {
//            ViewXii incidence_count_view = ViewXi(incidence_count, 3, dim.no_var);
//            ViewXd incidence_count_score_view = ViewXd(incident_count_score, 3, dim.no_var);
            size_t n = incidence_count_view(0, col);
            size_t e = incidence_count_view(1, col);
            size_t p = incidence_count_view(2, col);

            double tn = incidence_count_score_view(0, col);
            double te = incidence_count_score_view(1, col);
            double tp = incidence_count_score_view(2, col);

            double s = scale;
            size_t N = dim._nObs;
            double sqrt_n = square_n;
            double upper = sqrt((n + p) * N * s * s / (n * p));

            double a2 = 0.0;
            sv sol1 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2);
            a2 = upper;
            sv sol2 = get_sol_from_a2_upper(N, n, e, p, tn, te, tp, s, upper, a2);
            sol1 = max_sv(sol1, sol2);

            a2 = a2_1(sqrt_n, N, n, e, p, tn, te, tp, s, upper);
            if (a2 != 0) {
                sol2 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2);
                sol1 = max_sv(sol1, sol2);
            }

            a2 = a2_2(sqrt_n, N, n, e, p, tn, te, tp, s, upper);
            if (a2 != 0) {
                sol2 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2);
                sol1 = max_sv(sol1, sol2);
            }
            a2 = a2_3(sqrt_n, N, n, e, p, tn, te, tp, s, upper);
            if (a2 != 0) {
                sol2 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2);
                sol1 = max_sv(sol1, sol2);
            }
            a2 = a2_4(sqrt_n, N, n, e, p, tn, te, tp, s, upper);
            if (a2 != 0) {
                sol2 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2);
                sol1 = max_sv(sol1, sol2);
            }

            return sol1;
        }

        sv &max_sv(sv &x, sv &y) {
            if (x.v > y.v) {
                return x;
            } else {
                return y;
            }
        }

        sv
        get_sol_from_a2(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper,
                        double a2) {
            sv result = sv(0.0, 0.0, a2, 0.0);
            double temp = -a2 * a2 * n * p + (n + p) * N * s * s;
            double L = sqrt((temp) / (e * N));
            double a0 = (-a2 * p + e * L) / (n + p);
            double a1 = -L;
            double fvalue = tn * a0 + te * a1 + (a0 + a2) * tp;
            double a0other = -(a2 * p + e * L) / (n + p);
            double a1other = L;
            double fvalue2 = tn * a0 + te * a1 + (a0 + a2) * tp;

            if (fvalue > fvalue2) {
                result.a0 = a0;
                result.a1 = a1;
                result.v = fvalue;
            } else {
                result.a0 = a0other;
                result.a1 = a1other;
                result.v = fvalue2;
            }
            return result;

        }

        sv get_sol_from_a2_upper(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                                 double upper, double a2) {
            auto result = sv(0.0, 0.0, a2, 0.0);
            double a0 = (-a2 * p) / (n + p);
            double a1 = 0;
            double fvalue = tn * a0 + (a0 + a2) * tp;
            result.a0 = a0;
            result.a1 = a1;
            result.v = fvalue;
            return result;
        }

        double a2_1(double sqrt_n, size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                    double upper) {
            double a2 = 0.0;
            double upperside = (e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp +
                                e * e * n * p * tp * tp + 2 * e * n * n * p * te * tn + 2 * e * n * n * p * te * tp +
                                N * e * n * n * tp * tp + 2 * e * n * p * p * te * tn +
                                2 * e * n * p * p * te * tp - 2 * N * e * n * p * tn * tp + N * e * p * p * tn * tn +
                                n * n * n * p * te * te + 2 * n * n * p * p * te * te + n * p * p * p * te * te);
            if (upperside <= 0) {
                a2 = 0.0;
            } else {
                double temp = (sqrt_n * s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                             2 * e * e * n * p * tn *
                                                                                             tp +
                                                                                             e * e * n * p * tp * tp +
                                                                                             2 * e * n * n * p * te *
                                                                                             tn +
                                                                                             2 * e * n * n * p * te *
                                                                                             tp +
                                                                                             N * e * n * n * tp * tp +
                                                                                             2 * e * n * p * p * te *
                                                                                             tn +
                                                                                             2 * e * n * p * p * te *
                                                                                             tp - 2 *
                                                                                                  N *
                                                                                                  e *
                                                                                                  n *
                                                                                                  p *
                                                                                                  tn *
                                                                                                  tp +
                                                                                             N * e * p * p * tn * tn +
                                                                                             n * n * n * p * te * te +
                                                                                             2 * n * n * p * p * te *
                                                                                             te +
                                                                                             n * p * p * p * te * te));
                if (temp < 0 || temp > upper) a2 = 0.0;
                else a2 = temp;
            }

            return a2;
        }

        double a2_2(double sqrt_n, size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                    double upper) {
            double a2 = 0.0;
            double upperside = N * e * n * p * (n + p) * (
                    e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp +
                    2 * e * n * n * p * te * tn + 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp +
                    2 * e * n * p * p * te * tn + 2 * e * n * p * p * te * tp - 2 *
                                                                                N * e * n * p * tn * tp +
                    N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
                    n * p * p * p * te * te);
            if (upperside <= 0) {
                a2 = 0.0;
            } else {
                double temp = -(sqrt_n * s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                              2 * e * e * n * p * tn *
                                                                                              tp +
                                                                                              e * e * n * p * tp * tp +
                                                                                              2 * e * n * n * p * te *
                                                                                              tn +
                                                                                              2 * e * n * n * p * te *
                                                                                              tp +
                                                                                              N * e * n * n * tp * tp +
                                                                                              2 * e * n * p * p * te *
                                                                                              tn +
                                                                                              2 * e * n * p * p * te *
                                                                                              tp - 2 *
                                                                                                   N *
                                                                                                   e *
                                                                                                   n *
                                                                                                   p *
                                                                                                   tn *
                                                                                                   tp +
                                                                                              N * e * p * p * tn * tn +
                                                                                              n * n * n * p * te * te +
                                                                                              2 * n * n * p * p * te *
                                                                                              te +
                                                                                              n * p * p * p * te * te));
                if (temp < 0 || temp > upper) a2 = 0.0;
                else a2 = temp;
            }
            return a2;
        }

        double a2_3(double sqrt_n, size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                    double upper) {
            double a2 = 0.0;
            double upperside = N * e * n * p * (n + p) * (
                    e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp -
                    2 * e * n * n * p * te * tn - 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp -
                    2 * e * n * p * p * te * tn - 2 * e * n * p * p * te * tp - 2 *
                                                                                N * e * n * p * tn * tp +
                    N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
                    n * p * p * p * te * te);
            if (upperside <= 0) a2 = 0.0;
            else {
                double temp = sqrt_n * (s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                             2 * e * e * n * p * tn *
                                                                                             tp +
                                                                                             e * e * n * p * tp * tp -
                                                                                             2 * e * n * n * p * te *
                                                                                             tn -
                                                                                             2 * e * n * n * p * te *
                                                                                             tp +
                                                                                             N * e * n * n * tp * tp -
                                                                                             2 * e * n * p * p * te *
                                                                                             tn -
                                                                                             2 * e * n * p * p * te *
                                                                                             tp - 2 *
                                                                                                  N *
                                                                                                  e *
                                                                                                  n *
                                                                                                  p *
                                                                                                  tn *
                                                                                                  tp +
                                                                                             N * e * p * p * tn * tn +
                                                                                             n * n * n * p * te * te +
                                                                                             2 * n * n * p * p * te *
                                                                                             te +
                                                                                             n * p * p * p * te * te));
                if (temp < 0 or temp > upper) a2 = 0.0;
                else a2 = temp;
            }
            return a2;
        }

        double a2_4(double sqrt_n, size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                    double upper) {
            double a2 = 0.0;
            double upperside = N * e * n * p * (n + p) * (
                    e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp -
                    2 * e * n * n * p * te * tn - 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp -
                    2 * e * n * p * p * te * tn - 2 * e * n * p * p * te * tp - 2 *
                                                                                N * e * n * p * tn * tp +
                    N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
                    n * p * p * p * te * te);
            if (upperside <= 0) a2 = 0.0;
            else {
                double temp = -(sqrt_n * s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                              2 * e * e * n * p * tn *
                                                                                              tp +
                                                                                              e * e * n * p * tp * tp -
                                                                                              2 * e * n * n * p * te *
                                                                                              tn -
                                                                                              2 * e * n * n * p * te *
                                                                                              tp +
                                                                                              N * e * n * n * tp * tp -
                                                                                              2 * e * n * p * p * te *
                                                                                              tn -
                                                                                              2 * e * n * p * p * te *
                                                                                              tp - 2 *
                                                                                                   N *
                                                                                                   e *
                                                                                                   n *
                                                                                                   p *
                                                                                                   tn *
                                                                                                   tp +
                                                                                              N * e * p * p * tn * tn +
                                                                                              n * n * n * p * te * te +
                                                                                              2 * n * n * p * p * te *
                                                                                              te +
                                                                                              n * p * p * p * te * te));
                if (temp < 0 || temp > upper) a2 = 0.0;
                else a2 = temp;
            }
            return a2;
        }

    } ////End of namespace "internal"
} ////End of namespace "NSPCA"