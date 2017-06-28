//
// Created by user on 15-11-16.
//

#include "Solver_old.h"
#include "../nspcaCudalib/nspca_cuda.h"

namespace NSPCA {

    using cuStat::standardize;

    void
    solve_z(const size_t j, MatrixXi &incidence_count_host, MatrixXd &cumulation_host, MatrixXd &solutions_for_z_host,
            size_t N, size_t P, double scale) {
        size_t n = incidence_count_host(0, j);
        size_t e = incidence_count_host(1, j);
        size_t p = incidence_count_host(2, j);

        ////Need to perform the cumulation on GPU
        double tn = cumulation_host(0, j);
        double te = cumulation_host(1, j);
        double tp = cumulation_host(2, j);
        double s = scale;
        double lower = 0.0;
        double upper = sqrt((n + p) * s * s / (n * p));

        solutionFvalue solution1;
        solutionFvalue solution2;
        solutionFvalue solution3;
        solutionFvalue solution4;
        solutionFvalue solution5;
        solutionFvalue solution6;

        solution1.solution = 0.0;

        double a2 = solution1.solution;
        getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution1, a2);
        a2 = upper;
        getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution2, a2);
        solution1 = compareSolution(solution1, solution2);

        a2 = geta2sol1(N, n, e, p, tn, te, tp, s, upper);
        getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution3, a2);
        solution1 = compareSolution(solution1, solution3);
        a2 = geta2Sol2(N, n, e, p, tn, te, tp, s, upper);
        getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution4, a2);
        solution1 = compareSolution(solution1, solution4);

        geta2sol3(N, n, e, p, tn, te, tp, s, upper);
        getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution5, a2);
        solution1 = compareSolution(solution1, solution5);

        a2 = geta2Sol4(N, n, e, p, tn, te, tp, s, upper);
        getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution6, a2);
        solution1 = compareSolution(solution1, solution2);

        solutions_for_z_host(0, j) = solution1.solution1;
        solutions_for_z_host(1, j) = solution1.solution2;
        solutions_for_z_host(2, j) = solution1.solution;
    }

    struct z_solver {
        int operator()() {
            for (int i = lower; i < upper; i++) {
                solve_z(i, incidence_count, cumulation, solutions, N, P, scale);
            }
            return 0;
        }

        z_solver(MatrixXi &incidence_count, MatrixXd &cumulation, MatrixXd &solutions, size_t N, size_t P, double scale,
                 size_t lower,
                 size_t upper) : incidence_count(incidence_count), cumulation(cumulation), solutions(solutions), N(N),
                                 P(P), scale(scale), lower(lower), upper(upper) {}


        MatrixXi &incidence_count;
        MatrixXd &cumulation;
        MatrixXd &solutions;

        size_t N;
        size_t P;

        size_t lower;
        size_t upper;
        double scale;

    };

    Solver::Solver(const size_t _N, const size_t _P, const size_t _p, const double scale_, const int numThreads,
                   const int numStreams) : threadPool(numThreads), streamPool(numStreams), cuSolver(),
                                           workSpace(N * P, N * P), N(_N), P(_P), p(_p), devZ(N, P),
                                           devA(N, p), devP(p, P), devU(N, N),
                                           devVT(p, p), devZTP(N, p), devATZ(p, P), devS(p, 1), hostVT(p, p),
                                           devData(N, P),
                                           devRestriction(p, P), cumulation(P, 3), incidence_count(3, P), matZP(N, P),
                                           matATZ(p, P), incidence_count_host(3, P), cumulation_host(3, P),
                                           solutions_for_z_host(3, P), solutions_for_z_gpu(3, P) {


/*
 *  Initialize CPU data
 * */
        matZ = MatrixXd::Constant(N, P, 0.0);
        matA = MatrixXd::Constant(N, p, 0.0);
        matP = MatrixXd::Constant(p, P, 0.0);
        incidence_count_host = MatrixXi::Constant(3, P, 0);
        cumulation_host = MatrixXd::Constant(3, P, 0);

        lambda = 0.0;
        scale = scale_;
        scale_square = scale * scale;
/*
 *  Initialize steps and lambda information
 * */
        lower_lambda = 0;
        upper_lambda = 0;
        steps = 0;
        step_size = 0;
/*
 * Allocate Work Space
 * */
        svd = new JacobiSVD<MatrixXd>(); ////Used for initialization only
    }

    template<typename MatType>
    void Solver::init_with_data(const MatType &data, const MatType &restriction, const double _lower_lambda,
                                const double _upper_lambda, const double _steps) {

        ////Initiliaze the lambdas needed
        assert(_lower_lambda >= 0);
        assert(_upper_lambda >= _lower_lambda);
        assert(_steps > 0);
        init_lambdas(_lower_lambda, _upper_lambda, _steps);

        ////Check the dimensions are the same as specified
        assert(data.rows() == N && data.cols() == P);
        assert(restriction.rows() == p && restriction.cols() == P);

        ////Copy data and restriction to GPU
        copyFromEigenAsync(data, devData, streamPool.getCurrentStream());
        count_incidence_gpu(devData.data(), N, P, incidence_count.data(), 512, streamPool.getCurrentStream());
        copyToEigenAsync(incidence_count_host, incidence_count, streamPool.getCurrentStream());
        copyFromEigenAsync(restriction, devRestriction, streamPool.getNextStream());

        ////Initialize Z, A and P using SVD, consistent with classical PCA analysis
        init_with_svd(data);
        copyFromEigenAsync(matZ, devZ, streamPool.getNextStream());
        copyFromEigenAsync(matA, devA, streamPool.getNextStream());
        ////Check the signs of P's and make it consistent with restrictions
        check_signs_with_restriction(restriction);
        copyFromEigenAsync(matP, devP, streamPool.getNextStream());

        ////Reinitialize incidence count
        streamPool.syncAll();

        ////Synchronize

    }

    ////This routine checks whehter signs are consistent with the pre-specified restrictions
    template<typename MatType>
    void Solver::check_signs_with_restriction(const MatType &restriction) {
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < P; j++) {
                double temp = matP(i, j);
                if (temp > 0 && restriction(i, j) == -1) {
                    matP(i, j) = -temp;
                } else if (temp < 0 && restriction(i, j) == 1) {
                    matP(i, j) = -temp;
                } else if (restriction(i, j) == 0) {
                    matP(i, j) = 0;
                }
            }
        }
    }

    ////Initialize with Singular Value decomposition
    void Solver::init_with_svd(const MatrixXi &data) {
        matZ = data.cast<double>();
        standardize(matZ);
        matZ = matZ * scale;
        svd->compute(matZ, ComputeThinU | ComputeThinV);
        matA = scale_square * svd->matrixU().block(0, 0, N, p);
        MatrixXd temp = svd->singularValues().asDiagonal();
        matP = temp.block(0, 0, p, p) * svd->matrixV().transpose().block(0, 0, p, P);
        matP = matP * scale;
    }

    Solver::~Solver() {
        delete svd;
    }


    void Solver::init_lambdas(const double _lower_lambda, const double _upper_lambda, const double _steps) {
        lower_lambda = _lower_lambda;
        upper_lambda = _upper_lambda;
        steps = _steps;

        step_size = (upper_lambda - lower_lambda) / steps;
        lambda = lower_lambda;
    }

    void Solver::solve_a() {
        cuSolver.matmul_tn(1.0, devZ, devP, 0.0, devZTP);
        cudaDeviceSynchronize();
        cuSolver.thin_svd(devZTP, devU, devS, devVT);
        cuSolver.sync();
        cuStat::inverseWithEigenTranspose(devVT, streamPool.getCurrentStream());
        cudaStreamSynchronize(streamPool.getCurrentStream());
        cuSolver.matmul_nn(scale, devU, devVT, 0.0, devA);
        cudaDeviceSynchronize();
    }

    size_t Solver::getN() const {
        return N;
    }

    size_t Solver::getP() const {
        return P;
    }

    size_t Solver::getp() const {
        return p;
    }


    template void
    Solver::init_with_data<MatrixXi>(const MatrixXi &data, const MatrixXi &restriction, const double, const double,
                                     const double);


    double
    geta2Sol4(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper) {
        double a2;
        double upperside = N * e * n * p * (n + p) *
                           (e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp -
                            2 * e * n * n * p * te * tn - 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp -
                            2 * e * n * p * p * te * tn - 2 * e * n * p * p * te * tp - 2 *
                                                                                        N * e * n * p * tn * tp +
                            N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
                            n * p * p * p * te * te);
        if (upperside < 0) {
            a2 = 0.0;
        } else {
            double temp = -(s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                 2 * e * e * n * p * tn * tp +
                                                                                 e * e * n * p * tp * tp -
                                                                                 2 * e * n * n * p * te * tn -
                                                                                 2 * e * n * n * p * te * tp +
                                                                                 N * e * n * n * tp * tp -
                                                                                 2 * e * n * p * p * te * tn -
                                                                                 2 * e * n * p * p * te * tp - 2 *
                                                                                                               N * e *
                                                                                                               n * p *
                                                                                                               tn * tp +
                                                                                 N * e * p * p * tn * tn +
                                                                                 n * n * n * p * te * te +
                                                                                 2 * n * n * p * p * te * te +
                                                                                 n * p * p * p * te * te));
            if (temp < 0 || temp > upper) {
                a2 = 0.0;
            } else {
                a2 = temp;
            }
        }
        return a2;
    }

    double
    geta2sol3(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper) {
        double a2;
        double upperside = N * e * n * p * (n + p) *
                           (e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp -
                            2 * e * n * n * p * te * tn - 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp -
                            2 * e * n * p * p * te * tn - 2 * e * n * p * p * te * tp - 2 *
                                                                                        N * e * n * p * tn * tp +
                            N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
                            n * p * p * p * te * te);
        if (upperside < 0) {
            a2 = 0.0;
        } else {
            double temp = (s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                2 * e * e * n * p * tn * tp +
                                                                                e * e * n * p * tp * tp -
                                                                                2 * e * n * n * p * te * tn -
                                                                                2 * e * n * n * p * te * tp +
                                                                                N * e * n * n * tp * tp -
                                                                                2 * e * n * p * p * te * tn -
                                                                                2 * e * n * p * p * te * tp - 2 *
                                                                                                              N * e *
                                                                                                              n * p *
                                                                                                              tn * tp +
                                                                                N * e * p * p * tn * tn +
                                                                                n * n * n * p * te * te +
                                                                                2 * n * n * p * p * te * te +
                                                                                n * p * p * p * te * te));
            if (temp < 0 || temp > upper) {
                a2 = 0.0;
            } else {
                a2 = temp;
            }
        }
        return a2;
    }

    solutionFvalue &compareSolution(solutionFvalue &solution1, const solutionFvalue &solution2) {
        if (solution1.Fvalue > solution2.Fvalue) {
            solution1 = solution2;
        }
        return solution1;
    }

    double
    geta2Sol2(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper) {
        double a2;
        double upperside = N * e * n * p * (n + p) *
                           (e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp +
                            2 * e * n * n * p * te * tn + 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp +
                            2 * e * n * p * p * te * tn + 2 * e * n * p * p * te * tp - 2 *
                                                                                        N * e * n * p * tn * tp +
                            N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
                            n * p * p * p * te * te);
        if (upperside < 0) {
            a2 = 0.0;
        } else {
            double temp = -(s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                 2 * e * e * n * p * tn * tp +
                                                                                 e * e * n * p * tp * tp +
                                                                                 2 * e * n * n * p * te * tn +
                                                                                 2 * e * n * n * p * te * tp +
                                                                                 N * e * n * n * tp * tp +
                                                                                 2 * e * n * p * p * te * tn +
                                                                                 2 * e * n * p * p * te * tp - 2 *
                                                                                                               N * e *
                                                                                                               n * p *
                                                                                                               tn * tp +
                                                                                 N * e * p * p * tn * tn +
                                                                                 n * n * n * p * te * te +
                                                                                 2 * n * n * p * p * te * te +
                                                                                 n * p * p * p * te * te));
            if (temp < 0 || temp > upper) {
                a2 = 0.0;
            } else {
                a2 = temp;
            }
        }
        return a2;
    }

    double
    geta2sol1(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper) {
        double a2;
        double upperside = (e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp +
                            e * e * n * p * tp * tp + 2 * e * n * n * p * te * tn + 2 * e * n * n * p * te * tp +
                            N * e * n * n * tp * tp + 2 * e * n * p * p * te * tn +
                            2 * e * n * p * p * te * tp - 2 * N * e * n * p * tn * tp + N * e * p * p * tn * tn +
                            n * n * n * p * te * te + 2 * n * n * p * p * te * te + n * p * p * p * te * te);
        if (upperside <= 0) {
            a2 = 0.0;
        } else {
            double temp = (s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                2 * e * e * n * p * tn * tp +
                                                                                e * e * n * p * tp * tp +
                                                                                2 * e * n * n * p * te * tn +
                                                                                2 * e * n * n * p * te * tp +
                                                                                N * e * n * n * tp * tp +
                                                                                2 * e * n * p * p * te * tn +
                                                                                2 * e * n * p * p * te * tp - 2 *
                                                                                                              N * e *
                                                                                                              n * p *
                                                                                                              tn * tp +
                                                                                N * e * p * p * tn * tn +
                                                                                n * n * n * p * te * te +
                                                                                2 * n * n * p * p * te * te +
                                                                                n * p * p * p * te * te));
            if (temp < 0 || temp > upper) {
                a2 = 0.0;
            } else {
                a2 = temp;
            }
        }
        return a2;
    }

    void getSolutionFromGivenA2(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                                double upper, solutionFvalue &solution1, double a2) {
        if (a2 <= upper & a2 >= 0) {
            double L = sqrt((-a2 * a2 * n * p + (n + p) * s * s) / (e * N));
            double a0 = (-a2 * p + e * L) / (n + p);
            double a1 = -L;
            double fvalue = tn * a0 + te * a1 + (a0 + a2) * tp;

            double a0other = -(a2 * p + e * L) / (n + p);
            double a1other = L;
            solution1.solution = a2;
            double fvalue2 = tn * a0 + te * a1 + (a0 + a2) * tp;
            if (fvalue > fvalue2) {

                solution1.solution1 = a0;
                solution1.solution2 = a1;
                solution1.Fvalue = fvalue;
            } else {
                solution1.solution1 = a0other;
                solution1.solution2 = a1other;
                solution1.Fvalue = fvalue2;
            }
        }
    }

    void Solver::solve_p(const unsigned int numThreads, const unsigned int numBlocks) {
        cuSolver.matmul_tn(1.0, devA, devZ, 0.0, devATZ);
        cuSolver.sync();

        solve_p_nspca(devP.data(), N, P, p, devATZ.data(), devRestriction.data(), lambda, scale_square, numThreads,
                      numBlocks);
        cudaDeviceSynchronize();
    }


}




