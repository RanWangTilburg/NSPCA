#pragma once

#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <vector>

using std::vector;

#include "../custat/cuStat.h"
#include "../custat/cuStatHost.h"

#include "Threadpool/thread_pool.h"


//using cuExec::devWorkSpace;
//using cuExec::devScalar;
//using cuExec::cuSVD;
//using cuExec::standardize;
//using cuExec::cuMatExec;
//using cuExec::copyFromHostAsync;
//using cuExec::copyToHostAsync;
//using cuExec::cuView;




using namespace Eigen;
namespace NSPCA {


    union solutionFvalue {
        double solution;
        double solution1;
        double solution2;
        double Fvalue;
    };


    class Solver {
    public:

/*
 * Utilities
 * */
        JacobiSVD<MatrixXd> *svd;
        ThreadPool threadPool;
        cuStat::streamPool streamPool;
        cuStat::linSolver cuSolver;
        cuStat::WorkSpace workSpace;
    public:

        struct CPUSolution {
            //// This class encaptures the solution for the NSPCA algorithm on CPU

            MatrixXd matZ;
            MatrixXd matA;
            MatrixXd matP;

        };

        struct GPUSolution {
            //// This class encaptures the solution for the NSPCA algorithm on GPU

            cuStat::Matrix<double> devZ;
            cuStat::Matrix<double> devA;
            cuStat::Matrix<double> devP;

        };

        struct MemPool {
            ////Class for stream and thread pools

        };

        struct WorkData{
            ////The data that are internally needed for the computation
        };
        ////Dimensionality
        size_t N; ////Number of observations
        size_t P; ////Number of variables
        size_t p; ////The reduced number of dimension

        size_t getN() const;

        size_t getP() const;

        size_t getp() const;

        MatrixXd matZ; //N times P
        MatrixXd matA; //N times P
        MatrixXd matP; //p times P

        cuStat::Matrix<double> devZ;
        cuStat::Matrix<double> devA;
        cuStat::Matrix<double> devP;

        double lambda;
        double scale;
        double scale_square;

        double lower_lambda;
        double upper_lambda;
        double steps;
        double step_size;
        /*
         *
         *
         * Internal data for workspaces
         *
         * */

        cuStat::Matrix<double> devU;
        cuStat::Matrix<double> devVT;
        cuStat::Matrix<double> devZTP;
        cuStat::Matrix<double> devATZ;
        cuStat::Matrix<double> devS;

        MatrixXd hostVT;
        /*
         *
         * References to the data and restriction
         *
         * */
        cuStat::Matrix<int> devData;        ////N*P
        cuStat::Matrix<int> devRestriction; ////p * P


        cuStat::Matrix<double> cumulation; ////3*P
        cuStat::Matrix<int> incidence_count; ////3*P

        cuStat::Matrix<double> matZP;
        cuStat::Matrix<double> matATZ;

        MatrixXi incidence_count_host;
        MatrixXd cumulation_host;
        MatrixXd solutions_for_z_host;
        cuStat::Matrix<double> solutions_for_z_gpu;

        Solver(const size_t _N, const size_t _P, const size_t _p, const double scale_, const int numThreads,
               const int numStreams);

        ~Solver();

        template<typename MatType>
        void init_with_data(const MatType &data, const MatType &restriction, const double _lower_lambda,
                            const double _upper_lambda, const double _steps);

        template<typename MatType>
        void check_signs_with_restriction(const MatType &restriction);

        void init_with_svd(const MatrixXi &data);

        void solve_a();

        void solve_p(const unsigned int numThreads, const unsigned int numBlocks);

        void solve_z();

        void init_lambdas(const double _lower_lambda, const double _upper_lambda, const double _steps);

    private:
        void solve_z(const size_t j);


    };

    void
    getSolutionFromGivenA2(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s,
                           double upper,
                           solutionFvalue &solution1, double a2);

    double geta2sol1(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper);

    double geta2Sol2(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper);

    solutionFvalue &compareSolution(solutionFvalue &solution1, const solutionFvalue &solution2);

    double geta2sol3(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper);

    double geta2Sol4(size_t N, size_t n, size_t e, size_t p, double tn, double te, double tp, double s, double upper);
}




