#pragma once

#include <cassert>

#include <cuda_runtime.h>
#include <vector>

using std::vector;
#include <thread>
#include <future>
#include <cmath>
#include <cstdlib>

#include "../custat/cuStat.h"
#include "transformed_score.h"
#include "Threadpool/thread_pool.h"
#include "../nspcaCudalib/nspca_cuda.h"
#include "Constants.h"
#include "util.h"


extern "C" void cumulate_cpu_(const int *data, int *count, int no_obs, int no_var);

namespace NSPCA {

    class Solver {
    public:
        cuStat::linSolver linSolver;
        internal::Constants constants;
        internal::TmpMatrices tmp_matrices;
        internal::IncidenceCount count;
        internal::pools pools;

        SolutionGPU solutionG;
        Solution solution;

        MatrixXi data_gpu;
        MatrixXi restriction_gpu;

        Solver(size_t nObs, size_t nVar, size_t nVarAfterReduce, double scale,
               unsigned int nThreadsInPools, int nStreamsInPools);

        void init(const int *data, const int *restriction, double *transformed_score, double *principal_score,
                  double *component_loading);

        bool solve_v2(double *transformed_score, double *principal_score, double *component_loading, double *lambda,
                      double threshold,
                      int max_iter, unsigned int ts_thread1, unsigned int ts_thread2, unsigned int cl_thread,
                      unsigned int cl_block);

        /////////////////////////////////////////////////////////////////////////////
        ////Deprecated////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////

        void transformed_score(unsigned int numThreadsGPU, unsigned int numThreadsGPUZ);

        void principal_score_v2();


        void component_loading_v2(int numThreads, int numBlocks);

        void cumulate_gpu(unsigned int numThreads);

        void get_transformed_score(double *dst);

        void get_principal_score(double *dst);

        void get_component_loading(double *dst);



        void get_incidence_count(int *dst);

        void get_cumu_score(double *dst);

        void set_z(unsigned int numThreadsGPUZ);
    };


}; ////End of NSPCA