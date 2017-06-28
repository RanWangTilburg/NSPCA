#pragma once
#include "Solver.h"

#include <iostream>
#include <cstdlib>

namespace NSPCA{
    Solver::Solver(size_t nObs, size_t nVar, size_t nVarAfterReduce, double scale, unsigned int nThreadsInPools,
                   int nStreamsInPools):linSolver(), constants(nObs,nVar, nVarAfterReduce, scale),
                                   tmp_matrices(nObs, nVar, nVarAfterReduce), count(nVar), pools(nThreadsInPools, nStreamsInPools),
                                   solutionG(nObs, nVar, nVarAfterReduce),
                                   solution(nullptr, nullptr, nullptr), data_gpu(nObs, nVar),
                                   restriction_gpu(nVarAfterReduce, nVar) {}

    void Solver::init(const int *data, const int *restriction, double *transformed_score, double *principal_score,
                      double *component_loading) {
//            std::cout << "flag" <<std::endl;
            cuStat::copy_to_gpu(data, data_gpu, pools.streams.getCurrentStream());
            cuStat::copy_to_gpu(restriction, restriction_gpu, pools.streams.getNextStream());
            cuStat::copy_to_gpu(transformed_score, solutionG.transformed_score, pools.streams.getNextStream());
            cuStat::copy_to_gpu(principal_score, solutionG.principal_score, pools.streams.getNextStream());
            cuStat::copy_to_gpu(component_loading, solutionG.component_loading, pools.streams.getNextStream());
            int temp_no_obs = (int) constants._nObs;
            int temp_no_var = (int) constants._nVar;
            cumulate_cpu_(data, count.incidence_count, temp_no_obs, temp_no_var);

            //! Now really need to define the function that would change it
            auto set_omega_lambda = [](double frequency) { return frequency; };
            count.set_omega(constants._nObs, set_omega_lambda);
            cuStat::copy_to_gpu(count.omegaPtr, tmp_matrices.Omega, pools.streams.getNextStream());
            cuStat::copy_to_gpu(count.omegaVecPtr, tmp_matrices.omegaVec, pools.streams.getNextStream());
            pools.streams.syncAll();
//            std::cout <<  tmp_matrices.Omega << std::endl;
        }
}