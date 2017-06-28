#include "Solver.h"
#include "SolverV2Init.h"
#include "Constants.h"
#include <cstdlib>

namespace NSPCA {
    void Solver::transformed_score(unsigned int numThreadsGPU, unsigned int numThreadsGPUZ) {
        cumulate_gpu(numThreadsGPU);

        int numThreads = pools.threads.getNumThreads() + 1;
        int step_size = constants._nVar / numThreads;

        int start = 0;
        int end = step_size;
        vector<std::future<void>> futures((unsigned long) numThreads - 1);
        for (int thread = 0; thread < numThreads - 1; thread++) {
            auto task = internal::solve_transformed_score_cols(count.incidence_count, count.incidence_count_score,
                                                               count.transformed_score_solution, constants._scale,
                                                               constants.sqrNObs, start, end, this->constants);
            futures[thread] = pools.threads.submit(task);
            start = end;
            end += step_size;
        }

        auto task = internal::solve_transformed_score_cols(count.incidence_count, count.incidence_count_score,
                                                           count.transformed_score_solution, constants._scale,
                                                           constants.sqrNObs, start, constants._nVar, this->constants);
        task();
//        auto tmp = 0;
        for (int thread = 0; thread < numThreads - 1; thread++) {
            futures[thread].get();
        }

        set_z(numThreadsGPUZ);

//        std::cout << "Transformed score from CPU is " << std::endl << solutionG.transformed_score << std::endl;
//        int start =0; int end = dim.no_var;
//        auto task = internal::solve_transformed_score_cols(count.incidence_count, count.incidence_count_score,
//                                                           count.transformed_score_solution, constants.scale,
//                                                           constants.square_n, start, dim.no_var, this->dim);
//        task();
//
//        set_z(numThreadsGPU);
    }

    void Solver::cumulate_gpu(unsigned int numThreads) {
        linSolver.matmul_nn(1.0, solutionG.principal_score, solutionG.component_loading, 0.0, tmp_matrices.AP);
//        std::cout << "Matrix AP is " << std::endl;
//        std::cout << tmp_matrices.AP << std::endl;
//        std::cout << "Data is" << std::endl << data_gpu << std::endl;
        linSolver.sync();
//        std::cout << tmp_matrices.AP << std::endl;
        cumulate_kernel_gpu(data_gpu.data(), constants._nObs, constants._nVar, tmp_matrices.AP.data(),
                            count.incidence_count_score_gpu.data(),
                            numThreads, pools.streams.getCurrentStream());
        pools.streams.syncAll();
        count.copy_score_to_cpu(pools.streams.getCurrentStream());
        pools.streams.syncAll();

    }


    void Solver::get_principal_score(double *dst) {
        solutionG.get_principal_score(dst, pools.streams.getCurrentStream());
        pools.streams.syncAll();
    }

    void Solver::get_transformed_score(double *dst) {
        solutionG.get_transformed_score(dst, pools.streams.getCurrentStream());
        pools.streams.syncAll();
    }

    void Solver::get_component_loading(double *dst) {
        solutionG.get_component_loading(dst, pools.streams.getCurrentStream());
        pools.streams.syncAll();
    }


    void Solver::get_cumu_score(double *dst) {
        cuStat::copy_to_cpu(dst, count.incidence_count_score_gpu, pools.streams.getCurrentStream());
        pools.streams.syncAll();
    }

    void Solver::get_incidence_count(int *dst) {
        memcpy(dst, count.incidence_count, sizeof(int) * 3 * constants._nVar);
    }

    void Solver::set_z(unsigned int numThreadsGPUZ) {
        count.copy_solution_to_gpu(pools.streams.getCurrentStream());
        set_z_gpu(data_gpu.data(), solutionG.transformed_score.data(), count.transformed_score_solution_gpu.data(),
                  constants._nObs, constants._nVar, numThreadsGPUZ, pools.streams.getCurrentStream());
        pools.streams.syncAll();
    }

//    bool
//    Solver::solve(double *transformed_score, double *principal_score, double *component_loading, double lambda,
//                  double threshold,
//                  int max_iter, unsigned int ts_thread1, unsigned int ts_thread2, unsigned int cl_thread,
//                  unsigned int cl_block) {
//        int iter = 0;
//        bool convergence = false;
//
//        constants.set_lambda(lambda);
//
//        while (iter < max_iter && !convergence) {
//            std::cout << "iter is " << iter << std::endl;
//            this->transformed_score(ts_thread1, ts_thread2);
////            std::cout << "Transformed score after iteration" << std::endl << solutionG.transformed_score << std::endl;
//            this->component_loading(cl_thread, cl_block);
////            std::cout << "Component loading after iteration " << std::endl << solutionG.component_loading << std::endl;
//            this->principal_score();
////            std::cout << "Principal score after iteration" << std::endl << solutionG.principal_score << std::endl;
//
//
//            auto diff = solutionG.compare_difference(pools.streams.getCurrentStream());
//            cudaStreamSynchronize(pools.streams.getCurrentStream());
//
//            if (std::abs(diff) < threshold) convergence = true;
//            else {
//                iter++;
//                solutionG.copy_component_loading();
//            }
//
//        }
//
//        if (!convergence)
//            std::cout
//                    << "Warning: the algorithm has failed to converge. Returning results from final iteration as solution"
//                    << std::endl;
//
//        cuStat::copy_to_cpu(principal_score, solutionG.principal_score, pools.streams.getCurrentStream());
//        cuStat::copy_to_cpu(transformed_score, solutionG.transformed_score, pools.streams.getNextStream());
//        cuStat::copy_to_cpu(component_loading, solutionG.component_loading, pools.streams.getNextStream());
//        pools.streams.syncAll();
//        return convergence;
//    }

    bool
    Solver::solve_v2(double *transformed_score, double *principal_score, double *component_loading, double *lambda,
                     double threshold,
                     int max_iter, unsigned int ts_thread1, unsigned int ts_thread2, unsigned int cl_thread,
                     unsigned int cl_block) {
        int iter = 0;
        bool convergence = false;

        tmp_matrices.set_lambdas(lambda, pools.streams.getCurrentStream());
//        std::cout << "Initial Component Loading is " << solutionG.component_loading << std::endl;
        pools.streams.syncAll();
//        std::cout << tmp_matrices.lambdas << std::endl;
        while (iter < max_iter && !convergence) {
//            std::cout << "iter is " << iter << std::endl;
            this->transformed_score(ts_thread1, ts_thread2);
//            std::cout << "Transformed score after iteration" << std::endl << solutionG.transformed_score << std::endl;
            this->component_loading_v2(cl_thread, cl_block);
//            std::cout << "Component loading after iteration " << std::endl << solutionG.component_loading << std::endl;
            this->principal_score_v2();
//            std::cout << "Principal score after iteration" << std::endl << solutionG.principal_score << std::endl;


            auto diff = solutionG.compare_difference(pools.streams.getCurrentStream());
            cudaStreamSynchronize(pools.streams.getCurrentStream());

            if (std::abs(diff) < threshold) convergence = true;
            else {
                iter++;
                solutionG.copy_component_loading();
            }

        }

        if (!convergence)
            std::cout
                    << "Warning: the algorithm has failed to converge. Returning results from final iteration as solution"
                    << std::endl;

        cuStat::copy_to_cpu(principal_score, solutionG.principal_score, pools.streams.getCurrentStream());
        cuStat::copy_to_cpu(transformed_score, solutionG.transformed_score, pools.streams.getNextStream());
        cuStat::copy_to_cpu(component_loading, solutionG.component_loading, pools.streams.getNextStream());
        pools.streams.syncAll();
        return convergence;
    }

    void Solver::principal_score_v2() {
//        std::cout << "maybe here?" <<std::endl;
        linSolver.matmul_nt(1.0, tmp_matrices.Omega, solutionG.component_loading, 0.0, tmp_matrices.OmegaPT);
//        std::cout << "flag1" << std::endl;
        linSolver.matmul_nn(1.0, solutionG.transformed_score, tmp_matrices.OmegaPT, 0.0, tmp_matrices.ZPT);
//        std::cout << "flag2" << std::endl;
        linSolver.sync();
        linSolver.thin_svd(tmp_matrices.ZPT, tmp_matrices.U, tmp_matrices.V, tmp_matrices.diag);
        linSolver.sync();
        linSolver.matmul_nt(constants._scale * constants.sqrNObs, tmp_matrices.UView, tmp_matrices.V, 0.0,
                         solutionG.principal_score);
        linSolver.sync();
//        std::cout << solutionG.principal_score << std::endl;
    }
    void Solver::component_loading_v2(int numThreads, int numBlocks) {
        linSolver.matmul_tn(1.0, solutionG.principal_score, solutionG.transformed_score, 0.0, tmp_matrices.ATZ);
//        std::cout << tmp_matrices.ATZ << std::endl;
        linSolver.matmul_nn(1.0, tmp_matrices.ATZ, tmp_matrices.Omega, 0.0, tmp_matrices.ATZ);

        linSolver.sync();
//        std::cout << "Weights" << std::endl <<  tmp_matrices.weights << std::endl;
        solve_p_nspca_v2(solutionG.component_loading.data(), constants._nObs, constants._nVar, constants._nVarAfterReduce,
                         tmp_matrices.ATZ.data(), restriction_gpu.data(), tmp_matrices.lambdas.data(),
                         tmp_matrices.omegaVec.data(), constants.nTimesScaleSquare,
                         numThreads,
                         numBlocks, pools.streams.getCurrentStream());
    }


}