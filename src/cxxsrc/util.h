#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <functional>
#include "Threadpool/thread_pool.h"
#include "../custat/cuStat.h"


using cuStat::MatrixXd;
using cuStat::MatrixXi;
using cuStat::ViewXi;
using cuStat::ViewXd;
#include "../nspcaCudalib/nspca_cuda.h"
#include <cmath>
#include "TmpMatrices.h"

namespace NSPCA {
    namespace internal {
        class IncidenceCount {
        public:

            const size_t _no_var;

            ViewXi incidence_count_cpu_view;
            ViewXd incidence_count_score_view;
            ViewXd omega; //! This is a matrix that has weights on its diagonal
            ViewXd omegaVec;
            MatrixXd incidence_count_score_gpu;
            int *incidence_count;
            double *incidence_count_score;
            MatrixXd transformed_score_solution_gpu;
            double *transformed_score_solution;
            double *omegaPtr;
            double *omegaVecPtr;

            IncidenceCount(const size_t no_var) : _no_var(no_var), incidence_count_cpu_view(nullptr, 3, no_var),
                                                  incidence_count_score_view(nullptr, 3, no_var),
                                                  omega(nullptr, no_var, no_var), omegaVec(nullptr, no_var, 1),
                                                  incidence_count_score_gpu(3, no_var),
                                                  transformed_score_solution_gpu(3, no_var) {
                incidence_count = (int *) malloc(sizeof(int) * 3 * no_var);
                incidence_count_score = (double *) malloc(sizeof(double) * 3 * no_var);
                incidence_count_cpu_view.ptrDevice = incidence_count;
                incidence_count_score_view.ptrDevice = incidence_count_score;
                transformed_score_solution = (double *) malloc(sizeof(double) * 3 * no_var);
                omegaPtr = (double *) malloc(sizeof(double) * no_var * no_var);
                omega.ptrDevice = omegaPtr;
                omegaVecPtr = new double[no_var];
                omegaVec.ptrDevice = omegaVecPtr;
            }

            ~IncidenceCount() {
                free(incidence_count);
                free(incidence_count_score);
                free(transformed_score_solution);
                delete omegaVecPtr;
            }

            template<typename Func>
            void set_omega(const size_t no_obs, const Func &func) {
                for (int i = 0; i < _no_var; i++) {
                    for (int j = 0; j < _no_var; j++) {
                        if (i == j) {
                            double frequency = ((double)(no_obs-incidence_count_cpu_view(1,j)))/no_obs;
                            omega(i, j) = func(frequency);
                        } else omega(i, j) = 0;
                    }
                }

                for (int i=0;i<_no_var;i++){
                    omegaVecPtr[i]=omega(i,i);
                }
            }

            void copy_solution_to_gpu(cudaStream_t stream) {
                cudaMemcpyAsync(transformed_score_solution_gpu.data(), transformed_score_solution,
                                sizeof(double) * transformed_score_solution_gpu.rows() *
                                transformed_score_solution_gpu.cols(), cudaMemcpyHostToDevice, stream);
            }

            void copy_score_to_cpu(cudaStream_t stream) {
                cuStat::copy_to_cpu(incidence_count_score, incidence_count_score_gpu, stream);
//                std::cout << "From C++" << std::endl;
//                std::cout << incidence_count_score_gpu << std::endl;
            }


        };


        struct pools {
            ThreadPool threads;
            cuStat::streamPool streams;

            pools(const unsigned int numThreads, const int numStreams) : threads(numThreads), streams(numStreams) {}
        };


    }////End of namespace internal
    class Solution {
    public:
        double *get_transformed_score() const {
            return transformed_score;
        }

        double *get_principal_score() const {
            return principal_score;
        }

        double *get_component_loading() const {
            return component_loading;
        }

        double *transformed_score;
        double *principal_score;
        double *component_loading;


        Solution(double *_transformed_score, double *_principal_score, double *_component_loading) : transformed_score(
                _transformed_score), principal_score(_principal_score), component_loading(_component_loading) {}

    };

    class SolutionGPU {
    public:
        size_t reduce_dim;
        size_t no_var;
        MatrixXd transformed_score;
        MatrixXd principal_score;
        MatrixXd component_loading;
        double *component_loading_copy_new;
        double *component_loading_copy_old;

        SolutionGPU(size_t no_obs, size_t no_var, size_t reduce_dim) : reduce_dim(reduce_dim),
                                                                       no_var(no_var),
                                                                       transformed_score(no_obs, no_var),
                                                                       principal_score(no_obs, reduce_dim),
                                                                       component_loading(reduce_dim, no_var) {
            component_loading_copy_new = new double[reduce_dim * no_var];
            component_loading_copy_old = new double[reduce_dim * no_var];

            for (size_t i = 0; i < reduce_dim * no_var; i++) {
                component_loading_copy_old[i] = 100.0;
                component_loading_copy_new[i] = 100.0;
            }
        }

        ~SolutionGPU() {
            delete component_loading_copy_old;
            delete component_loading_copy_new;
        }

        void get_transformed_score(double *dst, cudaStream_t stream);

        void get_principal_score(double *dst, cudaStream_t stream);

        void get_component_loading(double *dst, cudaStream_t stream);

        void copy_component_loading();

        double compare_difference(cudaStream_t stream);
    };

    double SolutionGPU::compare_difference(cudaStream_t stream) {
        cudaMemcpyAsync(component_loading_copy_new, component_loading.data(), sizeof(double) * reduce_dim * no_var,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        double diff = 0;

        for (size_t i = 0; i < reduce_dim * no_var; i++) {
            diff += std::abs(component_loading_copy_new[i] - component_loading_copy_old[i]);
        }

        return diff;

    }

    void SolutionGPU::copy_component_loading() {
        memcpy(component_loading_copy_old, component_loading_copy_new, sizeof(double) * reduce_dim * no_var);
    }

    void SolutionGPU::get_component_loading(double *dst, cudaStream_t stream) {
        cuStat::copy_to_cpu(dst, component_loading, stream);
    }

    void SolutionGPU::get_principal_score(double *dst, cudaStream_t stream) {
        cuStat::copy_to_cpu(dst, principal_score, stream);
    }

    void SolutionGPU::get_transformed_score(double *dst, cudaStream_t stream) {
        cuStat::copy_to_cpu(dst, transformed_score, stream);
    }


}