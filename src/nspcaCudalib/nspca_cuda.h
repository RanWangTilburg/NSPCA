#pragma once

#include <cuda_runtime.h>

namespace NSPCA {
////Used in initialization. Check whether the signs of initialized signs of the initialized P is consistent with restrictions
    void check_res_gpu(double *matP, int *res, const size_t p, const size_t P, const unsigned int numThreads,
                       const unsigned int numBlocks, cudaStream_t stream);

////Count number of incidences in intialiazation
    void
    count_incidence_gpu(int *data, const size_t N, const size_t P, int *incidence_count, const unsigned int numThreads,
                        cudaStream_t stream);

////Solve p in the nspca algorithm
    void solve_p_nspca(double *devp, const size_t N, const size_t P, const size_t p, double *ATZ,
                       int *restriction, const double lambda, const double scale_square, const unsigned int numThreads,
                       const unsigned int numBlocks, cudaStream_t stream);

    void
    cumulate_kernel_gpu(int *data, const size_t N, const size_t P, double *AP, double *cumulate,
                        unsigned int numThreads, cudaStream_t stream);


    void get_upper_bound(int *incidence_count, const size_t P, double *upper, double s, unsigned int numThreads,
                         cudaStream_t stream);

    void set_z_gpu(int *data, double *matZ, double *solutions, size_t N, size_t P, unsigned int numThreads,
                   cudaStream_t stream);

    void solve_p_nspca_v2(double *devp, const size_t N, const size_t P, const size_t p, double *ATZ,
                          int *restriction, double *lambda, double *omegaVec, const double scale_square,
                          const unsigned int numThreads,
                          const unsigned int numBlocks, cudaStream_t stream);

    void update_weights(double *lambda_weights, double new_lambda, double P, const unsigned int numThreads,
                        const unsigned int numBlocks, cudaStream_t stream);
}////end of NSPCA namespace