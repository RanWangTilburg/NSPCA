#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include "linear_lasso_impl.h"
#include "../util/matrix.h"

namespace cuStat {
    class LinearLassoCDC {
    private:
        cudaStream_t stream;
        MatrixXd data_gpu;
        MatrixXd residual_gpu;
        MatrixXd beta_init_gpu;
        size_t no_obs;
        size_t no_var;

    public:

        LinearLassoCDC(const size_t no_obs, const size_t no_var, const double *data, const double *beta_init);

        ~LinearLassoCDC();

        void get_residual(double *dst);

        void update_residual(const size_t ncol, const size_t newpara, const size_t nblocks, const size_t nthreads);

        void get_temp_a();

    };

    LinearLassoCDC::LinearLassoCDC(const size_t no_obs, const size_t no_var, const double *data,
                                   const double *beta_init)
            : data_gpu(no_obs, no_var),
              residual_gpu(no_obs, 1),
              beta_init_gpu(no_var, 1), no_obs(no_obs), no_var(no_var) {
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(data_gpu.data(), data, sizeof(double) * no_obs * no_var, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(beta_init_gpu.data(), beta_init, sizeof(double) * no_var, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

    }

    void LinearLassoCDC::update_residual(const size_t ncol, const size_t newpara, const size_t nblocks,
                                         const size_t nthreads) {
        linear_lasso_impl::update_residual <<< nblocks, nthreads, 0, stream >>>
                                                                      (residual_gpu.data(), data_gpu.data(), no_obs, no_var, ncol, newpara);
        cudaStreamSynchronize(stream);
    }


    void LinearLassoCDC::get_residual(double *dst) {
        cudaMemcpyAsync(dst, residual_gpu.data(), sizeof(double) * no_obs, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    LinearLassoCDC::~LinearLassoCDC() {
        cudaStreamDestroy(stream);
    }
}