#pragma once

#include <cuda_runtime.h>
#include "../util/matrix.h"
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
//#include "../util/atomicAdd.h"
namespace cuStat {
    namespace linear_lasso_impl {


        __global__ void
        update_residual(double *residual, double *data, const size_t nrow, const size_t ncol, const size_t col,
                        const double new_para) {
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int numThreads = blockDim.x;
            const int index = tid + bid * numThreads;
            if (index < nrow) {
                ViewXd residual_view = ViewXd(residual, nrow, 1);
                ViewXd data_view = ViewXd(data, nrow, ncol);
                residual_view(index, 1) -= data_view(index, col) * new_para;
//                residual[row] += -data[row, col] * new_para
            }
        }

        template<unsigned int numThreads>
        __global__ void
        get_temp_a_kernel(double *out, const size_t col, double *data, double *residual, const size_t ncol,
                          const size_t nrow) {
            extern __shared__ double sPartials[];

            const int tid = threadIdx.x;

            ViewXd data_view = ViewXd(data, nrow, ncol);
            ViewXd residual_view = ViewXd(residual, nrow, ncol);

            double sum = 0.0;
            const size_t size = nrow;
            for (size_t row = tid + numThreads * blockIdx.x; row < size; row += numThreads * gridDim.x) {
                sum += data_view(row, col) * residual_view(row, 1);
            }

            sPartials[tid] = sum;
            __syncthreads();
            if (numThreads >= 1024) {
                if (tid < 512) {
                    sPartials[tid] += sPartials[tid + 512];
                }
                __syncthreads();
            }
            if (numThreads >= 512) {
                if (tid < 256) {
                    sPartials[tid] += sPartials[tid + 256];
                }
                __syncthreads();
            }
            if (numThreads >= 256) {
                if (tid < 128) {
                    sPartials[tid] += sPartials[tid + 128];
                }
                __syncthreads();
            }
            if (numThreads >= 128) {
                if (tid < 64) {
                    sPartials[tid] += sPartials[tid + 64];
                }
            }
            __syncthreads();
            // warp synchronous at the end
            if (tid < 32) {
                volatile double *wsSum = sPartials;
                if (numThreads >= 64) { wsSum[tid] += wsSum[tid + 32]; }
                if (numThreads >= 32) { wsSum[tid] += wsSum[tid + 16]; }
                if (numThreads >= 16) { wsSum[tid] += wsSum[tid + 8]; }
                if (numThreads >= 8) { wsSum[tid] += wsSum[tid + 4]; }
                if (numThreads >= 4) { wsSum[tid] += wsSum[tid + 2]; }
                if (numThreads >= 2) { wsSum[tid] += wsSum[tid + 1]; }
                if (tid == 0) {
                    out[blockIdx.x] = wsSum[0];
                }
            }

        }


        void
        get_temp_a_template(double *result_dev_ptr, const size_t col, double *data, double *residual, const size_t ncol,
                            const size_t nrow, unsigned int numBlocks, unsigned int numThreads, cudaStream_t stream) {
            switch (numThreads) {
                case (1):
                    get_temp_a_kernel<1> << < numBlocks, 1, sizeof(double), stream >> >
                                                                            (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (2):
                    get_temp_a_kernel<2> << < numBlocks, 2, sizeof(double), stream >> >
                                                                            (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (4):
                    get_temp_a_kernel<4> << < numBlocks, 4, sizeof(double), stream >> >
                                                                            (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (8):
                    get_temp_a_kernel<8> << < numBlocks, 8, sizeof(double), stream >> >
                                                                            (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (16):
                    get_temp_a_kernel<16> << < numBlocks, 16, sizeof(double), stream >> >
                                                                              (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (32):
                    get_temp_a_kernel<32> << < numBlocks, 32, sizeof(double), stream >> >
                                                                              (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (64):
                    get_temp_a_kernel<64> << < numBlocks, 64, sizeof(double), stream >> >
                                                                              (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (128):
                    get_temp_a_kernel<128> << < numBlocks, 128, sizeof(double), stream >> >
                                                                                (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (256):
                    get_temp_a_kernel<256> << < numBlocks, 256, sizeof(double), stream >> >
                                                                                (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (512):
                    get_temp_a_kernel<512> << < numBlocks, 512, sizeof(double), stream >> >
                                                                                (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;
                case (1024):
                    get_temp_a_kernel<1024> << < numBlocks, 1024, sizeof(double), stream >> >
                                                                                  (result_dev_ptr, col, data, residual, ncol, nrow);
                    break;

            }

        }


        double get_temp_a(MatrixXd &partial, const size_t col, MatrixXd &data, MatrixXd &residual, const size_t ncol,
                          const size_t nrow, unsigned int numBlocks, unsigned int numThreads, cudaStream_t stream) {

            get_temp_a_template(partial.data(), col, data.data(), residual.data(), ncol, nrow, numBlocks, numThreads,
                                stream);
            thrust::plus<double> addOp;
            cudaStreamSynchronize(stream);
            double result = thrust::reduce(partial.begin(), partial.end(), 0.0, addOp);
            return result;
        }
        //# def get_temp_a(col, data, residual, no_obs):
//#     temp_a = 0.0
//#     for row in range(0, no_obs):
//#         temp_a += data[row, col] * (residual[row])
//#     return -2.0 * temp_a / no_obs


    }////End of namespace linear_lasso_impl
}////End of namespace cuStat

