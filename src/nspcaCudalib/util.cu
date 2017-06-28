
#include "view.h"
#include "macro.h"
#include <cuda_runtime.h>
namespace NSPCA{
    //! \brief
    template<unsigned int numThreads>
    CUDA_KERNEL void check_res_gpu_kernel(double *matP, int *res, const size_t p, const size_t P) {
        const int tid = threadIdx.x;
        const int offset = numThreads * blockIdx.x;

        for (int index = tid + offset; index < p * P; index += numThreads * gridDim.x) {
            if ((res[index] == -1 && matP[index] > 0) || (res[index] == 1 && matP[index] < 0)) {
                matP[index] = -matP[index];
            } else if (res[index] == 0) {
                matP[index] = 0;
            }

        }

    }

    void check_res_gpu(double *matP, int *res, const size_t p, const size_t P, const unsigned int numThreads,
                       const unsigned int numBlocks, cudaStream_t stream) {
        switch (numThreads) {
            case (1):
                check_res_gpu_kernel<1> << < numBlocks, 1, 0, stream >> > (matP, res, p, P);
                break;
            case (2):
                check_res_gpu_kernel<2> << < numBlocks, 2, 0, stream >> > (matP, res, p, P);
                break;
            case (4):
                check_res_gpu_kernel<4> << < numBlocks, 4, 0, stream >> > (matP, res, p, P);
                break;
            case (8):
                check_res_gpu_kernel<8> << < numBlocks, 8, 0, stream >> > (matP, res, p, P);
                break;
            case (16):
                check_res_gpu_kernel<16> << < numBlocks, 16, 0, stream >> > (matP, res, p, P);
                break;
            case (32):
                check_res_gpu_kernel<32> << < numBlocks, 32, 0, stream >> > (matP, res, p, P);
                break;
            case (64):
                check_res_gpu_kernel<64> << < numBlocks, 64, 0, stream >> > (matP, res, p, P);
                break;
            case (128):
                check_res_gpu_kernel<128> << < numBlocks, 128, 0, stream >> > (matP, res, p, P);
                break;
            case (256):
                check_res_gpu_kernel<256> << < numBlocks, 256, 0, stream >> > (matP, res, p, P);
                break;
            case (512):
                check_res_gpu_kernel<512> << < numBlocks, 512, 0, stream >> > (matP, res, p, P);
                break;
            case (1024):
                check_res_gpu_kernel<1024> << < numBlocks, 1024, 0, stream >> > (matP, res, p, P);
                break;
        }
    }

    template<typename ScalarType, unsigned int numThreads>
    CUDA_KERNEL void frobenius_kernel(ScalarType *result_dev_ptr, const ScalarType *input1_devptr, const size_t size) {
        extern __shared__
                ScalarType
        sPartials[];
        const int tid = threadIdx.x;
        double sum = 0.0;
        for (size_t i = tid + numThreads * blockIdx.x; i < size; i += numThreads * gridDim.x) {
            ScalarType temp = input1_devptr[i];
//            printf("This is from thread %d with value %0.12lf \n", i, temp);
            sum += temp * temp;
        }
        sPartials[tid] = sum;
//        printf("This is thread %d, block %d, and value %0.12lf after calculating the sum \n", tid,  blockIdx.x, sPartials[tid]);
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
            __syncthreads();
        }
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
                result_dev_ptr[blockIdx.x] = wsSum[0];
            }
        }

    }


    template<typename ScalarType, unsigned int numThreads>
    CUDA_KERNEL void
    l2_diff_kernel(ScalarType *result_dev_ptr, const ScalarType *input1_devptr, const ScalarType *input2_devptr,
                   const size_t size) {
        extern __shared__
                ScalarType
        sPartials[];
        const unsigned int tid = threadIdx.x;
        double sum = 0.0;
        for (size_t i = tid + numThreads * blockIdx.x; i < size; i += numThreads * gridDim.x) {
            ScalarType temp = input1_devptr[i] - input2_devptr[i];
            sum += temp * temp;
//        printf("from Thread %d the value is %0.12lf \n", tid, temp);

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
                volatile ScalarType *wsSum = sPartials;
                if (numThreads >= 64) { wsSum[tid] += wsSum[tid + 32]; }
                if (numThreads >= 32) { wsSum[tid] += wsSum[tid + 16]; }
                if (numThreads >= 16) { wsSum[tid] += wsSum[tid + 8]; }
                if (numThreads >= 8) { wsSum[tid] += wsSum[tid + 4]; }
                if (numThreads >= 4) { wsSum[tid] += wsSum[tid + 2]; }
                if (numThreads >= 2) { wsSum[tid] += wsSum[tid + 1]; }
                if (tid == 0) {
                    result_dev_ptr[blockIdx.x] = wsSum[0];
                }
            }
        }
    }
}