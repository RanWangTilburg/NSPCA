#pragma once

#include <cstdlib>


#include <curand_kernel.h>
#include <cassert>
////Kernels related to matrix and view classes

#include "macro"
#include "exception.h"

namespace cuStat {


    namespace internal {

        template<typename Scalar, unsigned int numThreads>
        CUDA_KERNEL void
        rand_kernel(Scalar *dst, const size_t size, const Scalar lower, const Scalar upper, const unsigned int seed) {
            unsigned int tid = threadIdx.x;
            unsigned int bid = blockIdx.x;
            unsigned int position = tid + bid * numThreads;
            unsigned int seed_individual = position + seed;

            curandState s;

            // seed a random number generator

            Scalar diff = upper - lower;
            curand_init(seed_individual, 0, 0, &s);
            for (; position < size; position += numThreads * gridDim.x) {

//            printf("Value is %f\n", mean+std*curand_normal(&s));
                dst[position] = (Scalar) (diff * curand_uniform(&s) + lower);
//                dst[position ]=1;
            }
        };

        template<typename Scalar>
        void rand_func(Scalar *dst, const size_t size, const Scalar lower, const Scalar upper, const unsigned int seed,
                       cudaStream_t stream, const unsigned int numThreads,
                       const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError) {
            assert(lower < upper);
            try {
                handle_num_threads_error(numThreads);
                switch (numThreads) {
                    case (1):
                        (rand_kernel<Scalar, 1> << < numBlocks, 1, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (2):
                        (rand_kernel<Scalar, 2> << < numBlocks, 2, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (4):
                        (rand_kernel<Scalar, 4> << < numBlocks, 4, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (8):
                        (rand_kernel<Scalar, 8> << < numBlocks, 8, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (16):
                        (rand_kernel<Scalar, 16> << < numBlocks, 16, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (32):
                        (rand_kernel<Scalar, 32> << < numBlocks, 32, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (64):
                        (rand_kernel<Scalar, 64> << < numBlocks, 64, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (128):
                        (rand_kernel<Scalar, 128> << < numBlocks, 128, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (256):
                        (rand_kernel<Scalar, 256> << < numBlocks, 256, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (512):
                        (rand_kernel<Scalar, 512> << < numBlocks, 512, 0, stream >> > (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (1024):
                        (rand_kernel<Scalar, 1024> << < numBlocks, 1024, 0, stream >> >
                                                                            (dst, size, lower, upper, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;

                }
            }
            catch (cudaRunTimeError e) {
                throw e;
            }
        }


        template<typename Scalar, unsigned int numThreads>
        CUDA_KERNEL void fill_with_constant_kernel(Scalar *dst, const size_t size, const Scalar value) {
            unsigned int tid = threadIdx.x;
            unsigned int bid = blockIdx.x * numThreads;;

            unsigned int position = tid + bid;
            for (; position < size; position += gridDim.x * numThreads) {
                dst[position] = value;
            }
        };


        template<typename Scalar>
        void fill_with_constant(Scalar *dst, const size_t size, const Scalar value, cudaStream_t stream,
                                const unsigned int numThreads,
                                const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError) {
            try {
                handle_num_threads_error(numThreads);
                switch (numThreads) {
                    case (1):
                        fill_with_constant_kernel<Scalar, 1> << < numBlocks, 1, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (2):
                        fill_with_constant_kernel<Scalar, 2> << < numBlocks, 2, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (4):
                        fill_with_constant_kernel<Scalar, 4> << < numBlocks, 4, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (8):
                        fill_with_constant_kernel<Scalar, 8> << < numBlocks, 8, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (16):
                        fill_with_constant_kernel<Scalar, 16> << < numBlocks, 16, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (32):
                        fill_with_constant_kernel<Scalar, 32> << < numBlocks, 32, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (64):
                        fill_with_constant_kernel<Scalar, 64> << < numBlocks, 64, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (128):
                        fill_with_constant_kernel<Scalar, 128> << < numBlocks, 128, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (256):
                        fill_with_constant_kernel<Scalar, 256> << < numBlocks, 256, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (512):
                        fill_with_constant_kernel<Scalar, 512> << < numBlocks, 512, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (1024):
                        fill_with_constant_kernel<Scalar, 1024> << < numBlocks, 1024, 0, stream >> > (dst, size, value);
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;

                }

            }
            catch (cudaRunTimeError e) {
                throw e;
            }
        }


        ////kenrel that fills in random values from normal distribution
        template<typename Scalar, unsigned int numThreads>
        CUDA_KERNEL void
        randn_kernel(Scalar *dst, const size_t size, const Scalar mean, const Scalar std, const unsigned int seed) {
            unsigned int tid = threadIdx.x;
            unsigned int bid = blockIdx.x;
            unsigned int position = tid + bid * numThreads;
            unsigned int seed_individual = position + seed;

            curandState s;

            // seed a random number generator
            curand_init(seed_individual, 0, 0, &s);
            for (; position < size; position += numThreads * gridDim.x) {
//            printf("Value is %f\n", mean+std*curand_normal(&s));
                dst[position] = (Scalar) (mean + std * curand_normal(&s));
            }
        }

        template<typename Scalar>
        void randn_func(Scalar *dst, const size_t size, const Scalar mean, const Scalar std, const unsigned int seed,
                        cudaStream_t stream, const unsigned int numThreads,
                        const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError) {

            try {
                handle_num_threads_error(numThreads);
                switch (numThreads) {
                    case (1):
                        (randn_kernel<Scalar, 1> << < numBlocks, 1, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (2):
                        (randn_kernel<Scalar, 2> << < numBlocks, 2, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (4):
                        (randn_kernel<Scalar, 4> << < numBlocks, 4, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (8):
                        (randn_kernel<Scalar, 8> << < numBlocks, 8, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (16):
                        (randn_kernel<Scalar, 16> << < numBlocks, 16, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (32):
                        (randn_kernel<Scalar, 32> << < numBlocks, 32, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (64):
                        (randn_kernel<Scalar, 64> << < numBlocks, 64, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (128):
                        (randn_kernel<Scalar, 128> << < numBlocks, 128, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (256):
                        (randn_kernel<Scalar, 256> << < numBlocks, 256, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (512):
                        (randn_kernel<Scalar, 512> << < numBlocks, 512, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;
                    case (1024):
                        (randn_kernel<Scalar, 1024> << < numBlocks, 1024, 0, stream >> > (dst, size, mean, std, seed));
                        handle_cuda_runtime_error(cudaGetLastError());
                        break;


                }
            }
            catch (cudaRunTimeError e) {
                throw e;
            }
        }


    }////End of internal namespace
}////End of cuStat namespace