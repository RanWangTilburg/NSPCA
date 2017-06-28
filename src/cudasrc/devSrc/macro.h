//
// Created by user on 19-11-16.
//

#ifndef NSPCA_MACRO_H
#define NSPCA_MACRO_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_HOST __host__
#else
#define CUDA_HOST
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

#ifdef __CUDACC__
#define CUDA_KERNEL __global__
#else
#define CUDA_KERNEL
#endif
#include <cuda_runtime.h>

#endif //NSPCA_MACRO_H
