#pragma once

#include "cuda_runtime.h"
#include "macro"
//#include "../../../../../../../../usr/local/cuda-8.0/include/cuda_runtime_api.h"
//#include "../../../../../../../../usr/local/cuda-8.0/include/curand_mtgp32_kernel.h"
//#include "../../../../../../../../usr/local/cuda-8.0/include/device_launch_parameters.h"


namespace cuStat {
    namespace internal {
        template<typename LHS, typename RHS>
        CUDA_KERNEL void assign_matrix_kernel(LHS lhs, RHS rhs);

        //////////////////////////////////////////////////////////
        ////Implementations///////////////////////////////////////
        //////////////////////////////////////////////////////////
        template<typename LHS, typename RHS>
        CUDA_KERNEL void assign_matrix_kernel(LHS lhs, RHS rhs) {
//            printf("This is reached\n");
            unsigned int tid = threadIdx.x;
            unsigned int bid = blockIdx.x;
            unsigned int numThreads = blockDim.x;
            unsigned int numBlocks = gridDim.x;

            unsigned int stride = numThreads * numBlocks;
            unsigned int index = tid + bid * numThreads;
            unsigned int size = lhs.size();
//            printf("This is thread %d block %d\n", tid, bid);
            for (;index<size;index+=stride){
//                printf("The value is %f\n", rhs[index]);
                lhs[index] = rhs[index];
            }

        };
    }////End of namespace internal
}////End of namespace cuStat