//
// Created by user on 15-11-16.
//

#ifndef NSPCA_CUALGORITHM_H
#define NSPCA_CUALGORITHM_H
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>
#include <cstdlib>
namespace cuExec {
    template<typename Scalar, unsigned int numThreads>
    class cuAlgorithm {
    public:
        static void setConstant(cudaStream_t stream, unsigned int numBlocks, Scalar *data, const size_t size, const Scalar value);
        static void setConstant(unsigned int numBlocks, Scalar * data, const size_t size, const Scalar Value);
        static void rnorm(cudaStream_t stream, unsigned int numBlocks, Scalar * data, const size_t size, const Scalar mean, const Scalar std);
        static void rnorm(unsigned int numBlocks, Scalar * data, const size_t size, const Scalar mean, const Scalar std);
    };


    template class cuAlgorithm<int, 1>;
    template class cuAlgorithm<int, 2>;
    template class cuAlgorithm<int, 4>;
    template class cuAlgorithm<int, 8>;
    template class cuAlgorithm<int, 16>;
    template class cuAlgorithm<int, 32>;
    template class cuAlgorithm<int, 64>;
    template class cuAlgorithm<int, 128>;
    template class cuAlgorithm<int, 256>;
    template class cuAlgorithm<int, 516>;
    template class cuAlgorithm<int, 1024>;
    template class cuAlgorithm<double, 1>;
    template class cuAlgorithm<double, 2>;
    template class cuAlgorithm<double, 4>;
    template class cuAlgorithm<double, 8>;
    template class cuAlgorithm<double, 16>;
    template class cuAlgorithm<double, 32>;
    template class cuAlgorithm<double, 64>;
    template class cuAlgorithm<double, 128>;
    template class cuAlgorithm<double, 256>;
    template class cuAlgorithm<double, 516>;
    template class cuAlgorithm<double, 1024>;
    template class cuAlgorithm<float, 1>;
    template class cuAlgorithm<float, 2>;
    template class cuAlgorithm<float, 4>;
    template class cuAlgorithm<float, 8>;
    template class cuAlgorithm<float, 16>;
    template class cuAlgorithm<float, 32>;
    template class cuAlgorithm<float, 64>;
    template class cuAlgorithm<float, 128>;
    template class cuAlgorithm<float, 256>;
    template class cuAlgorithm<float, 516>;
    template class cuAlgorithm<float, 1024>;

}

#endif //NSPCA_CUALGORITHM_H
