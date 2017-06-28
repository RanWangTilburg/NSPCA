//
// Created by user on 15-11-16.
//

#include "cuMatExecDevice.h"
#include <curand_kernel.h>
#include <iostream>
using std::cout;
using std::endl;
namespace cuExec{

//    template<typename Scalar>
//    void cuAlgorithm<Scalar>::setConstant(Scalar *data, const size_t size, const Scalar value) {
//        thrust::device_ptr<Scalar> d_ptr(data);
//        thrust::fill_n(d_ptr, size, value);
//    }
//    template<typename Scalar>
//    thrust::device_ptr<Scalar> cuAlgorithm<Scalar>::getThrustPtr(Scalar *other) {
//        return thrust::device_ptr<Scalar>(other);
//    }

//    template<typename Scalar>
//    Scalar cuAlgorithm<Scalar>::getDeviceConstant(Scalar *dev_data) {
//        Scalar result;
//        cudaMemcpy(&result, dev_data, sizeof(Scalar), cudaMemcpyDeviceToHost);
//        return result;
//    }
//
    template<typename Scalar, unsigned int numThreads>
        __global__ void setConstantKernel(Scalar * data, const size_t size, const Scalar value){
//        printf("Here");
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        unsigned int position = tid +bid*numThreads;

        for (;position<size;position+=numThreads*blockDim.x){
            data[position]=value;
//            printf("Value is %f\n", data[position]);
//            printf("Here is the %f\n", data[position]);
        }
    };

    template<typename Scalar, unsigned int numThreads>
        __global__ void rnormKernel(Scalar * data, const size_t size, const Scalar mean, const Scalar std){
        unsigned int tid = threadIdx.x;
        unsigned int bid = blockIdx.x;
        unsigned int position = tid +bid*numThreads;
        unsigned int seed = position;

        curandState s;

        // seed a random number generator
        curand_init(seed, 0, 0, &s);
        for (;position<size;position+=numThreads*blockDim.x){
//            printf("Value is %f\n", mean+std*curand_normal(&s));
            data[position]=(Scalar)(mean+std*curand_normal(&s));
        }
    };
    template<typename Scalar, unsigned int numThreads>
    void cuAlgorithm<Scalar, numThreads>::setConstant(cudaStream_t stream, unsigned int numBlocks, Scalar *data, const size_t size,
                                                      const Scalar value) {
        setConstantKernel<Scalar, numThreads><<<numBlocks, numThreads, 0, stream>>>(data, size, value);
//        cudaDeviceSynchronize();
    }
    template<typename Scalar, unsigned int numThreads>
    void
    cuAlgorithm<Scalar, numThreads>::rnorm(cudaStream_t stream, unsigned int numBlocks, Scalar *data, const size_t size, const Scalar mean,
                       const Scalar std) {
        rnormKernel<Scalar, numThreads><<<numBlocks, numThreads, 0, stream>>>(data, size, mean, std);
    }

    template<typename Scalar, unsigned  int numThreads>
    void
    cuAlgorithm<Scalar, numThreads>::rnorm(unsigned int numBlocks, Scalar *data, const size_t size, const Scalar mean, const Scalar std) {
        rnormKernel<Scalar, numThreads><<<numBlocks, numThreads>>>(data, size, mean, std);
    }

    template<typename Scalar, unsigned int numThreads>
    void cuAlgorithm<Scalar, numThreads>::setConstant(unsigned int numBlocks, Scalar *data, const size_t size, const Scalar value) {
//        cout << "Value is " << value << endl;
//        cout << "Number of Blocks is " << numBlocks << endl;
//        cout << "Number of Threads is " << numThreads << endl;
        setConstantKernel<Scalar, numThreads><<<numBlocks, numThreads>>>(data, size, value);

//        cudaDeviceSynchronize();
    }
}