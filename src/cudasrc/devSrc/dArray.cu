//
// Created by user on 19-11-16.
//

#ifndef NSPCA_DARRAY_H
#define NSPCA_DARRAY_H
#include "macro.h"
#include <iostream>
using std::cout; using std::endl;

#include <cstdlib>

template<typename ScalarType, typename LHS, typename RHS>
class AddOp;

template<typename ScalarType>
class dArray;

template<typename Operation>
class Assignable;


template<typename Scalar, unsigned int numThreads,  typename Op>
__global__ void map_kernel(Scalar * data, const size_t size, const Op& op){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int position = tid +bid*numThreads;

    for (;position<size;position+=numThreads*blockDim.x){
        data[position]=op(position);
    }
};

template<typename Scalar, unsigned int numThreads>
__global__ void fill_kernel(Scalar * data, const size_t size, const Scalar value){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int position = tid +bid*numThreads;

    for (;position<size;position+=numThreads*blockDim.x){
        data[position]=value;
    }
};

template<typename Scalar, unsigned int numThreads>
__global__ void test_unified_memory(dArray<Scalar>& array, const Scalar value){
    const size_t size = array.size();
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int position = tid +bid*numThreads;

    for (;position<size;position+=numThreads*blockDim.x){
        array.write(position, value);
    }
}





template<typename ScalarType, typename Derived>
class dBase{
public:
    Derived& derived(){
        return *static_cast<Derived*>(this);
    }
    const Derived& derived() const {
        return *static_cast<const Derived*>(this);
    }

    template<typename OtherDerived>
            AddOp<ScalarType, Derived, OtherDerived> operator+(dBase<ScalarType, OtherDerived> & rhs){
                return AddOp<ScalarType, Derived, OtherDerived>(derived(), rhs.derived());
            }
};

class Managed{
    void* operator new(size_t size){
        void * ptr;
        cudaDeviceSynchronize();
        cudaMallocManaged(&ptr, size);
        return ptr;
    }
    void operator delete(void * ptr){
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};


template<typename ScalarType, typename LHS, typename RHS>
class AddOp:public dBase<ScalarType, AddOp<ScalarType, LHS, RHS>>{
public:
    LHS* lhs;
    RHS* rhs;
    using dBase<ScalarType, AddOp<ScalarType, LHS, RHS>>::operator+;

    __host__ __device__ AddOp(LHS* _lhs, RHS* _rhs):lhs(_lhs), rhs(_rhs){}

    __device__ ScalarType operator()(const size_t i) const {
        return lhs->operator()(i)+rhs->operator()(i);
    }


    Assignable<AddOp<ScalarType, LHS, RHS>> run(int numThreads, int numBlocks){
        return Assignable<AddOp<ScalarType, LHS, RHS>>(*this, numThreads, numBlocks);
    };

};

template<typename Operation>
class Assignable{
public:
    int numThreads;
    int numBlocks;
    Operation * operation;
    Assignable(Operation* _operation, int _numThreads, int _numBlocks):operation(_operation),numThreads(_numThreads), numBlocks(_numBlocks){}
};

template<typename Scalar>
class dArray:public dBase<Scalar, dArray<Scalar>>{
public:
    Scalar * data;
    size_t numElements;
    using dBase<Scalar, dArray<Scalar>>::operator+;

    dArray(size_t _size):numElements{_size}{
        cudaMalloc((void**)&data, sizeof(Scalar)*numElements);
    }

    ~dArray(){
        cudaFree(data);
    }

    __host__ __device__ size_t size() const {
        return numElements;
    }

    __host__ __device__ size_t size() {
        return numElements;
    }

    __device__ Scalar operator()(const size_t i) const{
        return data[i];
    }

    __device__ Scalar operator()(const size_t i){
        return data[i];
    }

    __device__ void write(const unsigned int i, const Scalar value){
        data[i]=value;
    }

    template<typename Op>
    dArray<Scalar> & operator=(const Assignable<Op>& assign){
        map_kernel<Scalar,1,  Op><<<assign.numBlocks, assign.numThreads>>>(data, numElements, assign.operation);
        return *this;
    }

    void fill(const Scalar value){
        fill_kernel<Scalar, 16><<<1, 16>>>(data, size(), value);
    }

    void print(){
        Scalar * host_data;
        cudaMallocHost((void**)&host_data, sizeof(Scalar)*size());
        cudaMemcpy(host_data, data, sizeof(Scalar)*size(), cudaMemcpyDeviceToHost);

        for (size_t i=0;i<size();i++){
            cout << host_data[i] << endl;
        }
        cudaFreeHost(host_data);
    }


};
//void test_unified_memory(){
//        dArray<double> a(10);
//        cudaDeviceSynchronize();
//        test_unified_memory<double, 1><<<1, 1>>>(a,1.0);
//        cudaDeviceSynchronize();
//};

namespace cuExec{
void test_fp(){
    dArray<double> a(10), b(10), c(10);
    a.fill(1.0); b.fill(1.0); c.fill(0.0);
    cudaDeviceSynchronize();
    a = (b+c).run(16, 1);
    cudaDeviceSynchronize();
    a.print();
    test_unified_memory();
}

}

#endif //NSPCA_DARRAY_H
