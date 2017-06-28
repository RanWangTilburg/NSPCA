//
// Created by user on 18-11-16.
//

#include "devWorkSpace.h"
#include <iostream>
using std::cout;
using std::endl;
namespace cuExec{

    template<typename Scalar>
    devWorkSpace<Scalar>::devWorkSpace(const size_t default_lsize, const size_t default_rsize) {
        cudaMalloc((void**)&lwork, sizeof(Scalar)*default_lsize);
        lsize=default_lsize;
        cudaMalloc((void**)&rwork, sizeof(Scalar)*default_rsize);
        rsize= default_rsize;
    }


    template<typename Scalar>
    devWorkSpace<Scalar>::~devWorkSpace() {
        cudaFree(lwork);
        cudaFree(rwork);
    }

    template<typename Scalar>
    void devWorkSpace<Scalar>::resizeLWork(const size_t newSize) {
        if ((int)newSize>lsize){
//            cout << "New Size is " << newSize << endl;
//            cout << "Old Lsize is " << lsize << endl;
//            cout << "Resizing l size" << endl;
            cudaFree(lwork);
            cudaMalloc((void **)&lwork, sizeof(Scalar)*newSize);
            lsize=newSize;
        }
    }

    template<typename Scalar>
    Scalar *devWorkSpace<Scalar>::getLWork() const {
        return lwork;
    }

    template<typename Scalar>
    size_t devWorkSpace<Scalar>::getLSize() const {
        return lsize;
    }

    template<typename Scalar>
    Scalar *devWorkSpace<Scalar>::getRWork() const {
        return rwork;
    }

    template<typename Scalar>
    size_t devWorkSpace<Scalar>::getRSize() const {
        return rsize;
    }

    template<typename Scalar>
    void devWorkSpace<Scalar>::resizeRWork(const size_t newSize) {
        if (newSize>rsize){
//            cout << "Resizing r size" << endl;
            cudaFree(rwork);
            cudaMalloc((void**)&rwork, sizeof(Scalar)*newSize);
            rsize=newSize;

        }

    }


}