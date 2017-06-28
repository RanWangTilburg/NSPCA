//
// Created by user on 17-11-16.
//

#include "devScalar.h"

namespace cuExec{
    template<typename Scalar>
    devScalar<Scalar>::devScalar(const Scalar _host_value) {
        host_value = _host_value;
        cudaMalloc((void**)&dev_ptr, sizeof(Scalar));
        cudaMemcpy(dev_ptr, &host_value, sizeof(Scalar), cudaMemcpyHostToDevice);
    }

    template<typename Scalar>
    devScalar<Scalar>::~devScalar() {
        cudaFree(dev_ptr);
    }
    template<typename Scalar>
    void devScalar<Scalar>::syncDeviceToHost() {
        cudaMemcpy(&host_value, dev_ptr, sizeof(Scalar), cudaMemcpyDeviceToHost);
    }

    template<typename Scalar>
    void devScalar<Scalar>::syncHostToDevice() {
        cudaMemcpy(dev_ptr, &host_value, sizeof(Scalar), cudaMemcpyHostToDevice);
    }

    template<typename Scalar>
    Scalar devScalar<Scalar>::getHostValue() {
        return host_value;
    }
    template<typename Scalar>
    Scalar *devScalar<Scalar>::getPtr() {
        return dev_ptr;
    }


}