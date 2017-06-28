//
// Created by user on 19-11-16.
//

#ifndef NSPCA_EIGENINTEROP_H
#define NSPCA_EIGENINTEROP_H

#include <Eigen/Dense>
#include <cuda_runtime.h>
using Eigen::Matrix;
using Eigen::Dynamic;


namespace cuExec{
    template<typename HostMat, typename DeviceMat>
    void copyFromHostMat(const HostMat & hostMat, const DeviceMat & deviceMat);

    template<typename HostMat, typename DeviceMat>
    void copyToHostMat(HostMat& hostMat, const DeviceMat& deviceMat);

    template<typename ScalarType, typename cuMatType>
    void copyToEigen(Matrix<ScalarType, Dynamic, Dynamic> & eigenMatrix, const cuMatType& cuMat);

    template<typename ScalarType, typename cuMatType>
    void copyFromEigen(const Matrix<ScalarType, Dynamic, Dynamic> & eigenMatrix, cuMatType& cuMat);

    template<typename ScalarType, typename cuMatType>
    void copyToEigenAsync(Matrix<ScalarType, Dynamic, Dynamic> & eigenMatrix,  const cuMatType & cuMat, cudaStream_t stream);

    template<typename ScalarType, typename cuMatType>
    void copyFromEigenAsync(const Matrix<ScalarType ,Dynamic, Dynamic> & eigenMatrix,  cuMatType& cuMat, cudaStream_t stream);

    template<typename cuMatType>
    void inverseWithEigenTranspose(cuMatType &cuMat, cudaStream_t stream);

    template<typename ScalarType, typename hostMatType>
    Matrix<ScalarType, Dynamic, Dynamic> getEigenMat(const hostMatType& hostMat);
}


#endif //NSPCA_EIGENINTEROP_H
