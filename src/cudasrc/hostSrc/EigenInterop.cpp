//
// Created by user on 19-11-16.
//

#include "EigenInterop.h"
#include "../devSrc/cuMatrix.h"
#include "../devSrc/cuView.h"
namespace cuExec{
    template<typename ScalarType,typename cuMatType>
    void copyToEigen(Matrix<ScalarType, Dynamic, Dynamic> &eigenMatrix, const cuMatType &cuMat) {
        ScalarType * hostPtr = (ScalarType*)eigenMatrix.data();
       const ScalarType * devPtr = cuMat.getPtr();

        long nRow = eigenMatrix.rows();
        long nCol = eigenMatrix.cols();

        cudaMemcpy(hostPtr, devPtr, sizeof(ScalarType)*nRow*nCol, cudaMemcpyDeviceToHost);

    }

    template<typename ScalarType, typename cuMatType>
    void copyFromEigen(const Matrix<ScalarType, Dynamic, Dynamic> &eigenMatrix, cuMatType &cuMat) {
        ScalarType * hostPtr = (ScalarType*)eigenMatrix.data();
        ScalarType * devPtr = cuMat.getPtr();

        long nRow = eigenMatrix.rows();
        long nCol = eigenMatrix.cols();

        cudaMemcpy(devPtr, hostPtr, sizeof(ScalarType)*nRow*nCol, cudaMemcpyHostToDevice);
    }

    template<typename ScalarType, typename cuMatType>
    void copyToEigenAsync(Matrix<ScalarType, Dynamic, Dynamic> & eigenMatrix,  const cuMatType & cuMat, cudaStream_t stream){
        ScalarType * hostPtr = (ScalarType*)eigenMatrix.data();
        const ScalarType * devPtr = cuMat.getPtr();

        long nRow = eigenMatrix.rows();
        long nCol = eigenMatrix.cols();

        cudaMemcpyAsync(hostPtr, devPtr, sizeof(ScalarType)*nRow*nCol, cudaMemcpyDeviceToHost, stream);
    };

    template<typename ScalarType, typename cuMatType>
    void copyFromEigenAsync(const Matrix<ScalarType, Dynamic, Dynamic> & eigenMatrix, cuMatType& cuMat, cudaStream_t stream){
        ScalarType * hostPtr = (ScalarType*)eigenMatrix.data();
        ScalarType * devPtr = cuMat.getPtr();

        long nRow = eigenMatrix.rows();
        long nCol = eigenMatrix.cols();

        cudaMemcpyAsync(devPtr, hostPtr, sizeof(ScalarType)*nRow*nCol, cudaMemcpyHostToDevice, stream);
    }

    template<typename cuMatType>
    void inverseWithEigenTranspose(cuMatType &cuMat, cudaStream_t stream) {
        assert(cuMat.rows()==cuMat.cols());
        using type = typename cuMatType::type;
        Eigen::Matrix<type, Dynamic, Dynamic> hostMat = Eigen::Matrix<type, Dynamic, Dynamic>::Constant(cuMat.rows(), cuMat.cols(), 0);
        copyToEigenAsync(hostMat, cuMat, stream);
        cudaStreamSynchronize(stream);
        hostMat = hostMat.transpose().inverse();
        copyFromEigenAsync(hostMat, cuMat, stream);
    }

    template<typename ScalarType, typename hostMatType>
    Matrix<ScalarType, Dynamic, Dynamic> getEigenMat(const hostMatType &hostMat) {
        Matrix<ScalarType, Dynamic, Dynamic> EigenMat(hostMat.rows(), hostMat.cols());

        for (size_t i=0;i<hostMat.rows();i++){
            for (size_t j=0;j<hostMat.cols();j++){
                EigenMat(i,j)=(ScalarType)hostMat(i,j);
            }
        }
        return EigenMat;
    }

    template<typename HostMat, typename DeviceMat>
    void copyFromHostMat(const HostMat &hostMat, const DeviceMat &deviceMat) {
        static_assert(HostMat::type==DeviceMat::type);
        using ScalarType = typename HostMat::type;
    };


    template void inverseWithEigenTranspose<cuMatrix<double>>(cuMatrix<double>&, cudaStream_t);
    template void copyToEigen<double, cuMatrix<double>>(Matrix<double, Dynamic, Dynamic>&, const cuMatrix<double>& );
    template void copyFromEigen<double, cuMatrix<double>>(const Matrix<double, Dynamic,Dynamic>&, cuMatrix<double>&);
    template void copyToEigenAsync<double, cuMatrix<double>>(Matrix<double, Dynamic, Dynamic>&, const cuMatrix<double>&, cudaStream_t);
    template void copyFromEigenAsync<double, cuMatrix<double>>(const Matrix<double, Dynamic, Dynamic>&, cuMatrix<double>&, cudaStream_t);
    template void copyFromEigenAsync<int, cuMatrix<int>>(const Matrix<int, Dynamic, Dynamic>&, cuMatrix<int> &, cudaStream_t);
    template Matrix<double, Dynamic, Dynamic> getEigenMat<double, cuView<int>>(const cuView<int>&);
//    void forced_init_eigen_interop(){
//        MatrixXd a_MatrixXd=MatrixXd::Constant(10,10,0);
//        MatrixXi a_MatrixXi=MatrixXi::Constant(10,10,0);
//        MatrixXf a_MatrixXf=MatrixXf::Constant(10,10,0);
//        cuMatrix<double> cuMat_double(10,10,0);
//        cuView<double> cuView_double(cuMat_double.getPtr(), 10, 10);
//        cuMatrix<int> cuMat_int(10,10,0);
//        cuView<int> cuView_int(cuMat_int.getPtr(), 10, 10);
//        cuMatrix<float> cuMat_float(10,10,0);
//        cuView<float> cuView_float(cuMat_float.getPtr(), 10, 10);
//        copyFromEigen(a_MatrixXd, cuMat_double);
//        copyFromEigen(a_MatrixXd, cuView_double);
//        copyToEigen(a_MatrixXd, cuMat_double);
//        copyToEigen(a_MatrixXd, cuView_double);
//        copyFromEigen(a_MatrixXi, cuMat_int);
//        copyFromEigen(a_MatrixXi, cuView_int);
//        copyToEigen(a_MatrixXi, cuMat_int);
//        copyToEigen(a_MatrixXi, cuView_int);
//        copyFromEigen(a_MatrixXf, cuMat_float);
//        copyFromEigen(a_MatrixXf, cuView_float);
//        copyToEigen(a_MatrixXf, cuMat_float);
//        copyToEigen(a_MatrixXf, cuView_float);
//
//    }
}