//
// Created by user on 17-11-16.
//

#include "cuMatExec.h"
#include "devScalar.h"
#include "../devSrc/cuMatrix.h"
#include <iostream>
using std::cout;
using std::endl;
#include "errorHandling.h"
namespace cuExec{

    template<typename Scalar>
    struct runner{
    public:
        static void runNN(){}
        static void runNT(){}
        static void runTN(){}
        static void runTT(){}
    };

    
    cuMatExec::cuMatExec() {
        cublasCreate(&cublasHandle);
        cublasGetStream_v2(cublasHandle, &stream);
    }

    cuMatExec::~cuMatExec() {
        cublasDestroy_v2(cublasHandle);
    }

    template<typename ScalarType, typename MatType1, typename MatType2, typename MatType3>
    void
    cuMatExec::matmulNN(ScalarType alpha, const MatType1 &lhs, const MatType2 &rhs, ScalarType beta, MatType3 &out) {
        if (lhs.cols()!=rhs.rows()){
            throw dimErrorMatMul(__FILE__, __LINE__, lhs.rows(), lhs.cols(), rhs.rows(), rhs.cols());
        }
        else {
            int m = out.rows();
            int n = out.cols();
            int k = lhs.cols();

            cublasStatus_t status;

            status = cublasDgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, lhs.getPtr(), lhs.rows(),
                                    rhs.getPtr(), rhs.rows(), &beta, out.getPtr(), out.rows());

        }
    }

    template<typename ScalarType, typename MatType1, typename MatType2, typename MatType3>
    void
    cuMatExec::matmulTN(ScalarType alpha, const MatType1 &lhs, const MatType2 &rhs, ScalarType beta, MatType3 &out) {
        int m = out.rows();
        int n = out.cols();
        int k = lhs.rows();
        cublasStatus_t status;
        status = cublasDgemm_v2(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m,n,k, &alpha, lhs.getPtr(), k,
                             rhs.getPtr(), k, &beta, out.getPtr(), m);

    }

    void cuMatExec::sync() {
        cudaStreamSynchronize(stream);
    }

    template<typename ScalarType, typename MatType1, typename MatType2, typename MatType3>
    void
    cuMatExec::matmulNT(ScalarType alpha, const MatType1 &lhs, const MatType2 &rhs, ScalarType beta, MatType3 &out) {
        int m = out.rows();
        int n = out.cols();
        int k = lhs.cols();
        cublasStatus_t status;
        status = cublasDgemm_v2(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m,n,k, &alpha, lhs.getPtr(), m,
                                rhs.getPtr(), n, &beta, out.getPtr(), m);
    }

    template<typename ScalarType, typename MatType1, typename MatType2, typename MatType3>
    void
    cuMatExec::matmulTT(ScalarType alpha, const MatType1 &lhs, const MatType2 &rhs, ScalarType beta, MatType3 &out) {
        int m = out.rows();
        int n = out.cols();
        int k = lhs.rows();
        cublasStatus_t status;
        status = cublasDgemm_v2(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, m,n,k, &alpha, lhs.getPtr(), k,
                                rhs.getPtr(), n, &beta, out.getPtr(), m);
    }

    template void
    cuMatExec::matmulNN<double, cuMatrix<double>, cuMatrix<double>, cuMatrix<double>>(double, const cuMatrix<double> &,
                                                                                      const cuMatrix<double> &, double,
                                                                                      cuMatrix<double> &);

    template void
    cuMatExec::matmulNT<double, cuMatrix<double>, cuMatrix<double>, cuMatrix<double>>(double, const cuMatrix<double> &,
                                                                                      const cuMatrix<double> &, double,
                                                                                      cuMatrix<double> &);

    template void
    cuMatExec::matmulTN<double, cuMatrix<double>, cuMatrix<double>, cuMatrix<double>>(double, const cuMatrix<double> &,
                                                                                      const cuMatrix<double> &, double,
                                                                                      cuMatrix<double> &);

    template void
    cuMatExec::matmulTT<double, cuMatrix<double>, cuMatrix<double>, cuMatrix<double>>(double, const cuMatrix<double> &,
                                                                                      const cuMatrix<double> &, double,
                                                                                      cuMatrix<double> &);

    template void
    cuMatExec::matmulNN<double, cuMatrix<double>, cuMatrix<double>, cuView<double>>(double, const cuMatrix<double> &,
                                                                                    const cuMatrix<double> &, double,
                                                                                    cuView<double> &);

    template void
    cuMatExec::matmulNT<double, cuMatrix<double>, cuMatrix<double>, cuView<double>>(double, const cuMatrix<double> &,
                                                                                    const cuMatrix<double> &, double,
                                                                                    cuView<double> &);

    template void
    cuMatExec::matmulTN<double, cuMatrix<double>, cuMatrix<double>, cuView<double>>(double, const cuMatrix<double> &,
                                                                                    const cuMatrix<double> &, double,
                                                                                    cuView<double> &);

    template void
    cuMatExec::matmulTT<double, cuMatrix<double>, cuMatrix<double>, cuView<double>>(double, const cuMatrix<double> &,
                                                                                    const cuMatrix<double> &, double,
                                                                                    cuView<double> &);

    template void
    cuMatExec::matmulNN<double, cuMatrix<double>, cuView<double>, cuMatrix<double>>(double, const cuMatrix<double> &,
                                                                                    const cuView<double> &, double,
                                                                                    cuMatrix<double> &);

    template void
    cuMatExec::matmulNT<double, cuMatrix<double>, cuView<double>, cuMatrix<double>>(double, const cuMatrix<double> &,
                                                                                    const cuView<double> &, double,
                                                                                    cuMatrix<double> &);

    template void
    cuMatExec::matmulTN<double, cuMatrix<double>, cuView<double>, cuMatrix<double>>(double, const cuMatrix<double> &,
                                                                                    const cuView<double> &, double,
                                                                                    cuMatrix<double> &);

    template void
    cuMatExec::matmulTT<double, cuMatrix<double>, cuView<double>, cuMatrix<double>>(double, const cuMatrix<double> &,
                                                                                    const cuView<double> &, double,
                                                                                    cuMatrix<double> &);

    template void
    cuMatExec::matmulNN<double, cuMatrix<double>, cuView<double>, cuView<double>>(double, const cuMatrix<double> &,
                                                                                  const cuView<double> &, double,
                                                                                  cuView<double> &);

    template void
    cuMatExec::matmulNT<double, cuMatrix<double>, cuView<double>, cuView<double>>(double, const cuMatrix<double> &,
                                                                                  const cuView<double> &, double,
                                                                                  cuView<double> &);

    template void
    cuMatExec::matmulTN<double, cuMatrix<double>, cuView<double>, cuView<double>>(double, const cuMatrix<double> &,
                                                                                  const cuView<double> &, double,
                                                                                  cuView<double> &);

    template void
    cuMatExec::matmulTT<double, cuMatrix<double>, cuView<double>, cuView<double>>(double, const cuMatrix<double> &,
                                                                                  const cuView<double> &, double,
                                                                                  cuView<double> &);

    template void
    cuMatExec::matmulNN<double, cuView<double>, cuMatrix<double>, cuMatrix<double>>(double, const cuView<double> &,
                                                                                    const cuMatrix<double> &, double,
                                                                                    cuMatrix<double> &);

    template void
    cuMatExec::matmulNT<double, cuView<double>, cuMatrix<double>, cuMatrix<double>>(double, const cuView<double> &,
                                                                                    const cuMatrix<double> &, double,
                                                                                    cuMatrix<double> &);

    template void
    cuMatExec::matmulTN<double, cuView<double>, cuMatrix<double>, cuMatrix<double>>(double, const cuView<double> &,
                                                                                    const cuMatrix<double> &, double,
                                                                                    cuMatrix<double> &);

    template void
    cuMatExec::matmulTT<double, cuView<double>, cuMatrix<double>, cuMatrix<double>>(double, const cuView<double> &,
                                                                                    const cuMatrix<double> &, double,
                                                                                    cuMatrix<double> &);

    template void
    cuMatExec::matmulNN<double, cuView<double>, cuMatrix<double>, cuView<double>>(double, const cuView<double> &,
                                                                                  const cuMatrix<double> &, double,
                                                                                  cuView<double> &);

    template void
    cuMatExec::matmulNT<double, cuView<double>, cuMatrix<double>, cuView<double>>(double, const cuView<double> &,
                                                                                  const cuMatrix<double> &, double,
                                                                                  cuView<double> &);

    template void
    cuMatExec::matmulTN<double, cuView<double>, cuMatrix<double>, cuView<double>>(double, const cuView<double> &,
                                                                                  const cuMatrix<double> &, double,
                                                                                  cuView<double> &);

    template void
    cuMatExec::matmulTT<double, cuView<double>, cuMatrix<double>, cuView<double>>(double, const cuView<double> &,
                                                                                  const cuMatrix<double> &, double,
                                                                                  cuView<double> &);

    template void
    cuMatExec::matmulNN<double, cuView<double>, cuView<double>, cuMatrix<double>>(double, const cuView<double> &,
                                                                                  const cuView<double> &, double,
                                                                                  cuMatrix<double> &);

    template void
    cuMatExec::matmulNT<double, cuView<double>, cuView<double>, cuMatrix<double>>(double, const cuView<double> &,
                                                                                  const cuView<double> &, double,
                                                                                  cuMatrix<double> &);

    template void
    cuMatExec::matmulTN<double, cuView<double>, cuView<double>, cuMatrix<double>>(double, const cuView<double> &,
                                                                                  const cuView<double> &, double,
                                                                                  cuMatrix<double> &);

    template void
    cuMatExec::matmulTT<double, cuView<double>, cuView<double>, cuMatrix<double>>(double, const cuView<double> &,
                                                                                  const cuView<double> &, double,
                                                                                  cuMatrix<double> &);

    template void
    cuMatExec::matmulNN<double, cuView<double>, cuView<double>, cuView<double>>(double, const cuView<double> &,
                                                                                const cuView<double> &, double,
                                                                                cuView<double> &);

    template void
    cuMatExec::matmulNT<double, cuView<double>, cuView<double>, cuView<double>>(double, const cuView<double> &,
                                                                                const cuView<double> &, double,
                                                                                cuView<double> &);

    template void
    cuMatExec::matmulTN<double, cuView<double>, cuView<double>, cuView<double>>(double, const cuView<double> &,
                                                                                const cuView<double> &, double,
                                                                                cuView<double> &);

    template void
    cuMatExec::matmulTT<double, cuView<double>, cuView<double>, cuView<double>>(double, const cuView<double> &,
                                                                                const cuView<double> &, double,
                                                                                cuView<double> &);




}