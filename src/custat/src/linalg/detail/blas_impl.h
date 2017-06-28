#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../util/exception.h"
#include "../../util/matrix.h"

////Contains implementation of the blas-3 routines

namespace cuStat{
    namespace internal{
        ////This class captures the details of linSolver
        ////Also select the right routine based on the correct type
        template<typename Scalar>
        class linSolver_impl {
        };

        ////Specialization for type "float"
        template<>
        class linSolver_impl<float> {
        public:
            template<typename MatType1, typename MatType2, typename MatType3>
            static void
            matmul_nn(cublasHandle_t cublas_handle, float alpha, MatType1 &A, MatType2 &B,
                      float beta, MatType3 &X) throw(dimMisMatch, cuBlasError);

            template<typename MatType1, typename MatType2, typename MatType3>
            static void
            matmul_nt(cublasHandle_t cublas_handle, float alpha, MatType1 &A, MatType2 &B,
                      float beta, MatType3 &X) throw(dimMisMatch, cuBlasError);

            template<typename MatType1, typename MatType2, typename MatType3>
            static void
            matmul_tn(cublasHandle_t cublas_handle, float alpha, MatType1 &A, MatType2 &B,
                      float beta, MatType3 &X)
            throw(dimMisMatch, cuBlasError);

            template<typename MatType1, typename MatType2, typename MatType3>
            static void
            matmul_tt(cublasHandle_t cublas_handle, float alpha, MatType1 &A, MatType2 &B,
                      float beta, MatType3 &X)
            throw(dimMisMatch, cuBlasError);
        };

        template<typename MatType1, typename MatType2, typename MatType3>
        void linSolver_impl<float>::matmul_nn(cublasHandle_t cublas_handle, float alpha, MatType1 &A,
                                              MatType2 &B, float beta,
                                              MatType3 &X) throw(dimMisMatch, cuBlasError) {
            try {
                if (X.rows() != A.rows() || X.cols() != B.cols() || A.cols() != B.rows()) {
                    throw dimMisMatch("Dimension mismatch in matrix product");
                } else {
                    size_t m = X.rows();
                    size_t n = X.cols();
                    size_t k = A.cols();


                    handle_cublas_error(cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                       m, n, k, &alpha, A.data(), A.rows(), B.data(), B.rows(), &beta,
                                                       X.data(), X.rows()));
                }
            }
            catch (dimMisMatch d) {
                throw d;
            }
            catch (cuBlasError d) {
                throw d;
            }
        }

        template<typename MatType1, typename MatType2, typename MatType3>
        void linSolver_impl<float>::matmul_nt(cublasHandle_t cublas_handle, float alpha, MatType1 &A,
                                              MatType2 &B, float beta,
                                              MatType3 &X) throw(dimMisMatch, cuBlasError) {
            try {
                if (X.rows() != A.rows() || X.cols() != B.rows() || A.cols() != B.cols()) {
                    throw dimMisMatch("Dimension mismatch in matrix product");
                } else {
                    size_t m = X.rows();
                    size_t n = X.cols();
                    size_t k = A.cols();

                    handle_cublas_error(cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                       m, n, k, &alpha, A.data(), A.rows(), B.data(), n, &beta,
                                                       X.data(), X.rows()));
                }
            }
            catch (dimMisMatch d) {
                throw d;
            }
            catch (cuBlasError d) {
                throw d;
            }
        }

        template<typename MatType1, typename MatType2, typename MatType3>
        void
        linSolver_impl<float>::matmul_tn(cublasHandle_t cublas_handle, float alpha, MatType1 &A,
                                         MatType2 &B, float beta,
                                         MatType3 &X) throw(dimMisMatch, cuBlasError) {
            try {
                if (X.rows() != A.cols() || X.cols() != B.cols() || A.rows() != B.rows()) {
                    throw dimMisMatch("Dimension mismatch in matrix product");
                } else {
                    size_t m = X.rows();
                    size_t n = X.cols();
                    size_t k = A.rows();

                    handle_cublas_error(cublasSgemm_v2(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                       m, n, k, &alpha, A.data(), k, B.data(), B.rows(), &beta,
                                                       X.data(), X.rows()));
                }
            }
            catch (dimMisMatch d) {
                throw d;
            }
            catch (cuBlasError d) {
                throw d;
            }

        }

        template<typename MatType1, typename MatType2, typename MatType3>
        void linSolver_impl<float>::matmul_tt(cublasHandle_t cublas_handle, float alpha, MatType1 &A,
                                              MatType2 &B, float beta,
                                              MatType3 &X) throw(dimMisMatch, cuBlasError) {
            try {
                if (X.rows() != A.cols() || X.cols() != B.rows() || A.rows() != B.cols()) {
                    throw dimMisMatch("Dimension mismatch in matrix product");
                } else {
                    size_t m = X.rows();
                    size_t n = X.cols();
                    size_t k = A.rows();

                    handle_cublas_error(cublasSgemm_v2(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                       m, n, k, &alpha, A.data(), k, B.data(), n, &beta,
                                                       X.data(), X.rows()));

                }
            }
            catch (dimMisMatch d) {
                throw d;
            }
            catch (cuBlasError d) {
                throw d;
            }
        }


        ////Specialization for type "double"
        template<>
        class linSolver_impl<double> {
        public:
            template<typename MatType1, typename MatType2, typename MatType3>
            static void
            matmul_nn(cublasHandle_t cublas_handle, double alpha, MatType1 &A, MatType2 &B,
                      double beta, MatType3 &X)
            throw(dimMisMatch, cuBlasError);

            template<typename MatType1, typename MatType2, typename MatType3>
            static void
            matmul_nt(cublasHandle_t cublas_handle, double alpha, MatType1 &A, MatType2 &B,
                      double beta, MatType3 &X)
            throw(dimMisMatch, cuBlasError);


            template<typename MatType1, typename MatType2, typename MatType3>
            static void
            matmul_tn(cublasHandle_t cublas_handle, double alpha, MatType1 &A, MatType2 &B,
                      double beta, MatType3 &X)
            throw(dimMisMatch, cuBlasError);

            template<typename MatType1, typename MatType2, typename MatType3>
            static void
            matmul_tt(cublasHandle_t cublas_handle, double alpha, MatType1 &A, MatType2 &B,
                      double beta, MatType3 &X)
            throw(dimMisMatch, cuBlasError);
        };

        template<typename MatType1, typename MatType2, typename MatType3>
        void
        linSolver_impl<double>::matmul_nn(cublasHandle_t cublas_handle,
                                          double alpha, MatType1 &A, MatType2 &B, double beta,
                                          MatType3 &X) throw(dimMisMatch, cuBlasError) {
            try {
                if (X.rows() != A.rows() || X.cols() != B.cols() || A.cols() != B.rows()) {
                    throw dimMisMatch("Dimension mismatch in matrix product");
                } else {
                    size_t m = X.rows();
                    size_t n = X.cols();
                    size_t k = A.cols();

                    handle_cublas_error(cublasDgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                       m, n, k, &alpha, A.data(), A.rows(), B.data(), B.rows(), &beta,
                                                       X.data(), X.rows()));
                }
            }
            catch (dimMisMatch d) {
                throw d;
            }
            catch (cuBlasError d) {
                throw d;
            }
        }

        template<typename MatType1, typename MatType2, typename MatType3>
        void linSolver_impl<double>::matmul_nt(cublasHandle_t cublas_handle, double alpha, MatType1 &A,
                                               MatType2 &B, double beta,
                                               MatType3 &X) throw(dimMisMatch, cuBlasError) {
            try {
                if (X.rows() != A.rows() || X.cols() != B.rows() || A.cols() != B.cols()) {
                    throw dimMisMatch("Dimension mismatch in matrix product");
                } else {
                    size_t m = X.rows();
                    size_t n = X.cols();
                    size_t k = A.cols();

                    handle_cublas_error(cublasDgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                       m, n, k, &alpha, A.data(), A.rows(), B.data(), n, &beta,
                                                       X.data(), X.rows()));
                }
            }
            catch (dimMisMatch d) {
                throw d;
            }
            catch (cuBlasError d) {
                throw d;
            }
        }

        template<typename MatType1, typename MatType2, typename MatType3>
        void linSolver_impl<double>::matmul_tn(cublasHandle_t cublas_handle, double alpha, MatType1 &A,
                                               MatType2 &B, double beta,
                                               MatType3 &X) throw(dimMisMatch, cuBlasError) {
            try {
                if (X.rows() != A.cols() || X.cols() != B.cols() || A.rows() != B.rows()) {
                    throw dimMisMatch("Dimension mismatch in matrix product");
                } else {
                    size_t m = X.rows();
                    size_t n = X.cols();
                    size_t k = A.rows();

                    handle_cublas_error(cublasDgemm_v2(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                       m, n, k, &alpha, A.data(), k, B.data(), B.rows(), &beta,
                                                       X.data(), X.rows()));
                }
            }
            catch (dimMisMatch d) {
                throw d;
            }
            catch (cuBlasError d) {
                throw d;
            }
        }

        template<typename MatType1, typename MatType2, typename MatType3>
        void linSolver_impl<double>::matmul_tt(cublasHandle_t cublas_handle, double alpha, MatType1 &A,
                                               MatType2 &B, double beta, MatType3 &X) throw(dimMisMatch, cuBlasError) {
            try {
                if (X.rows() != A.cols() || X.cols() != B.rows() || A.rows() != B.cols()) {
                    throw dimMisMatch("Dimension mismatch in matrix product");
                } else {
                    size_t m = X.rows();
                    size_t n = X.cols();
                    size_t k = A.rows();

                    handle_cublas_error(cublasDgemm_v2(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                                       m, n, k, &alpha, A.data(), k, B.data(), n, &beta,
                                                       X.data(), X.rows()));

                }
            }
            catch (dimMisMatch d) {
                throw d;
            }
            catch (cuBlasError d) {
                throw d;
            }
        }


    }////End of internal
}////End of cuStat