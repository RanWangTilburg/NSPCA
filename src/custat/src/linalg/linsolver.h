#pragma once

#include "../util/workspace.h"
#include "../util/matrix.h"

////Class which holds an instance that performs matrix algebra
////Note that due to the requirement of initialization
////The methods are in general not static
////All the matrices are assumed to be stored in column major (a.k.a. Fortran order)



#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include "detail/svd_impl.h"
#include "detail/blas_impl.h"
////A short hand macro for throwing verious exceptions
#ifndef THROW_LIN_ERROR
#define THROW_LIN_ERROR \
         catch (cudaRunTimeError e){ \
                throw e; \
        } \
        catch (cuSolverError e){ \
            throw e; \
        } \
        catch (cuBlasError e){ \
            throw e; \
        }
#endif


namespace cuStat {
    class linSolver {
    private:
        WorkSpace *workSpace;

        cudaStream_t stream;
        cublasHandle_t cublas_handle;
        cusolverDnHandle_t cusolver_handle;


        bool useOtherStream;

    public:
        linSolver() throw(cudaRunTimeError, cuBlasError, cuSolverError);

        linSolver(cudaStream_t stream) throw(cudaRunTimeError, cuBlasError, cuSolverError);

        ~linSolver() throw(cudaRunTimeError, cuBlasError, cuSolverError);

        void sync() const throw(cudaRunTimeError);

        cudaStream_t get_stream() const;


        ////Note that all the following routines are based on generic template, the each argument offered,
        ////The argument should have the following member functions
        ////data(): return a pointer to DEVICE memory
        ////rows(): return the number of rows
        ////cols(): return the number of columns

        ////Furthermore, the input matrix should offer a typedef (A::type) which returns the underlying type parameter
        ////Currently only the types "float", and "double" are supported



        ////Wrappers for cublas


        ////Blas level 3 routines
        ////returns alpha*op(A)*op(B) + beta*X
        ////where op takes values in transpose or no transpose
        template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
        void matmul_nn(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                       MatType3 &X) throw(dimMisMatch, cuBlasError);

        template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
        void matmul_nt(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                       MatType3 &X) throw(dimMisMatch, cuBlasError);

        template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
        void matmul_tn(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                       MatType3 &X) throw(dimMisMatch, cuBlasError);

        template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
        void matmul_tt(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                       MatType3 &X) throw(dimMisMatch, cuBlasError);


        ////Wrappers for cusolver


        ////Solve thin SVD
        ////Given input matrix A, compute the thin SVD decomposiion
        ////Note that the A's contents will be DESTROYED after the operation, if this is not desirable, make a copy of A first.
        ////Currently only supports the case where A have as many as rows in as columns
        ////Also note that although in the thin decomposition, the dimension of U is still required to be m * m, where m is the number of rows of A

        template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
        void thin_svd(MatType1 &A, MatType2 &U, MatType3 &VT, VecType &S) throw();

        template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
        void full_svd(MatType1 &A, MatType2 &U, MatType3 &VT, VecType &S) throw();


    };


    ////Implementaions////

    linSolver::linSolver() throw(cudaRunTimeError, cuBlasError, cuSolverError) {
        try {
            workSpace = new WorkSpace();
            cudaStream_t _stream;
            handle_cuda_runtime_error(cudaStreamCreate(&_stream));
            stream = _stream;
            handle_cusolver_error(cusolverDnCreate(&cusolver_handle));
            handle_cublas_error(cublasCreate_v2(&cublas_handle));
            handle_cusolver_error(cusolverDnSetStream(cusolver_handle, stream));
            handle_cublas_error(cublasSetStream_v2(cublas_handle, stream));
            handle_cuda_runtime_error(cudaStreamSynchronize(stream));
        }
        THROW_LIN_ERROR
    }

    linSolver::linSolver(cudaStream_t _stream) throw(cudaRunTimeError, cuBlasError, cuSolverError) {
        try {
            workSpace = new WorkSpace();
            stream = _stream;
            handle_cusolver_error(cusolverDnCreate(&cusolver_handle));
            handle_cublas_error(cublasCreate_v2(&cublas_handle));
            handle_cusolver_error(cusolverDnSetStream(cusolver_handle, stream));
            handle_cublas_error(cublasSetStream_v2(cublas_handle, stream));
            handle_cuda_runtime_error(cudaStreamSynchronize(stream));
        }
        THROW_LIN_ERROR
    }

    linSolver::~linSolver() throw(cudaRunTimeError, cuBlasError, cuSolverError) {
        try {
            handle_cuda_runtime_error(cudaStreamSynchronize(stream));
            //Sync before destroying everything

            handle_cusolver_error(cusolverDnDestroy(cusolver_handle));
            handle_cublas_error(cublasDestroy_v2(cublas_handle));

            if (!useOtherStream) {
                handle_cuda_runtime_error(cudaStreamDestroy(stream));
            }
        }
        THROW_LIN_ERROR

    }

    void linSolver::sync() const throw(cudaRunTimeError) {
        try {
            handle_cuda_runtime_error(cudaStreamSynchronize(stream));
        }
        catch (cudaRunTimeError e) {
            throw e;
        }

    }

    cudaStream_t linSolver::get_stream() const {
        return stream;
    }

    template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
    void linSolver::matmul_nn(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                              MatType3 &X) throw(dimMisMatch, cuBlasError) {
        try {
            cuStat::internal::linSolver_impl<Scalar>::matmul_nn(cublas_handle, alpha, A, B, beta, X);
        }
        catch (dimMisMatch d) {
            throw d;
        }
        catch (cuBlasError d) {
            throw d;
        }
    }

    template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
    void linSolver::matmul_nt(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                              MatType3 &X) throw(dimMisMatch, cuBlasError) {
        try {
            cuStat::internal::linSolver_impl<Scalar>::matmul_nt(cublas_handle, alpha, A, B, beta, X);
        }
        catch (dimMisMatch d) {
            throw d;
        }
        catch (cuBlasError d) {
            throw d;
        }

    }

    template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
    void linSolver::matmul_tn(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                              MatType3 &X) throw(dimMisMatch, cuBlasError) {
        try {
            cuStat::internal::linSolver_impl<Scalar>::matmul_tn(cublas_handle, alpha, A, B, beta, X);
        }
        catch (dimMisMatch d) {
            throw d;
        }
        catch (cuBlasError d) {
            throw d;
        }
    }

    template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
    void linSolver::matmul_tt(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                              MatType3 &X) throw(dimMisMatch, cuBlasError) {
        try {
            cuStat::internal::linSolver_impl<Scalar>::matmul_tt(cublas_handle, alpha, A, B, beta, X);
        }
        catch (dimMisMatch d) {
            throw d;
        }
        catch (cuBlasError d) {
            throw d;
        }
    }

    template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
    void linSolver::thin_svd(MatType1 &A, MatType2 &U, MatType3 &VT, VecType &S) throw() {
        try {
            using type = typename MatType1::type;
            cuStat::internal::svd_solver_impl<type>::query_workspace(cusolver_handle, *workSpace, A);
            cuStat::internal::svd_solver_impl<type>::thin_svd(cusolver_handle, *workSpace, A, U, VT, S);
        }
        catch (...){
            throw;
        }
    }

    template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
    void linSolver::full_svd(MatType1 &A, MatType2 &U, MatType3 &VT, VecType &S) throw(){
        try {
            using type = typename MatType1::type;
            cuStat::internal::svd_solver_impl<type>::query_workspace(cusolver_handle, *workSpace, A);
            cuStat::internal::svd_solver_impl<type>::full_svd(cusolver_handle, *workSpace, A, U, VT, S);
        }
        catch (...){
            throw;
        }
    }
}