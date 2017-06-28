#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>

#include "../../util/scalar.h"
#include "../../util/workspace.h"
#include "../../util/matrix.h"
#include "../../util/exception.h"
////Implementation of the svd solver////

namespace cuStat{
    namespace internal{
        template<typename Scalar>
        class svd_solver_impl {
        };

        template<>
        class svd_solver_impl<double> {
        public:
            static void
            query_workspace(cusolverDnHandle_t handle, WorkSpace &space, Matrix<double> &mat) throw();

            static void
            query_workspace(cusolverDnHandle_t handle, WorkSpace &space, View<double> &view) throw();

            template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
            static void thin_svd(cusolverDnHandle_t handle, WorkSpace &space, MatType1 &A, MatType2 &U, MatType3 &VT,
                                 VecType &S) throw();

            template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
            static void full_svd(cusolverDnHandle_t handle, WorkSpace &space, MatType1 &A, MatType2 &U, MatType3 &VT,
                                 VecType &S) throw();

            template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
            static void check_dim(const MatType1 &A, const MatType2 &U, const MatType3 &VT,
                                  const VecType &S) throw();
        };

        template<>
        class svd_solver_impl<float> {
        public:
            static void
            query_workspace(cusolverDnHandle_t handle, WorkSpace &space, Matrix<float> &mat) throw();

            static void
            query_workspace(cusolverDnHandle_t handle, WorkSpace &space, View<float> &view) throw();

            template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
            static void thin_svd(cusolverDnHandle_t handle, WorkSpace &space, MatType1 &A, MatType2 &U, MatType3 &VT,
                                 VecType &S) throw();

            template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
            static void full_svd(cusolverDnHandle_t handle, WorkSpace &space, MatType1 &A, MatType2 &U, MatType3 &VT,
                                 VecType &S) throw();

            template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
            static void check_dim(const MatType1 &A, const MatType2 &U, const MatType3 &VT,
                                  const VecType &S) throw();
        };
        void svd_solver_impl<double>::query_workspace(cusolverDnHandle_t handle, WorkSpace &space,
                                                      Matrix<double> &mat) throw() {
            try {
                int rwork_size = std::min((int) mat.rows(), (int) mat.cols()) - 1;
                int lwork_size = 0;

                handle_cusolver_error(
                        (cusolverDnDgesvd_bufferSize(handle, (int) mat.rows(), (int) mat.cols(), &lwork_size)));
                space.resize_lspace((size_t) lwork_size * sizeof(double));
                space.resize_rspace((size_t) rwork_size * sizeof(double));

            }
            catch (...){
                throw;
            }
        }

        void svd_solver_impl<double>::query_workspace(cusolverDnHandle_t handle, WorkSpace &space,
                                                      View<double> &view) throw() {
            try {
                int rwork_size = std::min((int) view.rows(), (int) view.cols()) - 1;
                int lwork_size = 0;

                handle_cusolver_error(
                        (cusolverDnDgesvd_bufferSize(handle, (int) view.rows(), (int) view.cols(), &lwork_size)));
                space.resize_lspace((size_t) lwork_size * sizeof(double));
                space.resize_rspace((size_t) rwork_size * sizeof(double));

            }
            catch (...) {
                throw;
            }
        }

        template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
        void svd_solver_impl<double>::thin_svd(cusolverDnHandle_t handle, WorkSpace &space, MatType1 &A, MatType2 &U,
                                               MatType3 &VT, VecType &S) throw() {
            try {
                svd_solver_impl<double>::check_dim(A, U, VT, S);
                Scalar<int> info(0);
                int M = A.rows();
                int N = A.cols();
                double *d_A = A.data();
                double *d_U = U.data();
                double *d_S = S.data();
                double *d_V = VT.data();
                double *work = (double *) space.lwork();
                int work_size = (int) space.lsize() / sizeof(double);

                handle_cusolver_error(
                        cusolverDnDgesvd(handle, 'S', 'S', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size,
                                         (double *) space.rwork(), info.data()));
            }
            catch (...){
                throw;
            }

        }


        template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
        void svd_solver_impl<float>::check_dim(const MatType1 &A, const MatType2 &U, const MatType3 &VT,
                                               const VecType &S) throw(){
            try {
                if (A.cols() > A.rows()) {
                    throw dimMisMatch(
                            "Currently SVD module only supports matrix with row numbers no smaller than column numbers");
                //} else if (U.rows() != U.cols()) {
                //    throw dimMisMatch("U must be a square matrix");
                } else if (U.rows() != A.rows()) {
                    throw dimMisMatch("Matrix U and Matrix A must be of the same row length");
                } else if (S.rows() != A.cols()) {
                    throw dimMisMatch("Vector S must be of the same length as A's column");
                } else if (VT.rows() != VT.cols()) {
                    throw dimMisMatch("V^T must be square matrix");
                } else if (VT.rows() != A.cols()) {
                    throw dimMisMatch("V^T must be of row length of A's column");
                }
            }
            catch (...){
                throw;
            }
        };

        template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
        void
        svd_solver_impl<double>::check_dim(const MatType1 &A, const MatType2 &U, const MatType3 &VT,
                                           const VecType &S) throw() {
            try {
                if (A.cols() > A.rows()) {
                    throw dimMisMatch(
                            "Currently SVD module only supports matrix with row numbers no smaller than column numbers");
                //} else if (U.rows() != U.cols()) {
                //    throw dimMisMatch("U must be a square matrix");
                } else if (U.rows() != A.rows()) {
                    throw dimMisMatch("Matrix U and Matrix A must be of the same row length");
                } else if (S.rows() != A.cols()) {
                    throw dimMisMatch("Vector S must be of the same length as A's column");
                } else if (VT.rows() != VT.cols()) {
                    throw dimMisMatch("V^T must be square matrix");
                } else if (VT.rows() != A.cols()) {
                    throw dimMisMatch("V^T must be of row length of A's column");
                }
            }
            catch (...){
                throw;
            }
        }

        template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
        void svd_solver_impl<double>::full_svd(cusolverDnHandle_t handle, WorkSpace &space, MatType1 &A, MatType2 &U,
                                               MatType3 &VT, VecType &S) throw() {
            try {
                svd_solver_impl<double>::check_dim(A, U, VT, S);
                Scalar<int> info(0);
                int M = A.rows();
                int N = A.cols();
                double *d_A = A.data();
                double *d_U = U.data();
                double *d_S = S.data();
                double *d_V = VT.data();
                double *work = (double *) space.lwork();
                int work_size = (int) space.lsize() / sizeof(double);

                handle_cusolver_error(
                        cusolverDnDgesvd(handle, 'A', 'A', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size,
                                         (double *) space.rwork(), info.data()));
            }
            catch (...){
                throw;
            }
        }

        void svd_solver_impl<float>::query_workspace(cusolverDnHandle_t handle, WorkSpace &space, Matrix<float> &mat) throw() {
            try {
                int rwork_size = std::min((int) mat.rows(), (int) mat.cols()) - 1;
                int lwork_size = 0;

                handle_cusolver_error(
                        (cusolverDnDgesvd_bufferSize(handle, (int) mat.rows(), (int) mat.cols(), &lwork_size)));
                space.resize_lspace((size_t) lwork_size * sizeof(float));
                space.resize_rspace((size_t) rwork_size * sizeof(float));

            }
            catch (...){
                throw;
            }
        }

        void svd_solver_impl<float>::query_workspace(cusolverDnHandle_t handle, WorkSpace &space, View<float> &view) throw() {
            try {
                int rwork_size = std::min((int) view.rows(), (int) view.cols()) - 1;
                int lwork_size = 0;

                handle_cusolver_error(
                        (cusolverDnDgesvd_bufferSize(handle, (int) view.rows(), (int) view.cols(), &lwork_size)));
                space.resize_lspace((size_t) lwork_size * sizeof(float));
                space.resize_rspace((size_t) rwork_size * sizeof(float));

            }
            catch (...){
                throw;
            }
        }

        template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
        void svd_solver_impl<float>::thin_svd(cusolverDnHandle_t handle, WorkSpace &space, MatType1 &A, MatType2 &U, MatType3 &VT,
                                              VecType &S) throw(){
            try {
                svd_solver_impl<float>::check_dim(A, U, VT, S);
                Scalar<int> info(0);
                int M = A.rows();
                int N = A.cols();
                float *d_A = A.data();
                float *d_U = U.data();
                float *d_S = S.data();
                float *d_V = VT.data();
                float *work = (float *) space.lwork();
                int work_size = (int) space.lsize() / sizeof(float);

                handle_cusolver_error(
                        cusolverDnSgesvd(handle, 'S', 'S', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size,
                                         (float *) space.rwork(), info.data()));
            }
            catch (...){
                throw;
            }
        };

        template<typename MatType1, typename MatType2, typename MatType3, typename VecType>
        void svd_solver_impl<float>::full_svd(cusolverDnHandle_t handle, WorkSpace &space, MatType1 &A, MatType2 &U, MatType3 &VT,
                                              VecType &S) throw(){
            try {
                svd_solver_impl<float>::check_dim(A, U, VT, S);
                Scalar<int> info(0);
                int M = A.rows();
                int N = A.cols();
                float *d_A = A.data();
                float *d_U = U.data();
                float *d_S = S.data();
                float *d_V = VT.data();
                float *work = (float *) space.lwork();
                int work_size = (int) space.lsize() / sizeof(float);

                handle_cusolver_error(
                        cusolverDnSgesvd(handle, 'A', 'A', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size,
                                         (float *) space.rwork(), info.data()));
            }
            catch (...){
                throw;
            }
        };
    }////End of internal
}////End of cuStat