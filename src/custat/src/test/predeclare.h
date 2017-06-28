#pragma once

////This is a header file that predeclare the instances in cuStat namespace
////Note that this file is created only for the purposes of testing

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifdef __CUDACC__
#define CUDA_HOST __host__
#else
#define CUDA_HOST
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

#ifdef __CUDACC__
#define CUDA_KERNEL __global__
#else
#define CUDA_KERNEL
#endif

#include <cstdlib>
#include <exception>
#include <string>
#include <ostream>
using std::ostream;

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>

namespace cuStat {
    bool test_add_one();

    class cudaRunTimeError : public std::exception {
    private:
        std::string errorMessage;
        cudaError_t error;
    public:
        cudaRunTimeError(cudaError_t _error);

        virtual void print() const;
    };

    ////Exception class that captures cuSolver errors
    class cuSolverError : public std::exception {
    private:
        std::string errorMessage;
        cusolverStatus_t status;
    public:
        cuSolverError(cusolverStatus_t _status);

        virtual void print() const;
    };

    ////Exception class that captures cuBlas errors
    class cuBlasError : public std::exception {
    private:
        std::string errorMessage;
        cublasStatus_t status;
    public:
        cuBlasError(cublasStatus_t _status);

        virtual void print() const;
    };

    class dimMisMatch : public std::exception {
    private:
        std::string errorMessage;
    public:
        dimMisMatch(const std::string _errorMessage);

    };

    class numThreadsError : public std::exception {
    private:
        std::string errorMessage;
    public:
        numThreadsError();
    };

    namespace internal {
        class Managed {
        public:
            void *operator new(size_t size);

            void operator delete(void *ptr);
        };

        template<typename Scalar, typename Derived>
        class Base : public Managed {
        public:
        };

        template<typename Scalar, typename LHS, typename RHS>
        class AddOp : Base<Scalar, AddOp<Scalar, LHS, RHS>> {
        public:
        };

        template<typename Scalar>
        class Assignable {
        public:

        };
    }
    class WorkSpace {
    private:
        size_t lspace_size;
        size_t rspace_size;
        void *lwork_ptr;
        void *rwork_ptr;

    public:
        WorkSpace(const size_t init_lsize = 4, const size_t init_rsize = 4) throw(cudaRunTimeError);

        ~WorkSpace() throw(cudaRunTimeError);

        size_t lsize();

        size_t rsize();

        void *lwork();

        void *rwork();

        void resize_lspace(const size_t newSize) throw(cudaRunTimeError);

        void resize_rspace(const size_t newSize) throw(cudaRunTimeError);
    };

    template<typename Scalar>
    class View : public cuStat::internal::Base<Scalar, View<Scalar>> {
    public:
        using super_t = cuStat::internal::Base<Scalar, View<Scalar>>;
        using type = Scalar;
    private:
        Scalar *ptrDevice;
        size_t numRows;
        size_t numCols;

        template<typename T>
        friend ostream &print_vec(ostream &os, const View<T> &data) throw(cudaRunTimeError);

        template<typename T>
        friend ostream &print_mat(ostream &os, const View<T> &data) throw(cudaRunTimeError);

    public:
        CUDA_CALLABLE_MEMBER View(Scalar *data, const size_t _numRow, const size_t _numCols);

        CUDA_CALLABLE_MEMBER View(const View<Scalar> &other);

        CUDA_CALLABLE_MEMBER
        Scalar &operator()(const size_t row, const size_t col);

        CUDA_CALLABLE_MEMBER
        Scalar operator()(const size_t row, const size_t Col) const;

        CUDA_CALLABLE_MEMBER
        Scalar &operator[](const size_t index);

        CUDA_CALLABLE_MEMBER
        Scalar operator[](const size_t index) const;

        CUDA_CALLABLE_MEMBER
        Scalar *data();

        CUDA_CALLABLE_MEMBER
        const Scalar *data() const;

        CUDA_CALLABLE_MEMBER
        size_t rows() const;

        CUDA_CALLABLE_MEMBER
        size_t cols() const;


        ////Note that using this function will block the pipeline of GPU due to calling cudaDeviceSynchronize
        template<typename T>
        friend ostream &operator<<(ostream &os, const View<T> &data) throw(cudaRunTimeError);

        ////Fill the matrix randomly generated  values that follows a normal distribution, with expectation "mean", and standard deviation "std"
        void randn(const unsigned int seed, const Scalar mean, const Scalar std, cudaStream_t stream,
                   const unsigned int numThreads,
                   const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError);

        ////Fill the matrix with randomly generated values that follow a uniform distribution
        ////The lower threshold is given in "lower", and upper in "upper"
        void rand(const unsigned int seed, const Scalar lower, const Scalar upper, cudaStream_t stream,
                  const unsigned int numThreads, const unsigned int numblocks) throw(numThreadsError, cudaRunTimeError);
    };

    template<typename Scalar>
    class Matrix : public thrust::device_vector<Scalar> {
    public:

        using type = Scalar;
        using super_t = thrust::device_vector<Scalar>;

    private:
        Scalar *ptrDevice;
        size_t numRows;
        size_t numCols;

        template<typename T>
        friend ostream &print_vec(ostream &os, const Matrix<T> &data) throw(cudaRunTimeError);

        template<typename T>
        friend ostream &print_mat(ostream &os, const Matrix<T> &data) throw(cudaRunTimeError);

    public:

        Matrix(const size_t _numRows, const size_t _numCols);

        ////Initialize a matrix and fill with constant values
        Matrix(const size_t _numRows, const size_t _numCols, const Scalar value);

        ////Initialize a matrix and fill with constant values, albeit async
        ////Note that the user should call cudaStreamSynchronize after calling this constrcutor
        Matrix(const size_t _numRows, const size_t _numCols, const Scalar value, cudaStream_t stream,
               const unsigned int numThreads, const unsigned int numBlocks) throw(cudaRunTimeError, numThreadsError);

        ////Fill the matrix randomly generated  values that follows a normal distribution, with expectation "mean", and standard deviation "std"
        void randn(const unsigned int seed, const Scalar mean, const Scalar std, cudaStream_t stream,
                   const unsigned int numThreads,
                   const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError);

        ////Fill the matrix with randomly generated values that follow a uniform distribution
        ////The lower threshold is given in "lower", and upper in "upper"
        void rand(const unsigned int seed, const Scalar lower, const Scalar upper, cudaStream_t stream,
                  const unsigned int numThreads, const unsigned int numblocks) throw(numThreadsError, cudaRunTimeError);

        size_t rows();

        size_t cols();


        size_t rows() const;

        size_t cols() const;

        Scalar *data();

        const Scalar *data() const;

        ////Note that using this function will block the pipeline of GPU due to calling cudaDeviceSynchronize
        template<typename T>
        friend ostream &operator<<(ostream &os, const Matrix<T> &data) throw(cudaRunTimeError);

        ////Returns a view object
        View<Scalar> view() const;

        ////Returns a smart pointer to view object
        std::unique_ptr<View<Scalar>> view_uptr() const;

        std::shared_ptr<View<Scalar>> view_sptr() const;

        ////Returns a normal pointer to view object
        View<Scalar> *view_ptr() const;

    };

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
        void matmul_nn(Scalar alpha, MatType1 &A,  MatType2 &B, Scalar beta,
                       MatType3 &X) throw(dimMisMatch, cuBlasError) ;

        template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
        void matmul_nt(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                       MatType3 &X) throw(dimMisMatch, cuBlasError);

        template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
        void matmul_tn(Scalar alpha, MatType1 &A, MatType2 &B, Scalar beta,
                       MatType3 &X) throw(dimMisMatch, cuBlasError);

        template<typename Scalar, typename MatType1, typename MatType2, typename MatType3>
        void matmul_tt(Scalar alpha,  MatType1 &A, MatType2 &B, Scalar beta,
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


    ////Typedefs
    using MatrixXd = Matrix<double>;
    using MatrixXf = Matrix<float>;
    using MatrixXi = Matrix<int>;

    using ViewXd = View<double>;
    using ViewXf = View<float>;
    using ViewXi = View<int>;


}