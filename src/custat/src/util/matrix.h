#pragma once

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include <thrust/device_vector.h>
#include <cstdlib>
#include <ostream>
#include <memory>
#include <bits/unique_ptr.h>
#include <bits/shared_ptr.h>


using std::ostream;
using std::endl;


#include "base.h"
#include "macro"
#include "matrix_kernels.h"
#include "exception.h"
#include "scalar.h"
#include "view.h"
#include "unary_op.h"
#include "scalar_unary_op.h"
//#include "../../../../../../../../usr/include/c++/4.8/cstddef"

//#include "../../../../../../../../usr/include/c++/4.8/clocale"


////Matrix Class including Matrix and Matrix View, both of which is column majored

namespace cuStat {

//
//    template<typename Scalar>
//    class View : public cuStat::internal::Base<Scalar, View<Scalar>> {
//    public:
//        using super_t = cuStat::internal::Base<Scalar, View<Scalar>>;
//        using type = Scalar;
//    public:
//        Scalar *ptrDevice;
//        const size_t numRows;
//        const size_t numCols;
//        const size_t no_elem;
//
//        template<typename T>
//        friend ostream &print_vec(ostream &os, const View<T> &data) throw(cudaRunTimeError);
//
//        template<typename T>
//        friend ostream &print_mat(ostream &os, const View<T> &data) throw(cudaRunTimeError);
//
//    public:
//        CUDA_CALLABLE_MEMBER View(Scalar *data, const size_t _numRow, const size_t _numCols);
////        CUDA_CALLABLE_MEMBER View(Scalar *data, size_t _numRow, size_t _numCol);
//        CUDA_CALLABLE_MEMBER View(const View<Scalar> &other);
//
//        CUDA_CALLABLE_MEMBER
//        Scalar &operator()(const size_t row, const size_t col);
//
//        CUDA_CALLABLE_MEMBER
//        Scalar operator()(const size_t row, const size_t Col) const;
//
//        CUDA_CALLABLE_MEMBER
//        Scalar &operator[](const size_t index);
//
//        CUDA_CALLABLE_MEMBER
//        Scalar operator[](const size_t index) const;
//
//        CUDA_CALLABLE_MEMBER
//        Scalar *data();
//
//        CUDA_CALLABLE_MEMBER
//        const Scalar *data() const;
//
//        CUDA_CALLABLE_MEMBER
//        size_t rows() const;
//
//        CUDA_CALLABLE_MEMBER
//        size_t cols() const;
//
//        CUDA_CALLABLE_MEMBER
//        size_t size() const;
//
//        ////Note that using this function will block the pipeline of GPU due to calling cudaDeviceSynchronize
//        template<typename T>
//        friend ostream &operator<<(ostream &os, const View<T> &data) throw(cudaRunTimeError);
//
//        ////Fill the matrix randomly generated  values that follows a normal distribution, with expectation "mean", and standard deviation "std"
//        void randn(const unsigned int seed, const Scalar mean, const Scalar std, cudaStream_t stream,
//                   const unsigned int numThreads,
//                   const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError);
//
//        ////Fill the matrix with randomly generated values that follow a uniform distribution
//        ////The lower threshold is given in "lower", and upper in "upper"
//        void rand(const unsigned int seed, const Scalar lower, const Scalar upper, cudaStream_t stream,
//                  const unsigned int numThreads, const unsigned int numblocks) throw(numThreadsError, cudaRunTimeError);
//    };

    template<typename Scalar>
    class Matrix : public thrust::device_vector<Scalar> {
    public:

        using type = Scalar;
        using super_t = thrust::device_vector<Scalar>;
        using thrust::device_vector<Scalar>::begin;
        using thrust::device_vector<Scalar>::end;

    private:
        Scalar *ptrDevice;
        const size_t numRows;
        const size_t numCols;

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

        View<Scalar> view();
        ////Returns a smart pointer to view object
//        std::unique_ptr <View<Scalar>> view_uptr() const;
//
//        std::shared_ptr <View<Scalar>> view_sptr() const;
//
//        ////Returns a normal pointer to view object
//        View<Scalar> *view_ptr() const;

        Scalar sum() const;

        void print() const;

    };
    ////////////////////////////////////////////////////////////
    ////Implementation/////////////////////////////////////////
    ////////////////////////////////////////////////////////////

    template<typename Scalar>
    void Matrix<Scalar>::print() const {
        std::cout << *this << std::endl;
    }

    template<typename Scalar>
    Scalar Matrix<Scalar>::sum() const {
        thrust::plus <Scalar> addOp;
        return thrust::reduce(begin(), end(), 0.0, addOp);
    }

//
//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER
//    View<Scalar>::View(Scalar *data, const size_t _numRow, const size_t _numCols):ptrDevice(data), numRows(_numRow),
//                                                                                  numCols(_numCols),
//                                                                                  no_elem(_numRow * _numCols) {}
//
//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER Scalar &View<Scalar>::operator()(const size_t row, const size_t col) {
//        return ptrDevice[row + numRows * col];
//    }
//
//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER Scalar View<Scalar>::operator()(const size_t row, const size_t col) const {
//        return ptrDevice[row + numRows * col];
//    }
//
//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER Scalar &View<Scalar>::operator[](const size_t index) {
//        return ptrDevice[index];
//    }
//
//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER Scalar View<Scalar>::operator[](const size_t index) const {
//        return ptrDevice[index];
//    }
//
//
//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER Scalar *View<Scalar>::data() {
//        return ptrDevice;
//    }
//
//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER const Scalar *View<Scalar>::data() const {
//        return ptrDevice;
//    }
//
//
//    template<typename Scalar>
//    size_t View<Scalar>::rows() const {
//        return numRows;
//    }
//
//    template<typename Scalar>
//    size_t View<Scalar>::cols() const {
//        return numCols;
//    }
//
//    template<typename T>
//    ostream &print_vec(ostream &os, const View<T> &data) throw(cudaRunTimeError) {
//        try {
//            T *host_ptr;
//            const size_t size = data.rows() * data.cols();
//            handle_cuda_runtime_error(cudaMallocHost(&host_ptr, sizeof(T) * size));
//            handle_cuda_runtime_error(
//                    cudaMemcpy(host_ptr, data.data(), sizeof(T) * data.rows() * data.cols(), cudaMemcpyDeviceToHost));
//
//            for (int i = 0; i < size; i++) {
//                os << host_ptr[i] << "\t";
//            }
//            return os;
//        }
//        catch (cudaRunTimeError e) {
//            throw e;
//        }
//    }
//
//    template<typename T>
//    ostream &print_mat(ostream &os, const View<T> &data) throw(cudaRunTimeError) {
//        try {
//            T *host_ptr;
//            const size_t size = data.rows() * data.cols();
//            handle_cuda_runtime_error(cudaMallocHost(&host_ptr, sizeof(T) * size));
//            handle_cuda_runtime_error(
//                    cudaMemcpy(host_ptr, data.data(), sizeof(T) * data.rows() * data.cols(), cudaMemcpyDeviceToHost));
//            const size_t nRow = data.rows();
//            for (int row = 0; row < nRow; row++) {
//                os << host_ptr[row];
//                for (int col = 1; col < data.cols(); col++) {
//                    os << "\t" << host_ptr[row + nRow * col];
//                }
//                os << endl;
//            }
//            return os;
//        }
//        catch (cudaRunTimeError e) {
//            throw e;
//        }
//    }
//
//    template<typename T>
//    ostream &operator<<(ostream &os, const View<T> &data) throw(cudaRunTimeError) {
//        cudaDeviceSynchronize();
//        try {
//            if (1 == data.cols()) {
//                return print_vec(os, data);
//            } else {
//                return print_mat(os, data);
//            }
//        }
//        catch (cudaRunTimeError e) {
//            throw e;
//        }
//    }
//
//    template<typename Scalar>
//    void View<Scalar>::randn(const unsigned int seed, const Scalar mean, const Scalar std, cudaStream_t stream,
//                             const unsigned int numThreads,
//                             const unsigned int numBlocks) throw(cudaRunTimeError, numThreadsError) {
//        try {
//
//            cuStat::internal::randn_func(ptrDevice, numRows * numCols, mean, std, seed, stream, numThreads, numBlocks);
//        }
//        catch (numThreadsError e) {
//            throw e;
//        }
//        catch (cudaRunTimeError e) {
//            throw e;
//        }
//
//    }
//
//    template<typename Scalar>
//    void View<Scalar>::rand(const unsigned int seed, const Scalar lower, const Scalar upper, cudaStream_t stream,
//                            const unsigned int numThreads,
//                            const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError) {
//        try {
//            cuStat::internal::rand_func(ptrDevice, numRows * numCols, lower, upper, seed, stream, numThreads,
//                                        numBlocks);
//        }
//        catch (numThreadsError e) {
//            throw e;
//        }
//        catch (cudaRunTimeError e) {
//            throw e;
//        }
//    }

//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER
//    View<Scalar>::View(const View<Scalar> &other): ptrDevice(other.ptrDevice), numRows(other.numRows),
//                                                   numCols(other.numCols), no_elem(other.numRows*other.numCols){}
//    template<typename Scalar>
//    CUDA_CALLABLE_MEMBER
//    size_t View<Scalar>::size() const {
//        return no_elem;
//    }


    template<typename Scalar>
    Matrix<Scalar>::Matrix(const size_t _numRows, const size_t _numCols):super_t(_numRows * _numCols),
                                                                         numRows(_numRows), numCols(_numCols) {
        ptrDevice = thrust::raw_pointer_cast(&(this->operator[](0)));
    }

    template<typename Scalar>
    Matrix<Scalar>::Matrix(const size_t _numRows, const size_t _numCols, const Scalar value):super_t(
            _numRows * _numCols, value), numRows(_numRows), numCols(_numCols) {
//        numRows = _numRows;
//        numCols = _numCols;
        ptrDevice = thrust::raw_pointer_cast(&(this->operator[](0)));
    }

    template<typename Scalar>
    Matrix<Scalar>::Matrix(const size_t _numRows, const size_t _numCols, const Scalar value, cudaStream_t stream,
                           const unsigned int numThreads, const unsigned int numBlocks)
    throw(cudaRunTimeError, numThreadsError):super_t(_numRows * _numCols), numRows(_numRows), numCols(_numCols) {
        try {
            ptrDevice = thrust::raw_pointer_cast(&(this->operator[](0)));
//            std::cout << "Number of Threads is " << numThreads << std::endl;
            internal::fill_with_constant(ptrDevice, _numRows * numCols, value, stream, numThreads, numBlocks);
        }
        catch (cudaRunTimeError e) {
            throw e;
        }
        catch (numThreadsError e) {
            throw e;
        }
    }

    template<typename Scalar>
    size_t Matrix<Scalar>::rows() {
        return numRows;
    }

    template<typename Scalar>
    size_t Matrix<Scalar>::cols() {
        return numCols;
    }

    template<typename Scalar>
    Scalar *Matrix<Scalar>::data() {
        return ptrDevice;
    }

    template<typename Scalar>
    const Scalar *Matrix<Scalar>::data() const {
        return ptrDevice;
    }

    template<typename Scalar>
    size_t Matrix<Scalar>::rows() const {
        return numRows;
    }

    template<typename Scalar>
    size_t Matrix<Scalar>::cols() const {
        return numCols;
    }

    template<typename Scalar>
    void Matrix<Scalar>::randn(const unsigned int seed, const Scalar mean, const Scalar std, cudaStream_t stream,
                               const unsigned int numThreads,
                               const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError) {
        try {

            cuStat::internal::randn_func(ptrDevice, numRows * numCols, mean, std, seed, stream, numThreads, numBlocks);
        }
        catch (numThreadsError e) {
            throw e;
        }
        catch (cudaRunTimeError e) {
            throw e;
        }

    }

    template<typename Scalar>
    void Matrix<Scalar>::rand(const unsigned int seed, const Scalar lower, const Scalar upper, cudaStream_t stream,
                              const unsigned int numThreads,
                              const unsigned int numBlocks) throw(numThreadsError, cudaRunTimeError) {
        try {
            cuStat::internal::rand_func(ptrDevice, numRows * numCols, lower, upper, seed, stream, numThreads,
                                        numBlocks);
        }
        catch (numThreadsError e) {
            throw e;
        }
        catch (cudaRunTimeError e) {
            throw e;
        }


    }

    template<typename T>
    ostream &print_vec(ostream &os, const Matrix<T> &data) throw(cudaRunTimeError) {
        try {
            T *host_ptr;
            const size_t size = data.rows() * data.cols();
            handle_cuda_runtime_error(cudaMallocHost(&host_ptr, sizeof(T) * size));
            handle_cuda_runtime_error(
                    cudaMemcpy(host_ptr, data.data(), sizeof(T) * data.rows() * data.cols(), cudaMemcpyDeviceToHost));

            for (int i = 0; i < size; i++) {
                os << host_ptr[i] << "\t";
            }
            return os;
        }
        catch (cudaRunTimeError e) {
            throw e;
        }
    }

    template<typename T>
    ostream &print_mat(ostream &os, const Matrix<T> &data) throw(cudaRunTimeError) {
        try {
            T *host_ptr;
            const size_t size = data.rows() * data.cols();
            handle_cuda_runtime_error(cudaMallocHost(&host_ptr, sizeof(T) * size));
            handle_cuda_runtime_error(
                    cudaMemcpy(host_ptr, data.data(), sizeof(T) * data.rows() * data.cols(), cudaMemcpyDeviceToHost));
            const size_t nRow = data.rows();
            for (int row = 0; row < nRow; row++) {
                os << host_ptr[row];
                for (int col = 1; col < data.cols(); col++) {
                    os << "\t" << host_ptr[row + nRow * col];
                }
                os << endl;
            }
            return os;
        }
        catch (cudaRunTimeError e) {
            throw e;
        }
    }

    template<typename T>
    ostream &operator<<(ostream &os, const Matrix<T> &data) throw(cudaRunTimeError) {
        cudaDeviceSynchronize();
        try {
            if (1 == data.cols()) {
                return print_vec(os, data);
            } else {
                return print_mat(os, data);
            }
        }
        catch (cudaRunTimeError e) {
            throw e;
        }
    }

    template<typename Scalar>
    View<Scalar> Matrix<Scalar>::view() const {
        return View<Scalar>(ptrDevice, numRows, numCols);
    }

    template<typename Scalar>
    View<Scalar> Matrix<Scalar>::view() {
        return View<Scalar>(ptrDevice, numRows, numCols);
    }


//    CUDA_KERNEL void test_passing_classes_kernel(View<double> dst){
//        unsigned int numThreads=1;
//        unsigned int tid = threadIdx.x;
//        unsigned int bid = blockIdx.x * numThreads;;
//
//        unsigned int position = tid + bid;
//        unsigned int size= dst.rows()*dst.cols();
//        for (; position < size; position += gridDim.x * numThreads) {
//            dst[position] = 1;
//        }
//    }
//
//    void test_pass_classes(View<double> & dst){
//        test_passing_classes_kernel<<<1, 1>>>(dst);
//        cudaDeviceSynchronize();
//    }
    template<typename T>
    void copy_to_cpu(T *dst, const Matrix<T> &from, cudaStream_t stream) throw(cudaRunTimeError);

    template<typename T>
    void copy_to_gpu(const T *from, Matrix<T> &dst, cudaStream_t stream) throw(cudaRunTimeError);

    template<typename T>
    void copy_to_cpu(T *dst, const Matrix<T> &from, cudaStream_t stream) throw(cudaRunTimeError) {

        try {
            const size_t n_row = from.rows();
            const size_t n_col = from.cols();

            handle_cuda_runtime_error(
                    cudaMemcpyAsync(dst, from.data(), sizeof(T) * n_row * n_col, cudaMemcpyDeviceToHost, stream));
        }
        catch (...) {
            throw;
        }
    }

    template<typename T>
    void copy_to_gpu(const T *from, Matrix<T> &dst, cudaStream_t stream) throw(cudaRunTimeError) {
        try {
            const size_t n_row = dst.rows();
            const size_t n_col = dst.cols();

            handle_cuda_runtime_error(
                    cudaMemcpyAsync(dst.data(), from, sizeof(T) * n_row * n_col, cudaMemcpyHostToDevice, stream));
        }
        catch (...) {
            throw;
        }
    }

    ////Typedefs
    using MatrixXd = Matrix<double>;
    using MatrixXf = Matrix<float>;
    using MatrixXi = Matrix<int>;

    using ViewXd = View<double>;
    using ViewXf = View<float>;
    using ViewXi = View<int>;


    bool test_add_one(){
        MatrixXd mat(10, 10, 2);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
//        internal::add_one<double> func;
        mat.view() = (mat.view()+1.0).run(2,2, stream);
        cudaStreamSynchronize(stream);

        std::cout << mat << std::endl;
        cudaStreamDestroy(stream);
        return false;

    }

}////End of namespace cuStat
