#pragma once

#include "macro"
#include <cuda_runtime.h>
#include "exception.h"
#include "base.h"

namespace cuStat {

    template<typename Scalar>
    class View : public cuStat::internal::Base<Scalar, View<Scalar>> {
    public:
        using super_t = cuStat::internal::Base<Scalar, View<Scalar>>;
        using type = Scalar;
        using ScalarType = Scalar;
    public:
        Scalar *ptrDevice;
        const size_t numRows;
        const size_t numCols;
        const size_t no_elem;

        template<typename T>
        friend ostream &print_vec(ostream &os, const View<T> &data) throw(cudaRunTimeError);

        template<typename T>
        friend ostream &print_mat(ostream &os, const View<T> &data) throw(cudaRunTimeError);

    public:
        CUDA_CALLABLE_MEMBER View(Scalar *data, const size_t _numRow, const size_t _numCols);
//        CUDA_CALLABLE_MEMBER View(Scalar *data, size_t _numRow, size_t _numCol);
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

        CUDA_CALLABLE_MEMBER
        size_t size() const;

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

        template<typename AssignMatrix>
        View<Scalar> &operator=(const AssignMatrix &other);

//
//        internal::add_one<View<Scalar>> add_one(){
//            return internal::add_one<View<Scalar>>(*this);
//        };
//

    };


    //////////////////////////////////////////////////////////////
    ///////////////////Implementation/////////////////////////////
    //////////////////////////////////////////////////////////////

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER
    View<Scalar>::View(Scalar *data, const size_t _numRow, const size_t _numCols):ptrDevice(data), numRows(_numRow),
                                                                                  numCols(_numCols),
                                                                                  no_elem(_numRow * _numCols) {}

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER Scalar &View<Scalar>::operator()(const size_t row, const size_t col) {
        return ptrDevice[row + numRows * col];
    }

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER Scalar View<Scalar>::operator()(const size_t row, const size_t col) const {
        return ptrDevice[row + numRows * col];
    }

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER Scalar &View<Scalar>::operator[](const size_t index) {
        return ptrDevice[index];
    }

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER Scalar View<Scalar>::operator[](const size_t index) const {
        return ptrDevice[index];
    }


    template<typename Scalar>
    CUDA_CALLABLE_MEMBER Scalar *View<Scalar>::data() {
        return ptrDevice;
    }

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER const Scalar *View<Scalar>::data() const {
        return ptrDevice;
    }


    template<typename Scalar>
    size_t View<Scalar>::rows() const {
        return numRows;
    }

    template<typename Scalar>
    size_t View<Scalar>::cols() const {
        return numCols;
    }

    template<typename T>
    ostream &print_vec(ostream &os, const View<T> &data) throw(cudaRunTimeError) {
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
    ostream &print_mat(ostream &os, const View<T> &data) throw(cudaRunTimeError) {
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
    ostream &operator<<(ostream &os, const View<T> &data) throw(cudaRunTimeError) {
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
    void View<Scalar>::randn(const unsigned int seed, const Scalar mean, const Scalar std, cudaStream_t stream,
                             const unsigned int numThreads,
                             const unsigned int numBlocks) throw(cudaRunTimeError, numThreadsError) {
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
    void View<Scalar>::rand(const unsigned int seed, const Scalar lower, const Scalar upper, cudaStream_t stream,
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

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER
    View<Scalar>::View(const View<Scalar> &other): ptrDevice(other.ptrDevice), numRows(other.numRows),
                                                   numCols(other.numCols), no_elem(other.numRows * other.numCols) {}

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER
    size_t View<Scalar>::size() const {
        return no_elem;
    }

    template<typename Scalar>
    template<typename AssignMatrix>
    View<Scalar> &View<Scalar>::operator=(const AssignMatrix &other) {
        other.run(*this);
        return *this;
    }


}////End of namespace cuStat