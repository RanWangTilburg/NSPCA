//
// Created by user on 15-11-16.
//

#ifndef NSPCA_CUMATRIX_H
#define NSPCA_CUMATRIX_H

#include <thrust/device_vector.h>
#include <cstdlib>
#include <ostream>

using std::ostream;

#include "cuMatExecDevice.h"
#include "base.h"
#include "macro.h"
#include "cuView.h"
namespace cuExec {
    template<typename Scalar>
    class cuMatrix {
    private:
        size_t nRow;
        size_t nCol;
        Scalar *devPtr;

        template<typename T>
        friend ostream &output_vec(ostream &os, const cuMatrix<T> &ds);

        template<typename T>
        friend ostream &output_mat(ostream &os, const cuMatrix<T> &ds);

    public:
        using type = Scalar;

        cuMatrix(const size_t _rows, const size_t _cols);

        cuMatrix(const size_t _rows, const size_t _cols, const Scalar value);

        ~cuMatrix();

        void randn(const Scalar mean, const
        Scalar std);

        size_t rows() const;

        size_t cols() const;

        CUDA_DEVICE Scalar& operator()(const size_t i, const size_t j);

        CUDA_DEVICE Scalar operator()(const size_t i, const size_t j) const;

        CUDA_DEVICE Scalar& operator()(const size_t i);

        CUDA_DEVICE Scalar operator()(const size_t i) const;

        template<typename T>
        friend ostream &operator<<(ostream &os, const cuMatrix<T> &ds);

        void print();

        Scalar *getPtr();

        const Scalar *getPtr() const;

        thrust::device_ptr<Scalar> getThrustPtr();

        template<typename Op>
        cuMatrix<Scalar> &operator=(const Assignable<Op> * assign);

        cuView<Scalar> *v();
    };

    template
    class cuMatrix<double>;

    template
    class cuMatrix<int>;

    template
    class cuMatrix<float>;


    void test();

    void test_fp();
//    void init_cu_matrix();
}

#endif //NSPCA_CUMATRIX_H
