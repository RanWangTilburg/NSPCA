#pragma once
#include "macro.h"

namespace NSPCA{

    template<typename Scalar>
    class cuView {
    private:

    public:
        using type = Scalar;
        size_t nRow;
        size_t nCol;
        Scalar *data;


        CUDA_CALLABLE_MEMBER cuView(Scalar *data, size_t nRow, size_t nCol);

        CUDA_DEVICE Scalar &operator()(const size_t i, const size_t j);

        CUDA_DEVICE Scalar operator()(const size_t i, const size_t j) const;

        CUDA_DEVICE Scalar &operator()(const unsigned int i);

        CUDA_DEVICE Scalar operator()(const unsigned int i) const;

        CUDA_CALLABLE_MEMBER const Scalar *getPtr() const;

        CUDA_CALLABLE_MEMBER Scalar *getPtr();

        CUDA_CALLABLE_MEMBER size_t rows() const;

        CUDA_CALLABLE_MEMBER size_t cols() const;
    };

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER
    cuView<Scalar>::cuView(Scalar *data, size_t nRow, size_t nCol):data(data), nRow(nRow), nCol(nCol) {}

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER size_t cuView<Scalar>::rows() const {
        return nRow;
    }

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER size_t cuView<Scalar>::cols() const {
        return nCol;
    }

    template<typename Scalar>
    CUDA_CALLABLE_MEMBER const Scalar *cuView<Scalar>::getPtr() const {
        return data;
    }


    template<typename Scalar>
    CUDA_DEVICE Scalar &cuView<Scalar>::operator()(const size_t i, const size_t j) {
        return data[i + j * nRow];
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar cuView<Scalar>::operator()(const size_t i, const size_t j) const {
        return data[i + j * nRow];
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar &cuView<Scalar>::operator()(const unsigned int i) {
        return data[i];
    }

    template<typename Scalar>
    CUDA_DEVICE Scalar cuView<Scalar>::operator()(const unsigned int i) const {
        return data[i];

    }


    template<typename Scalar>
    CUDA_CALLABLE_MEMBER Scalar *cuView<Scalar>::getPtr() {
        return data;
    }

}