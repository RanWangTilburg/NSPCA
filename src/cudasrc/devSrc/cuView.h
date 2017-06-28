//
// Created by user on 15-11-16.
//

#ifndef NSPCA_CUVIEW_H
#define NSPCA_CUVIEW_H

#include <cstdlib>
#include <ostream>

using std::ostream;

#include "macro.h"
#include "base.h"
namespace cuExec {

    template<typename Scalar, typename Derived>
    class Base : public Managed {
    public:
        Derived *derived() {
            return static_cast<Derived *>(this);
        }

        const Derived *derived() const {
            return static_cast<const Derived *>(this);
        }

        template<typename OtherDerived>
        const AddOp<Scalar, Derived, OtherDerived> * operator+(Base<Scalar, OtherDerived>& rhs) {
            return new AddOp<Scalar, Derived, OtherDerived>(derived(), rhs.derived());
        }

    };


    template<typename ScalarType, typename LHS, typename RHS>
    class AddOp : public Base<ScalarType, AddOp<ScalarType, LHS, RHS>> {
    public:
        LHS *lhs;
        RHS *rhs;
        using Base<ScalarType, AddOp<ScalarType, LHS, RHS>>::operator+;

        AddOp(LHS *_lhs, RHS *_rhs) : lhs(_lhs), rhs(_rhs) {}
        ~AddOp(){
            delete lhs;
            delete rhs;
        }
        __device__ ScalarType
        get(const size_t i, const size_t j) const {
            return lhs->get(i, j) + rhs->get(i, j);
        }


        __device__ ScalarType

        operator()(const size_t i) const {
            return lhs->operator()(i) + rhs->operator()(i);
        }

        Assignable<AddOp<ScalarType, LHS, RHS>> *
        run(unsigned int numThreads, unsigned int numBlocks, cudaStream_t stream) const {
            return new Assignable<AddOp<ScalarType, LHS, RHS>>(this, numThreads, numBlocks, stream);
        }

    };

    template<typename Operation>
    class Assignable {
    public:
        unsigned int numThreads;
        unsigned int numBlocks;
        cudaStream_t stream;
        const Operation *operation;

        Assignable(const Operation *_operation, unsigned int _numThreads, unsigned int _numBlocks, cudaStream_t _stream)
                : operation(
                _operation), numThreads(_numThreads), numBlocks(_numBlocks), stream(_stream) {}

        ~Assignable() {
            delete operation;
        }
    };

    template<typename Scalar>
    class cuView:public Base<Scalar, cuView<Scalar>> {
    private:
        template<typename T>
        CUDA_HOST friend ostream &print_vec(ostream &os, const cuView<T> &dst);

        template<typename T>
        CUDA_HOST        friend ostream &print_mat(ostream &os, const cuView<T> &dst);

    public:
        using type = Scalar;
        size_t nRow;
        size_t nCol;
        Scalar *data;


        CUDA_CALLABLE_MEMBER cuView(Scalar *data, size_t nRow, size_t nCol);
        void print();

        template<typename T>
        CUDA_HOST friend std::ostream &operator<<(std::ostream &os, const cuView<T> &dt);
        CUDA_DEVICE Scalar& operator()(const size_t i, const size_t j);
        CUDA_DEVICE Scalar operator()(const size_t i, const size_t j) const ;
        CUDA_DEVICE Scalar& operator()(const unsigned int i);
        CUDA_DEVICE Scalar operator()(const unsigned int i) const{
            return data[i];
        }
        CUDA_CALLABLE_MEMBER const Scalar *getPtr() const;

        CUDA_CALLABLE_MEMBER Scalar *getPtr();

        CUDA_CALLABLE_MEMBER size_t rows() const;

        CUDA_CALLABLE_MEMBER size_t cols() const;
    };

    template
    class cuView<double>;

    template
    class cuView<int>;

    template
    class cuView<float>;
}


#endif //NSPCA_CUVIEW_H
