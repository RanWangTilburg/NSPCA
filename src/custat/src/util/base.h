#pragma once

#include <cstdlib>
#include "managed.h"
#include "exception.h"
#include "kernels.h"

#include <iostream>
#include <cuda_runtime.h>
#include "forward_declare"
#include "unary_op.h"

namespace cuStat {
    namespace internal {

        template<typename Scalar, class Derived>
        class Base : public Managed {
        public:
            /** \returns a reference to the derived object */
            Derived &derived() { return *static_cast<Derived *>(this); }

            /** \returns a const reference to the derived object */
            const Derived &derived() const { return *static_cast<const Derived *>(this); }

            template<typename OP>
            UnaryOp<Scalar, Derived, OP> map(OP op) {
                return UnaryOp<Scalar, Derived, OP>(this->derived(), op);
            }
        };


        template<typename Scalar, typename LHS, typename RHS>
        class AddOp : Base<Scalar, AddOp<Scalar, LHS, RHS>> {
        public:
        };

//        template<typename Scalar, template<typename ...> class DST>
//        struct add_one{};
//
//        template<typename Scalar, typename... Other, template<typename ...> class DST>
//        struct add_one<Scalar, DST<Scalar, Other...>>{
//            DST<Scalar, Other...> dst;
//
//            add_one(DST<Scalar, Other...> _dst):dst(_dst){}
//
//            CUDA_CALLABLE_MEMBER add_one(add_one<Scalar, DST<Scalar, Other...>> other):dst(other.dst){}
//        };

        template<typename RHS>
        class MatrixAssign {
        private:
            RHS rhs;
            unsigned int numThreads;
            unsigned int numBlocks;
            cudaStream_t stream;
        public:
            MatrixAssign(RHS _rhs, unsigned int _numThreads, unsigned int _numBlocks, cudaStream_t _stream) : rhs(_rhs),
                                                                                                              numThreads(
                                                                                                                      _numThreads),
                                                                                                              numBlocks(
                                                                                                                      _numBlocks),
                                                                                                              stream(_stream) {
                std::cout << "Has yet to implement check on the number of threads and blocks" << std::endl;
            }

            template<typename LHS>
            void run(LHS lhs) const {
//                std::cout << "HERE?" << std::endl;
                assign_matrix_kernel << < numThreads, numBlocks, 0, stream >> > (lhs, rhs);
            }


        };

        template<typename Scalar, typename SRC, typename OP>
        class UnaryOp : public Base<Scalar, UnaryOp<Scalar, SRC, OP>> {
        public:
            SRC src;
            OP op;

            using ScalarType = typename SRC::ScalarType;
            CUDA_CALLABLE_MEMBER UnaryOp(SRC _src, OP _op) : src(_src), op(_op) {}

            CUDA_CALLABLE_MEMBER UnaryOp(const UnaryOp &other) : src(other.src), op(other.op) {}

            CUDA_CALLABLE_MEMBER ScalarType operator[](const unsigned int index) {
                return op(src[index]);
            }

            CUDA_CALLABLE_MEMBER ScalarType operator[](const size_t index) {
                return op(src[index]);
            }

            MatrixAssign<UnaryOp<Scalar, SRC, OP>>
            run(unsigned int numThreads, unsigned int numBlocks, cudaStream_t stream) {
                return MatrixAssign<UnaryOp<Scalar, SRC, OP>>(*this, numThreads, numBlocks, stream);
            }

        };

        template<typename Scalar, typename LHS, typename RHS, typename OP>
        class BinaryOp: public Base<Scalar, BinaryOp<Scalar, LHS, RHS, OP>>{
        public:
            LHS lhs;
            RHS rhs;
            OP op;

            using ScalarType =  Scalar;
            CUDA_CALLABLE_MEMBER BinaryOp(LHS _lhs, RHS _rhs, OP _op):lhs(_lhs), rhs(_rhs), op(_op){}

            CUDA_CALLABLE_MEMBER BinaryOp(const BinaryOp& other):lhs(other.lhs), rhs(other.rhs), op(other.op){}

            CUDA_CALLABLE_MEMBER ScalarType operator[](const unsigned int index) {
                return op(lhs[index], rhs[index]);
            }

            CUDA_CALLABLE_MEMBER ScalarType operator[](const size_t index) {
                return op(lhs[index], rhs[index]);
            }

            MatrixAssign<BinaryOp<Scalar, LHS, RHS, OP>>
            run(unsigned int numThreads, unsigned int numBlocks, cudaStream_t stream) {
                return MatrixAssign<BinaryOp<Scalar, LHS,RHS, OP>>(*this, numThreads, numBlocks, stream);
            }
        };


//        template<typename DST>
//        struct add_one {
//            DST dst;
//            using ScalarType = typename DST::ScalarType;
//            CUDA_CALLABLE_MEMBER add_one(DST _dst) : dst(_dst) {}
//
//            CUDA_CALLABLE_MEMBER add_one(const add_one &other) : dst(other.dst) {}
//
//            CUDA_CALLABLE_MEMBER ScalarType operator[](unsigned int index) {
//                return dst[index]+1;
//            }
//
//            CUDA_CALLABLE_MEMBER ScalarType operator[](size_t index) {
//                return dst[index]+1;
//            }
//
//            MatrixAssign<add_one<DST>>
//            run(unsigned int numThreads, unsigned int numBlocks, cudaStream_t stream) {
//                return MatrixAssign<add_one<DST>>(*this, numThreads, numBlocks, stream);
//            }
//
//
//
//        };


    }
}