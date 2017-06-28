#pragma once

/**
  * \brief This file includes the commonly used unary operations
  * \include
  *
  */
#include <cassert>
#include <cmath>
#include "base.h"
#include "macro"

namespace cuStat {
    namespace internal {


        template<typename Scalar>
        struct plus {
            Scalar constant;
            CUDA_CALLABLE_MEMBER plus(Scalar _other) : constant(_other) {}

            CUDA_CALLABLE_MEMBER plus(const plus<Scalar> &other) : constant(other.constant) {}

            CUDA_CALLABLE_MEMBER Scalar operator()(Scalar input) {
                return input + constant;
            }
        };

        template<typename Scalar>
        struct multiply {
            Scalar constant;

            CUDA_CALLABLE_MEMBER multiply(Scalar _other) : constant(_other) {}

            CUDA_CALLABLE_MEMBER multiply(const multiply<Scalar> &other) : constant(other.constant) {}

            CUDA_CALLABLE_MEMBER Scalar operator()(Scalar input) {
                return input * constant;
            }
        };

        template<typename Scalar>
        struct divide {
            Scalar constant;

            CUDA_CALLABLE_MEMBER divide(Scalar _other) {
                assert(_other != 0);
                constant = _other;
            }

            CUDA_CALLABLE_MEMBER divide(const divide<Scalar>& other) : constant(other.constant) {}


            CUDA_CALLABLE_MEMBER Scalar operator()(Scalar input) {
                return input / constant;
            }
        };

        template<typename Scalar>
        struct square{
            CUDA_CALLABLE_MEMBER square(){}

            CUDA_CALLABLE_MEMBER square(const square<Scalar>& other){}

            CUDA_CALLABLE_MEMBER Scalar operator()(Scalar input){
                return input*input;
            }
        };

        template<typename Scalar>
        struct logarithm{
            CUDA_CALLABLE_MEMBER logarithm(){}

            CUDA_CALLABLE_MEMBER logarithm(const logarithm<Scalar>& other){}

            CUDA_CALLABLE_MEMBER Scalar operator()(Scalar input){
                return log(input);
            }

        };
    }
}////end of namespace cuStat