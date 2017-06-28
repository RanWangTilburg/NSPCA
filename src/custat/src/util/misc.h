#pragma once

namespace cuStat {
    namespace internal {

        template<typename RHS, typename LHS>
        inline void assert_scalar(RHS &rhs, LHS &lhs) {
            static_assert(RHS::ScalarType == LHS::ScalarType);
        };

        template<typename RHS, typename LHS>
        inline void assert_scalar(const RHS &rhs, LHS &lhs) {
            static_assert(RHS::ScalarType == LHS::ScalarType);
        };

        template<typename RHS, typename LHS>
        inline void assert_scalar(RHS &rhs, const LHS &lhs) {
            static_assert(RHS::ScalarType == LHS::ScalarType);
        };

        template<typename RHS, typename LHS>
        inline void assert_scalar(const RHS &rhs, const LHS &lhs) {
            static_assert(RHS::ScalarType == LHS::ScalarType);
        };
    }
}