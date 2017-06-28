#pragma once
#include "base.h"
#include "view.h"
#include "unary_op.h"
namespace cuStat{
    namespace internal{
            template<class Derived>
            UnaryOp<double, Derived, plus<double>> operator+(double value, Derived data) {
                return data.map(plus<double>(value));
            };

            template<class Derived>
            UnaryOp<double, Derived, plus<double>> operator+(Derived data, double value) {
                return data.map(plus<double>(value));
            };

            template<class Derived>
            UnaryOp<double, Derived, plus<double>> operator-(Derived data, double value) {
                return data.map(plus<double>(-value));
            };

            template<class Derived>
            UnaryOp<double, Derived, multiply<double>> operator*(double value, Derived data) {
                return data.map(multiply<double>(value));
            };

            template<class Derived>
            UnaryOp<double, Derived, multiply<double>> operator*(Derived data, double value) {
                return data.map(multiply<double>(value));
            };

            template<class Derived>
            UnaryOp<double, Derived, divide<double>> operator/(Derived data, double value) {
                return data.map(divide<double>(value));
            };

            template<class Derived>
            UnaryOp<float, Derived, plus<float>> operator+(float value, Derived data) {
                return data.map(plus<float>(value));
            };

            template<class Derived>
            UnaryOp<float, Derived, plus<float>> operator+(Derived data, float value) {
                return data.map(plus<float>(value));
            };

            template<class Derived>
            UnaryOp<float, Derived, plus<float>> operator-(Derived data, float value) {
                return data.map(plus<float>(-value));
            };

            template<class Derived>
            UnaryOp<float, Derived, multiply<float>> operator*(float value, Derived data) {
                return data.map(multiply<float>(value));
            };

            template<class Derived>
            UnaryOp<float, Derived, multiply<float>> operator*(Derived data, float value) {
                return data.map(multiply<float>(value));
            };

            template<class Derived>
            UnaryOp<float, Derived, divide<float>> operator/(Derived data, float value) {
                return data.map(divide<float>(value));
            };

            template<class Derived>
            UnaryOp<int, Derived, plus<int>> operator+(int value, Derived data) {
                return data.map(plus<int>(value));
            };

            template<class Derived>
            UnaryOp<int, Derived, plus<int>> operator+(Derived data, int value) {
                return data.map(plus<int>(value));
            };

            template<class Derived>
            UnaryOp<int, Derived, plus<int>> operator-(Derived data, int value) {
                return data.map(plus<int>(-value));
            };

            template<class Derived>
            UnaryOp<int, Derived, multiply<int>> operator*(int value, Derived data) {
                return data.map(multiply<int>(value));
            };

            template<class Derived>
            UnaryOp<int, Derived, multiply<int>> operator*(Derived data, int value) {
                return data.map(multiply<int>(value));
            };

            template<class Derived>
            UnaryOp<int, Derived, divide<int>> operator/(Derived data, int value) {
                return data.map(divide<int>(value));
            };

            template<class Derived>
            UnaryOp<size_t, Derived, plus<size_t>> operator+(size_t value, Derived data) {
                return data.map(plus<size_t>(value));
            };

            template<class Derived>
            UnaryOp<size_t, Derived, plus<size_t>> operator+(Derived data, size_t value) {
                return data.map(plus<size_t>(value));
            };

            template<class Derived>
            UnaryOp<size_t, Derived, plus<size_t>> operator-(Derived data, size_t value) {
                return data.map(plus<size_t>(-value));
            };

            template<class Derived>
            UnaryOp<size_t, Derived, multiply<size_t>> operator*(size_t value, Derived data) {
                return data.map(multiply<size_t>(value));
            };

            template<class Derived>
            UnaryOp<size_t, Derived, multiply<size_t>> operator*(Derived data, size_t value) {
                return data.map(multiply<size_t>(value));
            };

            template<class Derived>
            UnaryOp<size_t, Derived, divide<size_t>> operator/(Derived data, size_t value) {
                return data.map(divide<size_t>(value));
            };

    };
}