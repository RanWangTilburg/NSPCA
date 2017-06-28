#pragma once
////A Scalar Class that faciliate transformation of device and host scalar
#include <cuda_runtime.h>
namespace cuStat {
    template<typename ScalarType>
    class Scalar {
    private:
        ScalarType *ptrData;
    public:
        Scalar(const ScalarType default_value = 0);

        ~Scalar();

        ScalarType *data();

        const ScalarType *data() const;

        ScalarType &operator()();

        ScalarType operator()() const;

        Scalar<ScalarType> operator=(const ScalarType  src);

    };

    template<typename ScalarType>
    Scalar<ScalarType>::Scalar(const ScalarType default_value) {
        cudaMallocManaged(&ptrData, sizeof(Scalar));
    }

    template<typename ScalarType>
    Scalar<ScalarType>::~Scalar() {
        cudaFree(ptrData);
    }

    template<typename ScalarType>
    ScalarType *Scalar<ScalarType>::data() {
        return ptrData;
    }

    template<typename ScalarType>
    const ScalarType *Scalar<ScalarType>::data() const {
        return ptrData;
    }

    template<typename ScalarType>
    ScalarType &Scalar<ScalarType>::operator()() {
        return *ptrData;
    }

    template<typename ScalarType>
    ScalarType Scalar<ScalarType>::operator()() const {
        return *ptrData;
    }

    template<typename ScalarType>
    Scalar<ScalarType> Scalar<ScalarType>::operator=(const ScalarType src) {
        *ptrData=src;
        return *this;
    }


}