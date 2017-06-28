//
// Created by user on 17-11-16.
//

#ifndef NSPCA_CUMATEXEC_H
#define NSPCA_CUMATEXEC_H
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../devSrc/cuView.h"
#include "../devSrc/cuMatrix.h"
namespace cuExec {
    class cuMatExec {
    private:
        cublasHandle_t cublasHandle;
        cudaStream_t  stream;
    public:
        cuMatExec();
        ~cuMatExec();
        template<typename ScalarType, typename MatType1, typename MatType2, typename MatType3>
               void matmulNN(ScalarType alpha, const MatType1 & lhs, const MatType2 & rhs, ScalarType beta, MatType3 & out);
        template<typename ScalarType, typename MatType1, typename MatType2, typename MatType3>
               void matmulTN(ScalarType alpha, const MatType1 & lhs, const MatType2 & rhs, ScalarType beta, MatType3 & out);
        template<typename ScalarType, typename MatType1, typename MatType2, typename MatType3>
                void matmulNT(ScalarType alpha, const MatType1 & lhs, const MatType2 & rhs, ScalarType beta, MatType3 & out);
        template<typename ScalarType, typename MatType1, typename MatType2, typename MatType3>
                void matmulTT(ScalarType alpha, const MatType1 & lhs, const MatType2 & rhs, ScalarType beta, MatType3 & out);
        void sync();
    };


}



#endif //NSPCA_CUMATEXEC_H
