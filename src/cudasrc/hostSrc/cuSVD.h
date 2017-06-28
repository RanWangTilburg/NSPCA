//
// Created by user on 18-11-16.
//

#ifndef NSPCA_CUSVD_H
#define NSPCA_CUSVD_H

#include <cuda_runtime.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <cstdlib>
#include "devWorkSpace.h"
#include "devScalar.h"
namespace cuExec{
    class cuSVD{
    private:
        cusolverDnHandle_t  handle;
        cudaStream_t stream;
    public:
         cudaStream_t getStream() const;

    private:
        bool useOtherStream;
    public:
        cuSVD();
        cuSVD(cudaStream_t);
        template<typename MatType1, typename MatType2, typename MatType3, typename MatType4, typename Scalar>
        void solveThinUThinV(MatType1& A, MatType2& U, MatType3& S, MatType4& V, const devWorkSpace<Scalar>& workSpace);
        template<typename MatType1, typename Scalar>
        devWorkSpace<Scalar>& queryWorkSpace(MatType1 &A, devWorkSpace<Scalar> &workspace);
        void sync();
        ~cuSVD();
    };
}


#endif //NSPCA_CUSVD_H
