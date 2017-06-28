//
// Created by user on 18-11-16.
//

#include "cuSVD.h"
#include "../devSrc/cuMatrix.h"
#include <algorithm>
#include "errorHandling.h"
#include <iostream>

using std::cout;
using std::endl;


namespace cuExec {

    cuSVD::cuSVD(cudaStream_t stream) {
        cusolverDnCreate(&handle);
        cusolverDnSetStream(handle, stream);
        useOtherStream = true;
    }

    cuSVD::cuSVD() {
        cusolverDnCreate(&handle);
        cudaStreamCreate(&stream);
        cusolverDnSetStream(handle, stream);
        useOtherStream = false;

    }


    cuSVD::~cuSVD() {
        cusolverDnDestroy(handle);
        if (!useOtherStream) {
            cudaStreamDestroy(stream);
        }
    }

    template<typename MatType1, typename Scalar>
    devWorkSpace<Scalar> &cuSVD::queryWorkSpace(MatType1 &A, devWorkSpace<Scalar> &workspace) {
        try {
            int rwork_size = std::min(A.rows(), A.cols()) - 1;
            int lwork_size = 0;

            HANDLE_CUSOLVER_ERROR(cusolverDnDgesvd_bufferSize(handle, (int) A.rows(), (int) A.cols(), &lwork_size))
            workspace.resizeLWork((size_t) lwork_size);
            workspace.resizeRWork((size_t) rwork_size);

        }
        catch (cuSolverError &e) {
            e.traceback();
        }

        return workspace;
    }

    template<typename MatType1, typename MatType2, typename MatType3, typename MatType4, typename Scalar>
    void
    cuSVD::solveThinUThinV(MatType1 &A, MatType2 &U, MatType3 &S, MatType4 &V, const devWorkSpace<Scalar> &workSpace) {
        try {
            devScalar<int> info(0);
            int M = A.rows();
            int N = A.cols();
            Scalar *d_A = A.getPtr();
            Scalar *d_U = U.getPtr();
            Scalar *d_S = S.getPtr();
            Scalar *d_V = V.getPtr();
            Scalar *work = workSpace.getLWork();
            int work_size = (int) workSpace.getLSize();

            HANDLE_CUSOLVER_ERROR(cusolverDnDgesvd(handle, 'S', 'S', M, N, d_A, M, d_S, d_U, M, d_V, N, work, work_size,
                                                   workSpace.getRWork(), info.getPtr()))
        }
        catch (cuSolverError &e) {
            e.traceback();
        }
    }

    cudaStream_t cuSVD::getStream() const {
        return stream;
    }

    void cuSVD::sync() {
        cudaStreamSynchronize(stream);

    }


    template devWorkSpace<double> &
    cuSVD::queryWorkSpace<cuMatrix<double>, double>(cuMatrix<double> &A, devWorkSpace<double> &workspace);

    template void
    cuSVD::solveThinUThinV(cuMatrix<double> &A, cuMatrix<double> &U, cuMatrix<double> &S, cuMatrix<double> &V,
                           const devWorkSpace<double> &workSpace);
}