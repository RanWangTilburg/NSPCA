#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

#include "../custat/cuStat.h"
using cuStat::MatrixXd;
using cuStat::MatrixXi;
using cuStat::ViewXi;
using cuStat::ViewXd;
#include "../nspcaCudalib/nspca_cuda.h"

#include "Dim.h"
namespace NSPCA{
    namespace internal{

        struct TmpMatrices:public Dim{
            MatrixXd ZPT;
            MatrixXd U;
            MatrixXd V;
            MatrixXd diag;
            MatrixXd ATZ;
            MatrixXd AP;
            ViewXd UView;
            MatrixXd weights;
            MatrixXd Omega;
            MatrixXd omegaVec;
            MatrixXd lambdas;
            MatrixXd OmegaPT;

            TmpMatrices(size_t nObs, size_t nVar, size_t nVarAfterReduce): Dim(nObs, nVar, nVarAfterReduce),
                    ZPT(nObs, nVarAfterReduce),
                    U(nObs, nVar),
                    V(nVarAfterReduce, nVarAfterReduce),
                    diag(nVarAfterReduce, 1),
                    ATZ(nVarAfterReduce, nVar),
                    AP(nObs, nVar), UView(U.data(), nObs, nVarAfterReduce), weights(nVar, 1),
                    Omega(nVar, nVar), omegaVec(nVar, 1), lambdas(nVar, 1), OmegaPT(nVar, nVarAfterReduce) {}

            void set_weights(double new_weights, cudaStream_t stream) {
                update_weights(weights.data(), new_weights, _nVar, 4, weights.rows() / 4, stream);
            }

            void set_lambdas(double *new_lambdas, cudaStream_t stream) {
                double * truelambdas = new double[_nVar];
                for (int i=0;i<_nVar;i++){

                    truelambdas[i] =_nObs*new_lambdas[i];
                }
                cudaMemcpyAsync(lambdas.data(), truelambdas, sizeof(double) * _nVar, cudaMemcpyHostToDevice, stream);
                delete truelambdas;
            }
        };

    }
}