#pragma once
#include <cstdlib>
namespace NSPCA{
    namespace internal{
        struct Dim {
            size_t _nObs;
            size_t _nVar;
            size_t _nVarAfterReduce;

            Dim(size_t nObs, size_t nVar, size_t nVarAfterReduce) : _nObs(nObs), _nVar(nVar),
                                                                    _nVarAfterReduce(nVarAfterReduce) {}
        };
    }
}

