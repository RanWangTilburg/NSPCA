#pragma once

#include <cstdlib>
#include "Dim.h"
namespace NSPCA {
    namespace internal {
        struct Constants : public Dim {
            double _scale;
            double _scaleSquare;
            double sqrNObs;
            double nTimesScaleSquare;

            Constants(size_t nObs, size_t nVar, size_t nVarAfterReduce, double scale) : Dim(nObs, nVar,
                                                                                            nVarAfterReduce) {
                double alpha = 0;
                this->_scale = scale;
                _scaleSquare = scale * scale;
                sqrNObs = sqrt(double(nObs));
                nTimesScaleSquare = nObs * _scaleSquare;
            }


        };
    }
}
