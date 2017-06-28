//
// Created by user on 29-11-16.
//

#ifndef NSPCA_NSPCASRC_H
#define NSPCA_NSPCASRC_H
#include "cuView.h"
using cuExec::cuView;

#include "macro.h"
namespace NSPCA{
    void solve_p_nspca(double * devp, const size_t N, const size_t P, const size_t p,  double * ATZ,
                       int * restriction, const double lambda, const double scale_square, const unsigned int numThreads, const unsigned int numBlocks );

    
}


#endif //NSPCA_NSPCASRC_H
