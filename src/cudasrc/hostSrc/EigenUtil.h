//
// Created by user on 19-11-16.
//

#ifndef NSPCA_EIGENUTIL_H
#define NSPCA_EIGENUTIL_H

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace cuExec{
    template<typename Matrix>
    void standardize(Matrix& X);

    template<typename Vector>
    double variance(const Vector & v);

    template<typename Vector>
    double average(const Vector &v);

    template<typename Matrix, typename Scalar>
    void rescale(Matrix & matrix, const Scalar scale);



}


#endif //NSPCA_EIGENUTIL_H
