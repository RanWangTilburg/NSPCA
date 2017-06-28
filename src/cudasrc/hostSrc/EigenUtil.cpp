//
// Created by user on 19-11-16.
//

#include "EigenUtil.h"

namespace cuExec{

//    template<typename Matrix, typename Vector, typename Scalar>
//    class init_eigen_util{
//    public:
//        static void standardize_init(Matrix & X);
//        static double variance_init(const Vector & v);
//        static double average_init(const Vector & v);
//        static void rescale_init(Matrix & X, const Scalar scale);
//    };
//    template<typename Matrix, typename Vector, typename Scalar>
//    void init_eigen_util<Matrix, Vector, Scalar>::standardize_init(Matrix &X) {
//        standardize(X);
//    }
//    template<typename Matrix, typename Vector, typename Scalar>
//    double init_eigen_util<Matrix, Vector, Scalar>::variance_init(const Vector &v) {
//        return variance(v);
//    }
//    template<typename Matrix, typename Vector, typename Scalar>
//    double init_eigen_util<Matrix, Vector, Scalar>::average_init(const Vector &v){
//        return average(v);
//    };
//
//    template<typename Matrix, typename Vector, typename Scalar>
//    void init_eigen_util<Matrix, Vector, Scalar>::rescale_init(Matrix & x, const Scalar scale){
//        rescale(x, scale);
//    };



    template<typename Matrix>
    void standardize(Matrix& X){
        for (int col =0;col<X.cols();col++){
            double mean = average(X.col(col));
            double std = sqrt(variance(X.col(col)));
            for (int row=0;row<X.rows();row++){
                X(row, col) = (X(row, col)-mean)/std;
            }
        }
    }
    template<typename Vector>
    double variance(const Vector & v){
        double result= 0.0;
        double mean = average(v);
        for (int i =0 ;i<v.rows();i++){
            result += (v(i)-mean)*(v(i)-mean);
        }

        return result/v.rows();
    }
    template<typename Vector>
    double average(const Vector &v){
        double result = 0.0;
        for (int i = 0;i<v.rows();i++){
            result+= v(i);
        }
        return result/v.rows();
    }

    template<typename Matrix, typename Scalar>
    void rescale(Matrix & matrix, const Scalar scale){
        for (int i=0;i<matrix.rows();i++){
            for (int j=0;j<matrix.cols();j++)
            {
                matrix(i,j)=matrix(i,j)*scale;
            }
        }
    }

    template void standardize<MatrixXd>(MatrixXd &x);
    template double variance<VectorXd>(const VectorXd & v);
    template double average<VectorXd>(const VectorXd& v);
    template void rescale<MatrixXd, double>(MatrixXd&, const double);





}