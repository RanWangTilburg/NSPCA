#pragma once

#include <Eigen/Dense>
#include <cstdlib>


namespace NSPCA {
    void get_data(Eigen::MatrixXi& data);

    class test_util {
    public:

        size_t no_row;
        size_t no_var;
        size_t reduced_dim;

        Eigen::MatrixXd transformed_score;
        Eigen::MatrixXd principal_score;
        Eigen::MatrixXd component_loading;

        Eigen::MatrixXi data;
        Eigen::MatrixXi restriction;

        test_util(size_t no_row, size_t no_var, size_t reduced_dim);

        double *get_ts() {
            return (double *) transformed_score.data();
        }

        double *get_ps() {
            return (double *) principal_score.data();
        }

        double *get_cl() {
            return (double *) component_loading.data();
        }

        int *get_data() {
            return (int *) data.data();
        }

        int *get_res() {
            return (int *) restriction.data();
        }
    };

    test_util::test_util(size_t no_row, size_t no_var, size_t reduced_dim) : no_row(no_row), no_var(no_var),
                                                                             reduced_dim(reduced_dim) {
        transformed_score = Eigen::MatrixXd::Random(no_row, no_var);
        principal_score = Eigen::MatrixXd::Random(no_row, reduced_dim);
        component_loading = Eigen::MatrixXd::Random(reduced_dim, no_var);
        data = Eigen::MatrixXi::Constant(no_row, no_var, 0);
        NSPCA::get_data(data);
        restriction = Eigen::MatrixXi::Constant(reduced_dim, no_var, 2);

    }

    void get_data(Eigen::MatrixXi& data){

        Eigen::MatrixXd temp = Eigen::MatrixXd::Random(data.rows(), data.cols());

        for (auto i =0;i<data.rows();i++){
            for (auto j=0;j<data.cols();j++){
                if (temp(i,j)>0.67){
                    data(i,j)=1;
                }
                else if (temp(i,j) < 0.33){
                    data(i,j)=-1;
                }
                else data(i,j)=0;
            }
        }
    }

}////End of Namespace NSPCA