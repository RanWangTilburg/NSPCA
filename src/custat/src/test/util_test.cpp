#include <iostream>

using std::cout; using std::endl;

#include <cstdlib>
#include <vector>

using std::vector;

#include <gtest/gtest.h>

#include "util.h"
#include "predeclare.h"

class UtilTest : public ::testing::Test {
public:
    vector<int> nrows = {1, 2, 4, 50, 100, 1000, 1024, 5000};
    vector<int> ncols = {1, 2, 4, 20, 50, 100, 1000, 1024, 5000};
    vector<int> ms = {2,4,10,50,144,300};
    vector<int> ns = {2,4,10,50,144,300};
    vector<int> ks = {2,4,10,50,144,300};
};
////Sanity check for copying to and from eigen
////As well as assert
TEST_F(UtilTest, DISABLED_TEST_EIGEN_INTEROP) {
    cuStat::Matrix<double> dev_matrix(10, 10);
    Eigen::MatrixXd host_matrix = Eigen::MatrixXd::Random(10, 10);
    cout << "Prior to copying " << endl << host_matrix << endl;
    copy_from_eigen(host_matrix, dev_matrix);
    copy_to_eigen(host_matrix, dev_matrix);
    cout << "After copying " << endl << host_matrix << endl;

    bool result = assert_near(host_matrix, dev_matrix, 0.01);
    ASSERT_TRUE(result);
}

TEST_F(UtilTest, DISABLED_TEST_FOR_CONSTANT_INIT) {
    cout << "Testing initializing matrix using thrust api" << endl;
    cout << "Progress: " << endl;
    bool result = true;

    double total_size = nrows.size() * ncols.size();
    pBar p;
    for (auto row_step : nrows) {
        for (auto col_step : ncols) {
            p.update(100 / total_size);
            p.print();
            Eigen::MatrixXd host_mat = Eigen::MatrixXd::Constant(row_step, col_step, 1.0);
            cuStat::Matrix<double> dev_mat(row_step, col_step, 1.0);
            result = result && assert_near(host_mat, dev_mat, 0.0001);

        }
    }
    p.complete();
    cout << endl;
    ASSERT_TRUE(result);

}

TEST_F(UtilTest, DISABLED_TEST_FOR_CONSTANT_INIT_ASYNC) {
    cout << "Testing initializing matrix using own kernel" << endl;
    cout << "Progress: " << endl;
    bool result = true;
    double total_size = nrows.size() * ncols.size();
    pBar p;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (auto row_step : nrows) {
        for (auto col_step : ncols) {
            p.update(100 / total_size);
            Eigen::MatrixXd host_mat = Eigen::MatrixXd::Constant(row_step, col_step, 1.0);
            cuStat::Matrix<double> dev_mat(row_step, col_step, 1.0, stream, 64, 500);
            cudaStreamSynchronize(stream);
            result = result && assert_near(host_mat, dev_mat, 0.0001);
            p.print();
        }
    }
    p.complete();
    cout << endl;
    cudaStreamDestroy(stream);
    ASSERT_TRUE(result);


}

TEST_F(UtilTest, DISABLED_TEST_PRINT_VECTOR) {
    cout << "Testing printing a vector" << endl;
    cuStat::Matrix<double> dev_vec(10, 1, 0.2);
    std::cout << dev_vec << std::endl;
    ASSERT_TRUE(true);
}

TEST_F(UtilTest, DISABLED_TEST_PRINT_MATRIX) {
    cout << "Testing printing a matrix" << endl;
    cuStat::Matrix<double> dev_mat(10, 2, 0.2);
    std::cout << dev_mat << std::endl;
    ASSERT_TRUE(true);
}

////Test fill the matrix with random normal variables
TEST_F(UtilTest, DISABLED_TEST_RANDN) {
    try {
        cout << "Testing initializing a matrix with random normal numbers " << endl;
        cuStat::Matrix<double> dev_mat(100, 10);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        dev_mat.randn(10, 0.0, 1.0, stream, 64, 10);
        cudaStreamSynchronize(stream);
        std::cout << dev_mat << std::endl;
//        throw 5;
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }
    ASSERT_TRUE(true);
}

TEST_F(UtilTest, DISABLED_TEST_RAND) {
    try {
        cout << "Testing initializing a matrix with uniform random numbers " << endl;
        cuStat::Matrix<double> dev_mat(100, 10);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        dev_mat.rand(10, 0.0, 1.0, stream, 64, 1);
        cudaStreamSynchronize(stream);
        std::cout << dev_mat << std::endl;
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }

    ASSERT_TRUE(true);
}

//TEST_F(UtilTest, TEST_PASSING_CLASSES){
//    cuStat::Matrix<double> a(10,2,0.0);
//    auto b = a.view();
//    cuStat::test_pass_classes(b);
//
//    std::cout << a << std::endl;
//    ASSERT_TRUE(true);
//}

TEST_F(UtilTest, DISABLED_TEST_MATMUL_NN_DOUBLE) {
    try {
        cout << "Testing matrix product with (n-tran, n-tran)" << endl;
        bool result = true;
        pBar p;
        double size = ms.size() * ns.size() * ks.size();
        cuStat::linSolver solver;
        for (auto m : ms) {
            for (auto n:ns) {
                for (auto k:ks) {
                    p.update(100 / size);
                    p.print();
                    Eigen::MatrixXd lhs = Eigen::MatrixXd::Random(m, k);
                    Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(k, n);
                    Eigen::MatrixXd result_host = lhs * rhs;

                    cuStat::Matrix<double> lhs_dev(m, k);
                    cuStat::Matrix<double> rhs_dev(k, n);
                    cuStat::Matrix<double> result_dev(m, n);

                    copy_from_eigen(lhs, lhs_dev);
                    copy_from_eigen(rhs, rhs_dev);
                    solver.matmul_nn(1.0, lhs_dev, rhs_dev, 0.0, result_dev);
                    solver.sync();

                    result = result && assert_near(result_host, result_dev, 0.01);


                }
            }
        }

        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }


}

TEST_F(UtilTest, DISABLED_TEST_MATMUL_NT_DOUBLE) {
    try {
        cout << "Testing matrix product with (n-tran, tran)" << endl;
        bool result = true;
        pBar p;
        double size = ms.size() * ns.size() * ks.size();
        cuStat::linSolver solver;
        for (auto m : ms) {
            for (auto n:ns) {
                for (auto k:ks) {
                    p.update(100 / size);
                    p.print();
                    Eigen::MatrixXd lhs = Eigen::MatrixXd::Random(m, k);
                    Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(n, k);
                    Eigen::MatrixXd result_host = lhs * rhs.transpose();

                    cuStat::Matrix<double> lhs_dev(m, k);
                    cuStat::Matrix<double> rhs_dev(n, k);
                    cuStat::Matrix<double> result_dev(m, n);

                    copy_from_eigen(lhs, lhs_dev);
                    copy_from_eigen(rhs, rhs_dev);


                    solver.matmul_nt(1.0, lhs_dev, rhs_dev, 0.0, result_dev);
                    solver.sync();

                    result = result && assert_near(result_host, result_dev, 0.01);


                }
            }
        }

        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }

}

TEST_F(UtilTest, DISABLED_TEST_MATMUL_TN_DOUBLE) {
    try {
        cout << "Testing matrix product with (tran, n-tran)" << endl;
        bool result = true;
        pBar p;
        double size = ms.size() * ns.size() * ks.size();
        cuStat::linSolver solver;
        for (auto m : ms) {
            for (auto n:ns) {
                for (auto k:ks) {
                    p.update(100 / size);
                    p.print();
                    Eigen::MatrixXd lhs = Eigen::MatrixXd::Random(k, m);
                    Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(k, n);
                    Eigen::MatrixXd result_host = lhs.transpose()*rhs;

                    cuStat::Matrix<double> lhs_dev(k, m);
                    cuStat::Matrix<double> rhs_dev(k, n);
                    cuStat::Matrix<double> result_dev(m, n);

                    copy_from_eigen(lhs, lhs_dev);
                    copy_from_eigen(rhs, rhs_dev);


                    solver.matmul_tn(1.0, lhs_dev, rhs_dev, 0.0, result_dev);
                    solver.sync();

                    result = result && assert_near(result_host, result_dev, 0.01);


                }
            }
        }

        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }
}

TEST_F(UtilTest, DISABLED_TEST_MATMUL_TT_DOUBLE) {
    try {
        cout << "Testing matrix product with (tran, tran)" << endl;
        bool result = true;
        pBar p;
        double size = ms.size() * ns.size() * ks.size();
        cuStat::linSolver solver;
        for (auto m : ms) {
            for (auto n:ns) {
                for (auto k:ks) {
                    p.update(100 / size);
                    p.print();
                    Eigen::MatrixXd lhs = Eigen::MatrixXd::Random(k, m);
                    Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(n,k);
                    Eigen::MatrixXd result_host = lhs.transpose()*rhs.transpose();

                    cuStat::Matrix<double> lhs_dev(k, m);
                    cuStat::Matrix<double> rhs_dev(n, k);
                    cuStat::Matrix<double> result_dev(m, n);

                    copy_from_eigen(lhs, lhs_dev);
                    copy_from_eigen(rhs, rhs_dev);


                    solver.matmul_tt(1.0, lhs_dev, rhs_dev, 0.0, result_dev);
                    solver.sync();

                    result = result && assert_near(result_host, result_dev, 0.01);


                }
            }
        }

        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }
}


TEST_F(UtilTest, DISABLED_TEST_MATMUL_NN_FLOAT) {
    try {
        cout << "Testing matrix product with (n-tran, n-tran)" << endl;
        bool result = true;
        pBar p;
        float size = ms.size() * ns.size() * ks.size();
        cuStat::linSolver solver;
        for (auto m : ms) {
            for (auto n:ns) {
                for (auto k:ks) {
                    p.update(100 / size);
                    p.print();
                    Eigen::MatrixXf lhs = Eigen::MatrixXf::Random(m, k);
                    Eigen::MatrixXf rhs = Eigen::MatrixXf::Random(k, n);
                    Eigen::MatrixXf result_host = lhs * rhs;

                    cuStat::Matrix<float> lhs_dev(m, k);
                    cuStat::Matrix<float> rhs_dev(k, n);
                    cuStat::Matrix<float> result_dev(m, n);

                    copy_from_eigen(lhs, lhs_dev);
                    copy_from_eigen(rhs, rhs_dev);
                    solver.matmul_nn((float)1.0, lhs_dev, rhs_dev, (float)0.0, result_dev);
                    solver.sync();

                    result = result && assert_near(result_host, result_dev, 0.01);


                }
            }
        }

        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }


}

TEST_F(UtilTest, DISABLED_TEST_MATMUL_NT_FLOAT) {
    try {
        cout << "Testing matrix product with (n-tran, tran)" << endl;
        bool result = true;
        pBar p;
        float size = ms.size() * ns.size() * ks.size();
        cuStat::linSolver solver;
        for (auto m : ms) {
            for (auto n:ns) {
                for (auto k:ks) {
                    p.update(100 / size);
                    p.print();
                    Eigen::MatrixXf lhs = Eigen::MatrixXf::Random(m, k);
                    Eigen::MatrixXf rhs = Eigen::MatrixXf::Random(n, k);
                    Eigen::MatrixXf result_host = lhs * rhs.transpose();

                    cuStat::Matrix<float> lhs_dev(m, k);
                    cuStat::Matrix<float> rhs_dev(n, k);
                    cuStat::Matrix<float> result_dev(m, n);

                    copy_from_eigen(lhs, lhs_dev);
                    copy_from_eigen(rhs, rhs_dev);


                    solver.matmul_nt((float)1.0, lhs_dev, rhs_dev, (float)0.0, result_dev);
                    solver.sync();

                    result = result && assert_near(result_host, result_dev, 0.01);


                }
            }
        }

        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }

}

TEST_F(UtilTest, DISABLED_TEST_MATMUL_TN_FLOAT) {
    try {
        cout << "Testing matrix product with (tran, n-tran)" << endl;
        bool result = true;
        pBar p;
        float size = ms.size() * ns.size() * ks.size();
        cuStat::linSolver solver;
        for (auto m : ms) {
            for (auto n:ns) {
                for (auto k:ks) {
                    p.update(100 / size);
                    p.print();
                    Eigen::MatrixXf lhs = Eigen::MatrixXf::Random(k, m);
                    Eigen::MatrixXf rhs = Eigen::MatrixXf::Random(k, n);
                    Eigen::MatrixXf result_host = lhs.transpose()*rhs;

                    cuStat::Matrix<float> lhs_dev(k, m);
                    cuStat::Matrix<float> rhs_dev(k, n);
                    cuStat::Matrix<float> result_dev(m, n);

                    copy_from_eigen(lhs, lhs_dev);
                    copy_from_eigen(rhs, rhs_dev);


                    solver.matmul_tn((float)1.0, lhs_dev, rhs_dev, (float)0.0, result_dev);
                    solver.sync();

                    result = result && assert_near(result_host, result_dev, 0.01);


                }
            }
        }

        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }
}

TEST_F(UtilTest, DISABLED_TEST_MATMUL_TT_FLOAT) {
    try {
        cout << "Testing matrix product with (tran, tran)" << endl;
        bool result = true;
        pBar p;
        float size = ms.size() * ns.size() * ks.size();
        cuStat::linSolver solver;
        for (auto m : ms) {
            for (auto n:ns) {
                for (auto k:ks) {
                    p.update(100 / size);
                    p.print();
                    Eigen::MatrixXf lhs = Eigen::MatrixXf::Random(k, m);
                    Eigen::MatrixXf rhs = Eigen::MatrixXf::Random(n,k);
                    Eigen::MatrixXf result_host = lhs.transpose()*rhs.transpose();

                    cuStat::Matrix<float> lhs_dev(k, m);
                    cuStat::Matrix<float> rhs_dev(n, k);
                    cuStat::Matrix<float> result_dev(m, n);

                    copy_from_eigen(lhs, lhs_dev);
                    copy_from_eigen(rhs, rhs_dev);


                    solver.matmul_tt((float)1.0, lhs_dev, rhs_dev, (float)0.0, result_dev);
                    solver.sync();

                    result = result && assert_near(result_host, result_dev, 0.01);


                }
            }
        }

        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...) {
        OUTPUT_FILE_LINE;
        std::terminate();
    }
}

TEST_F(UtilTest, DISABLED_TEST_THIN_SVD){
    try {
        bool result = true;
        pBar p;

        double size= ms.size()*(ms.size()+1)/2.0;

        cuStat::linSolver solver;
        for (auto m:ms){
            for (auto n:ns){
                if (m>=n) {
                    p.update(100 / size);
                    p.print();

                    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

                    Eigen::MatrixXd U = svd.matrixU();
                    Eigen::MatrixXd VT = svd.matrixV().transpose();

                    cuStat::MatrixXd Ad(m, n);
                    cuStat::MatrixXd Ud(m, m);
                    cuStat::MatrixXd VTd(n, n);
                    cuStat::MatrixXd Sd(n, 1);

                    copy_from_eigen(A, Ad);

                    cuStat::ViewXd UView(Ud.data(), Ud.rows(), n);

                    solver.thin_svd(Ad, Ud, VTd, Sd);
                    solver.sync();

                    result = result && assert_near_abs(U, UView, 0.01);
                    result = result && assert_near_abs(VT, VTd, 0.01);
                }
            }
        }
        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...){
        ASSERT_TRUE(false);
        OUTPUT_FILE_LINE;
        std::terminate();
    }
}

TEST_F(UtilTest, DISABLED_TEST_THIN_SVD_FLOAT){
    try {
        bool result = true;
        pBar p;

        double size= ms.size()*(ms.size()+1)/2.0;

        cuStat::linSolver solver;
        for (auto m:ms){
            for (auto n:ns){
                if (m>=n) {
                    p.update(100 / size);
                    p.print();

                    Eigen::MatrixXf A = Eigen::MatrixXf::Random(m, n);
                    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

                    Eigen::MatrixXf U = svd.matrixU();
                    Eigen::MatrixXf VT = svd.matrixV().transpose();

                    cuStat::MatrixXf Ad(m, n);
                    cuStat::MatrixXf Ud(m, m);
                    cuStat::MatrixXf VTd(n, n);
                    cuStat::MatrixXf Sd(n, 1);

                    copy_from_eigen(A, Ad);

                    cuStat::ViewXf UView(Ud.data(), Ud.rows(), n);

                    solver.thin_svd(Ad, Ud, VTd, Sd);
                    solver.sync();

                    result = result && assert_near_abs(U, UView, 0.01);
                    result = result && assert_near_abs(VT, VTd, 0.01);
                }
            }
        }
        p.complete();
        ASSERT_TRUE(result);
    }
    catch (...){
        ASSERT_TRUE(false);
        OUTPUT_FILE_LINE;
        std::terminate();
    }
}

//TEST_F(UtilTest, TEST_FOR_UNARY_OP){
//    try{
//        bool is_same = true;
//        Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(10, 10);
//        cuStat::MatrixXd matrix_gpu(10, 10);
//        copy_from_eigen(matrix, matrix_gpu);
//
//    }
//    catch (...){
//        ASSERT_TRUE(false);
//    }
//}

TEST_F(UtilTest, TEST_FOR_ADD_ONE){
    cuStat::test_add_one();
}