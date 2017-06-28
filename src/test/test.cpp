#include <gtest/gtest.h>
#include "../cxxsrc/Solver.h"
#include <iostream>
#include "../cxxsrc/Threadpool/thread_pool.h"
#include "../cxxsrc/util.h"
#include "../custat/cuStat.h"
#include "../cxxsrc/test_util.h"
//#include "../custat/cuStatHost.h"


class test_init : public ::testing::Test {
public:

};

TEST_F(test_init, DISABLED_TESTING_FOR_INIT) {
    NSPCA::Solver solver(100, 10, 2, 2.0, 2, 2);
    std::cout << "flag " << std::endl;
    ASSERT_TRUE(true);
}

//TEST_F(test_init, DISABLED_TESTING_FOR_INIT_WITH_DATA) {
//    NSPCA::Solver solver(10, 4, 2, 2.0, 2, 2);
//    NSPCA::test_util util(10, 4, 2);
//
//    solver.init(util.get_data(), util.get_res(), util.get_ts(), util.get_ps(), util.get_cl());
//
//    auto thres = 0.1;
//    auto result1 = cuStat::is_approx(solver.solutionG.transformed_score, util.transformed_score, thres);
//    auto result2 = cuStat::is_approx(solver.solutionG.principal_score, util.principal_score, thres);
//    auto result3 = cuStat::is_approx(solver.solutionG.component_loading, util.component_loading, thres);
//    auto result4 = cuStat::is_approx(solver.data_gpu, util.data, thres);
//    auto result5 = cuStat::is_approx(solver.restriction_gpu, util.restriction, thres);
//
//    auto result = result1 && result2 && result3 && result4 && result5;
//    ASSERT_TRUE(result);
//}

TEST_F(test_init, TESTING_TRANSFORMED_SCORE) {
    NSPCA::Solver solver(10, 4, 2, 2.0, 2, 2);
    NSPCA::test_util util(10, 4, 2);

    solver.init(util.get_data(), util.get_res(), util.get_ts(), util.get_ps(), util.get_cl());
    solver.transformed_score(1, 1);

    ASSERT_TRUE(true);
}

TEST_F(test_init, TESTING_FOR_COMPONENT_LOADING) {
    NSPCA::Solver solver(100, 10, 2, 2.0, 2, 2);
    NSPCA::test_util util(100, 10, 2);

    solver.init(util.get_data(), util.get_res(), util.get_ts(), util.get_ps(), util.get_cl());
    solver.component_loading(4, 4);
    ASSERT_TRUE(true);
}

TEST_F(test_init, TESTING_FOR_PRINCIPAL_SCORE) {
    NSPCA::Solver solver(100, 10, 2, 2.0, 2, 2);
    NSPCA::test_util util(100, 10, 2);

    solver.init(util.get_data(), util.get_res(), util.get_ts(), util.get_ps(), util.get_cl());
    solver.principal_score();
    ASSERT_TRUE(true);
}

TEST_F(test_init, DISABLED_TESTING_FOR_COUNT) {
    NSPCA::internal::IncidenceCount count(10);
    ASSERT_TRUE(true);
}

TEST_F(test_init, DISABLED_TESTING_FOR_TEST_UTIL) {
    NSPCA::test_util util(100, 10, 2);
//    std::cout << util.data << std::endl;
    ASSERT_TRUE(true);
}
//TEST_F(test_init, TESTING_FOR_SOLVER){
//    auto solver = cuStat::linSolver();
//    ASSERT_TRUE(true);
//}
//TEST_F(test_init, TESTING_FOR_THREAD_POOL){
//    auto pool = NSPCA::internal::pools(2, 2);
//    ASSERT_TRUE(true);
//}
