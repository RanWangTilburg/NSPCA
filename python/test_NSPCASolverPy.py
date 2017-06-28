from unittest import TestCase
import unittest
from montecarlo import monte_carlo_tel
from montecarlo import monte_carlo_results
from nspcasolverimpl import assert_present
from nspcasolverimpl import NSPCASolverImpl
from NSPCASolverV2 import NSPCASolverPy
from NSPCASolverV2 import init_svd
from NSPCASolverV2 import NSPCASolution
from nspcagpu import is_similar

import numpy as np


class TestNSPCASolverPy(TestCase):
    no_obs = 100
    no_var = 10
    reduced_dim = 2
    scale = 2.0
    rep_monte_carlo = 50
    thres = 0.01

    @unittest.skip("initialization and counting incidence")
    def test_initialization(self):
        is_same = True

        for rep in range(self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            alpha = np.random.rand()
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                solver_v1 = NSPCASolverImpl(self.no_obs, self.no_var, self.reduced_dim, self.scale)
                solver_v2 = NSPCASolverPy(self.no_obs, self.no_var, self.reduced_dim, self.scale, 2, 2)

                init_value = init_svd(data, self.reduced_dim)
                solver_v1.init_with_data(data)
                solver_v2.init(data)

                cumulation_v1 = solver_v1.get_incidence_count()
                cumulation_v2 = solver_v2.get_incidence_count()

                is_same = is_same and is_similar(cumulation_v1, cumulation_v2)

        self.assertTrue(is_same)

    @unittest.skip("cumulating score")
    def test_cumulate_score(self):
        is_same = True

        for rep in range(self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            alpha = np.random.rand()
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                solver_v1 = NSPCASolverImpl(self.no_obs, self.no_var, self.reduced_dim, self.scale)
                solver_v2 = NSPCASolverPy(self.no_obs, self.no_var, self.reduced_dim, self.scale, 2, 2)

                init_value = init_svd(data, self.reduced_dim)
                solver_v1.init_with_data(data)
                solver_v2.init(data)

                solver_v1.solve_transformed_score(data)
                solver_v2.transformed_score(4, 4)

                cumu_score_v1 = solver_v1.get_cumu_score()
                cumu_score_v2 = solver_v2.get_cumu_score()

                is_same = is_same and is_similar(cumu_score_v1, cumu_score_v2)

        self.assertTrue(is_same)

    @unittest.skip("Test for transformed score")
    def test_transformed_score(self):
        is_same = True

        for rep in range(self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            alpha = np.random.rand()
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                solver_v1 = NSPCASolverImpl(self.no_obs, self.no_var, self.reduced_dim, self.scale)
                solver_v2 = NSPCASolverPy(self.no_obs, self.no_var, self.reduced_dim, self.scale, 2, 2)

                init_value = init_svd(data, self.reduced_dim)
                solver_v1.init_with_data(data)
                solver_v2.init(data)

                solver_v1.solve_transformed_score(data)
                solver_v2.transformed_score(4, 4)

                transformed_score_v1 = solver_v1.transformed_score
                transformed_score_v2 = solver_v2.get_transformed_score()

                is_same = is_same and is_similar(transformed_score_v1, transformed_score_v2)

        self.assertTrue(is_same)

    # @unittest.skip("Test for principal score")
    def test_principal_score(self):
        is_same = True

        for rep in range(self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            alpha = np.random.rand()
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                solver_v1 = NSPCASolverImpl(self.no_obs, self.no_var, self.reduced_dim, self.scale)
                solver_v2 = NSPCASolverPy(self.no_obs, self.no_var, self.reduced_dim, self.scale, 2, 2)

                init_value = init_svd(data, self.reduced_dim)
                solver_v1.init_with_data(data)
                solver_v2.init(data)

                solver_v1.solve_transformed_score(data)
                solver_v2.transformed_score(4,4)

                solver_v1.solve_principal_score()
                solver_v2.principal_score()

                principal_score_v1 = solver_v1.principal_score
                principal_score_v2 = solver_v2.get_principal_score()
                # print principal_score_v1.shape
                # print principal_score_v2.shape
                is_same = is_same and  is_similar(principal_score_v1, principal_score_v2)
                # print "In the result "
                # print principal_score_v1
                # print principal_score_v2
                # print principal_score_v2-principal_score_v1


        self.assertTrue(is_same)

    @unittest.skip("Test for component loading")
    def test_component_loading(self):
        is_same = True

        for rep in range(self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            alpha = np.random.rand()
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                solver_v1 = NSPCASolverImpl(self.no_obs, self.no_var, self.reduced_dim, self.scale)
                solver_v2 = NSPCASolverPy(self.no_obs, self.no_var, self.reduced_dim, self.scale, 2, 2)

                # init_value = init_svd(data, self.reduced_dim)
                solver_v1.init_with_data(data)
                solver_v2.init(data)

                solver_v1.set_alpha(alpha)
                solver_v2.set_alpha(alpha)
                solver_v1.solve_transformed_score(data)
                solver_v2.transformed_score(4,4)

                solver_v1.solve_component_loading()
                solver_v2.component_loading(4,4)
                # print "here"
                component_loading_v1 = solver_v1.component_loading
                component_loading_v2 = solver_v2.get_component_loading()
                # print component_loading_v2

                is_same = is_same and is_similar(component_loading_v1, component_loading_v2)

        self.assertTrue(is_same)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(TestNSPCASolverPy))
