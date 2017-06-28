from unittest import TestCase
import unittest
from nspcasolverimpl import NSPCASolverImpl
from nspcasolverimpl import standardize
from montecarlo import monte_carlo_tel
import numpy as np
from nspcasolverimpl import assert_present
from math import sqrt
from nspcasolverimpl import get_sol_from_a2


def loss(transformed_score, principal_score, component_loading, alpha=0.0):
    assert isinstance(transformed_score, np.ndarray)
    assert isinstance(principal_score, np.ndarray)
    assert isinstance(component_loading, np.ndarray)

    assert alpha >= 0
    assert transformed_score.ndim == 2
    assert principal_score.ndim == 2
    assert component_loading.ndim == 2

    assert transformed_score.shape[0] == principal_score.shape[0]
    assert transformed_score.shape[1] == component_loading.shape[1]
    assert principal_score.shape[1] == component_loading.shape[0]

    no_obs = transformed_score.shape[0]

    value = np.linalg.norm(transformed_score - principal_score.dot(component_loading), 'fro')
    value *= value/no_obs
    value += alpha *np.sum(np.sum(np.abs(component_loading)))
    return value


class TestNSPCASolverImpl(TestCase):
    no_obs = 100
    no_var = 3
    reduced_dim = 2
    scale = 2.0
    solver = NSPCASolverImpl(no_obs, no_var, reduced_dim, scale)

    rep_monte_carlo = 100
    rep_random_solution = 100

    thres = 0.001

    @unittest.skip("Normalization for Transformed Score")
    def test_transformed_score_restriction(self):
        is_properly_standardized = True
        for reps in range(0, self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                self.solver.init_with_data(data)
                self.solver.AP = np.random.randn(self.no_obs, self.no_var)
                self.solver.solve_transformed_score(data)

                for col in range(0, self.no_var):
                    mean = np.mean(self.solver.transformed_score[:, col])
                    # print mean
                    std = np.std(self.solver.transformed_score[:, col])
                    # print std
                    if np.abs(mean) > self.thres or np.abs(std - self.scale) > self.thres:
                        is_properly_standardized = False

        self.assertTrue(is_properly_standardized)

    # @unittest.skip("Minimization Transformed Score")
    def test_transformed_score_minimization(self):
        is_minimized = True
        for reps in range(0, self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                self.solver.init_with_data(data)
                self.solver.AP = np.random.randn(self.no_obs, self.no_var)
                self.solver.init_with_data(data)
                self.solver.solve_transformed_score(data)

                value_tel = loss(self.solver.transformed_score, self.solver.principal_score, self.solver.component_loading)
                col = 1
                n = self.solver.incidence_count[0, col]
                e = self.solver.incidence_count[1, col]
                p = self.solver.incidence_count[2, col]
                tn = self.solver.incidence_count_score[0, col]
                te = self.solver.incidence_count_score[1, col]
                tp = self.solver.incidence_count_score[2, col]

                s = self.scale
                N = self.no_obs
                sqrt_n = sqrt(self.no_obs)
                upper = sqrt((n + p) * N * s * s / (n * p))
                a2 = 0.0
                step = upper / self.rep_random_solution

                for i in range(0, self.rep_random_solution):
                    a2 += step
                    if a2 < upper - 0.1:
                        result = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2)

                        self.solver.write_transformed_score(col, data, result)
                        value_random = loss(self.solver.transformed_score, self.solver.principal_score, self.solver.component_loading)

                        if value_tel > value_random + 1.0:
                            is_minimized = False
        self.assertTrue(is_minimized)

    # @unittest.skip("Normalization Test for Solving Principal Scores")
    def test_principal_score_restriction(self):
        product_is_diagonal_matrix = True
        identity = sqrt(self.no_obs) * self.scale* np.identity(self.reduced_dim)
        for reps in range(0, self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                self.solver.transformed_score = data
                self.solver.component_loading = beta

                self.solver.transformed_score = standardize(self.solver.transformed_score, self.scale * self.scale)
                self.solver.solve_principal_score()

                product = self.solver.principal_score.T.dot(self.solver.principal_score)
                diff = np.linalg.norm(product - identity)

                if np.abs(diff) > self.thres:
                    product_is_diagonol = False

        self.assertTrue(product_is_diagonal_matrix)

    # @unittest.skip("Minimization for Principal Score")
    def test_principal_score_minimization(self):
        is_smallest = True

        for reps in range(0, self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                self.solver.transformed_score = data
                self.solver.component_loading = beta

                self.solver.transformed_score = standardize(self.solver.transformed_score, self.scale)
                self.solver.solve_principal_score()

                loss_tel = loss(self.solver.transformed_score, self.solver.principal_score, self.solver.component_loading)
                for random_rep in range(0, self.rep_random_solution):
                    solution_random = np.random.randn(self.no_obs, self.reduced_dim)
                    solution_random, r = np.linalg.qr(solution_random)
                    solution_random *= sqrt(self.no_obs)*self.scale

                    loss_random = loss(self.solver.transformed_score, self.solver.principal_score, self.solver.component_loading)
                    if loss_tel < loss_random - 1.0:
                        is_smallest = False
        self.assertTrue(is_smallest)

    # @unittest.skip("Minimization for Component Loading")
    def test_component_loading(self):
        is_smallest = True

        for reps in range(0, self.rep_monte_carlo):
            beta = np.random.randn(self.reduced_dim, self.no_var)
            data = monte_carlo_tel(self.no_obs, beta, 0.5, -0.5).data
            if assert_present(data):
                alpha = np.random.rand()*10.0
                # alpha = 0.0
                self.solver.set_alpha(alpha)
                self.solver.transformed_score = data
                q, r = np.linalg.qr(np.random.randn(self.no_obs, self.reduced_dim))

                self.solver.principal_score = (sqrt(self.no_obs) * self.scale) * q
                product = self.solver.principal_score.T.dot(self.solver.principal_score)
                self.solver.component_loading = np.zeros((self.reduced_dim, self.no_var))

                self.solver.transformed_score = standardize(self.solver.transformed_score, self.scale)
                self.solver.solve_component_loading()
                product1 = self.solver.transformed_score - self.solver.principal_score.dot(
                    self.solver.component_loading)
                value_tel = np.trace(product1.T.dot(product1)) / self.no_obs + alpha * np.sum(
                    np.sum(np.absolute(self.solver.component_loading)))
                # value_loss = loss(self.solver.transformed_score, self.solver.principal_score, self.solver.component_loading, alpha)
                # assert abs(value_tel-value_loss) < 0.01
                for rep_random in range(0, self.rep_random_solution):
                    # random_solution = self.solver.principal_score.T.dot(self.solver.transformed_score) / (
                    # self.no_obs * self.scale * self.scale)
                    random_solution = np.random.randn(self.reduced_dim, self.no_var)
                    product2 = self.solver.transformed_score - self.solver.principal_score.dot(
                        random_solution)
                    value_random = np.trace(product2.T.dot(product2)) / self.no_obs + alpha * np.sum(
                        np.sum(np.absolute(random_solution)))

                    if value_tel > value_random + 1.0:
                        is_smallest = False
                        print "Solution is "
                        print self.solver.component_loading
                        print "With value"
                        print value_tel
                        print "Random solution is "
                        print random_solution
                        print "with value"
                        print value_random

                if is_smallest is False:
                    print
        self.assertTrue(is_smallest)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(TestNSPCASolverImpl))
