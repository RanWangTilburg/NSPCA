from __future__ import division
import numpy as np
from collections import namedtuple
from recordtype import recordtype

NSPCASolutionPy = recordtype('NSPCASolution', ['transformed_score', 'principal_score', 'component_loading'])


class NSPCASolverPy:
    def __init__(self, no_obs, no_var, reduced_dim, scale=1.0):

        self.scale = scale
        self.reduced_dim = reduced_dim
        self.no_obs = no_obs
        self.no_var = no_var

    def init_svd(self, data, restriction=None):
        """
        Initialize the solution with svd decomposition
        :param data: data for fitting, must be of dimension N times P
        :param restriction:  restrictions. If is None, then no restriction is imposed. Otherwise, must be P times p
        :return: NSPCASolution consisting of initial value
        """
        if data.shape[0] == self.no_obs:
            print "Number of observations inconsistent"
            return None
        elif data.shape[1] == self.no_var:
            print "Number of variables mismatches"
            return None

        init_transformed_score = self.normalize_transformed_score(data)
        [u, diag, v] = np.linalg.svd(data, full_matrices=False)
        column_number = self.reduced_dim - 1
        init_principal_score = u[:, 0:(self.reduced_dim - 1)]
        init_component_loading = np.diag(diag)[0:column_number, 0:column_number].dot(v[0:column_number, :])

        if restriction is None:
            return NSPCASolutionPy(init_transformed_score, init_principal_score, init_component_loading)
        elif restriction.shape[0] != self.reduced_dim or restriction.shape[1] != self.no_var:
            print "Warning: The dimension of restrictions do not match; discarding restrictions"
            return NSPCASolutionPy(init_transformed_score, init_principal_score, init_component_loading)
        else:
            init_component_loading = self.check_signs(init_component_loading, restriction)
            return NSPCASolutionPy(init_transformed_score, init_principal_score, init_component_loading)

    def normalize_transformed_score(self, data):
        init_transformed_score = np.copy(data)
        for i in range(0, data.shape[1]):
            init_transformed_score[:, i] = (init_transformed_score[:, i] - np.mean(init_transformed_score[:, i])) * (
                self.scale / np.sqrt(np.var(init_transformed_score[:, i])))

        return init_transformed_score

    def check_signs(self, init_component_loading, restriction):
        [nrow, ncol] = restriction.data.shape
        for row in range(0, nrow):
            for col in range(0, ncol):
                if restriction[row, col] == -1 and init_component_loading[row, col] > 0:
                    init_component_loading[row, col] = - init_component_loading[row, col]
                elif restriction[row, col] == 1 and init_component_loading[row, col] < 0:
                    init_component_loading[row, col] = -init_component_loading[row, col]
                elif restriction[row, col] == 0 and init_component_loading[row, col] != 0:
                    init_component_loading[row, col] = 0
        return init_component_loading

    #
    # def fit(self,  data,  penalty,
    #         np.ndarray[int, mode="fortran", ndim=2] restriction = None, NSPCASolution init = None, const int max_iter = 100,
    #         const double convergence_threshold = 0.001):
    #     """
    #     Fitting a NSPCA model using the iterated algorithm for a fixed penalty parameter
    #     :param data: Data to fit. Must be of dimension N times P
    #     :param penalty: the penalty (lambda) of the model, must be nonnegative
    #     :param restriction: a P times p matrix expressing the constraints. If None, no restriction is imposed.
    #     :param init: NSPCASolution that gives an initial value for the algorithm. If none, then using the
    #     previous solution in the GPU
    #     :param max_iter: maximum number of iterations. must be larger than 1
    #     :param convergence_threshold: the threshold to test convergence, must be larger than 0
    #     :return: if the algorithm converges, return a NSPCASolution contains the solution; else, return "None"
    #     """
    #
    #     self.dimension_check_in_fitting(convergence_threshold, init, max_iter, penalty)
    #
    #     if init is not None and restriction is not None:
    #         success = self.solver.init(&data[0, 0], &restriction[0, 0], penalty,
    #                                    init.get_transformed_score(), init.get_principal_score(),
    #                                    init.get_component_loading())
    #         if success is not True:
    #             print "Initialization failed"
    #             return None
    #
    #     cdef size_t iter = 0
    #     cdef bool converged = False
    #
    #     while iter < max_iter and not converged:
    #         self.solver.copy_old_component_loadings()
    #         self.solver.solve_principal_score()
    #         self.solver.solve_transformed_score()
    #         self.solver.solve_component_loadings()
    #
    #         converged = self.solver.check_convergence_on_component_loadings(convergence_threshold)
    #         iter += 1
    #         if iter >= max_iter:
    #             print "Warning: Algorithm did not converge, return results from last iteration"
    #
    #     cdef np.ndarray[double, mode ="fortran", ndim =2] transformed_score_solution = np.zeros(
    #         (self.no_obs, self.no_var), order="fortran")
    #     cdef np.ndarray[double, mode ="fortran", ndim =2] principal_score_solution = np.zeros(
    #         (self.no_obs, self.reduced_dim), order="fortran")
    #     cdef np.ndarray[double, mode ="fortran", ndim =2] component_loading_solution = np.zeros(
    #         (self.reduced_dim, self.no_var), order="fortran")
    #
    #     self.solver.get_result_from_gpu(&transformed_score_solution[0, 0], &principal_score_solution[0, 0],
    #                                     &component_loading_solution[0, 0])
    #     return NSPCASolution(transformed_score_solution, principal_score_solution, component_loading_solution)

    def dimension_check_in_fitting(self, convergence_threshold, init, max_iter, penalty):
        if init is not None:
            assert (
                isinstance(NSPCASolutionPy,
                           init)), "Error in specifying initial values, must be a class of NSPCASolution"
            assert (init.transformed_score == [self.no_obs,
                                               self.no_var]), "The dimension of initial transformed score mismatches"
            assert (init.principal_score == [self.no_var,
                                             self.reduced_dim]), "The dimension of initial principal score mismatches"
            assert (init.component_loading == [self.reduced_dim,
                                               self.no_var]), "The dimension of initial component loading mismatches"
        assert (max_iter > 1), "The maximum number of iterations must be larger than 1"
        assert (convergence_threshold > 0), "The threshold of testing convergence must be larger than 0"
        assert (penalty >= 0), "The penalty must be non-negative"



    def check_convergence(self, threshold):
        diff = np.linalg.norm(self.old_component_loading - self.component_loading)

        if diff > threshold:
            return False
        else:
            return True
