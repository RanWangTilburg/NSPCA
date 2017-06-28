import numpy as np
from nspcasolver import NSPCASolutionPy
from recordtype import recordtype
from math import sqrt

solution_value = recordtype('sv', ['a0', 'a1', 'a2', 'v'])


def max_sv(a, b):
    if a.v > b.v:
        return a
    else:
        return b


def standardize(x, sd=1.0):
    """
    standardize a matrix by column
    :param x: the matrix to be standardized
    :param sd: standard deviation
    :return: standardized matrix
    """
    if not isinstance(x, np.ndarray):
        raise Exception('Type Mismatch, must be 1d or 2d numpy array')

    if x.ndim != 1 and x.ndim != 2:
        raise Exception('Input must be 1d or 2d array')

    if np.isnan(x).any():
        raise Exception('The input contains NaN')

    if np.isinf(x).any():
        raise Exception('The input contains inf')
    if x.ndim == 1:
        result = np.copy(x)
        std = np.std(result)
        if std == 0:
            raise Exception('The input contains a constant column')
        else:
            result = sd * (result - np.mean(result)) / np.std(result)
        return result
    elif x.ndim == 2:
        result = np.copy(x)
        for col in range(0, x.shape[1]):
            std = np.std(result[:, col])
            if std == 0.0:
                raise Exception('The input contains a constant column')
            else:
                result[:, col] = sd * (result[:, col] - np.mean(result[:, col])) / np.std(result[:, col])
        return result


def get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2):
    result = solution_value(a0=0.0, a1=0.0, a2=a2, v=0.0)
    temp = -a2 * a2 * n * p + (n + p) * N * s * s
    L = sqrt((temp) / (e * N))
    a0 = (-a2 * p + e * L) / (n + p)
    a1 = -L
    fvalue = tn * a0 + te * a1 + (a0 + a2) * tp
    a0other = -(a2 * p + e * L) / (n + p)
    a1other = L
    fvalue2 = tn * a0 + te * a1 + (a0 + a2) * tp

    # print "first"
    # print n*a0+e*a1+p*(a0+a2)
    # print "second"
    # print n*a0other+e*a1other+p*(a0other+a2)
    if fvalue > fvalue2:
        result.a0 = a0
        result.a1 = a1
        result.v = fvalue
    else:
        result.a0 = a0other
        result.a1 = a1other
        result.v = fvalue2
    return result

    # if (a2 <= upper & a2 >= 0) {
    # double L = sqrt((-a2 * a2 * n * p + (n + p) * s * s) / (e * N));
    # double a0 = (-a2 * p + e * L) / (n + p);
    # double a1 = -L;
    # double fvalue = tn * a0 + te * a1 + (a0 + a2) * tp;
    #
    # double a0other = -(a2 * p + e * L) / (n + p);
    # double a1other = L;
    # solution1.solution = a2;
    # double fvalue2 = tn * a0 + te * a1 + (a0 + a2) * tp;
    # if (fvalue > fvalue2) {
    #
    # solution1.solution1 = a0;
    # solution1.solution2 = a1;
    # solution1.Fvalue = fvalue;
    # } else {
    # solution1.solution1 = a0other;
    # solution1.solution2 = a1other;
    # solution1.Fvalue = fvalue2;
    # }
    # }


def get_sol_from_a2_upper(N, n, e, p, tn, te, tp, s, upper, a2):
    result = solution_value(a0=0.0, a1=0.0, a2=a2, v=0.0)
    a0 = (-a2 * p) / (n + p)
    a1 = 0
    fvalue = tn * a0 + (a0 + a2) * tp
    result.a0 = a0
    result.a1 = a1
    result.v = fvalue
    return result


def assert_present(data):
    if not isinstance(data, np.ndarray):
        raise Exception('Input must be numpy array')

    if data.ndim != 2:
        raise Exception('Input must be 2-dimensional')

    should_be = sorted([-1, 0, 1])
    for col in range(0, data.shape[1]):
        values = sorted(np.unique(data[:, col]))
        if values != should_be:
            print "Warning: the %s columns contains not only -1, 0, 1" % col
            print "It contains"
            print values
            return False

    return True


class NSPCASolverImpl:
    def __init__(self, no_obs, no_var, reduced_dim, scale=1.0, alpha=0.0):
        self.no_obs = no_obs
        self.no_var = no_var
        self.reduced_dim = reduced_dim
        self.scale = scale
        self.scale_square = scale * scale
        self.alpha = alpha
        self.n_alpha = alpha * no_obs
        self.square_n = sqrt(no_obs)
        self.transformed_score = None
        self.principal_score = None
        self.component_loading = None
        self.old_component_loading = None
        self.n_scale_square = scale * no_obs * scale
        self.ZPT = np.zeros((no_obs, reduced_dim))
        self.U = np.zeros((no_obs, reduced_dim))
        self.V = np.zeros((reduced_dim, reduced_dim))
        self.ATZ = np.zeros((reduced_dim, no_var))
        self.AP = np.zeros((no_obs, no_var))

        self.incidence_count = np.zeros((3, no_var))
        self.incidence_count_score = np.zeros((3, no_var))
        self.solution = None

    def get_incidence_count(self):
        return self.incidence_count

    def get_cumu_score(self):
        return self.incidence_count_score

    def init_with_data(self, data):
        self.count_incidence(data)
        self.solution = self.init_svd(data)
        self.transformed_score = self.solution.transformed_score
        self.component_loading = self.solution.component_loading
        self.principal_score = self.solution.principal_score

    def set_alpha(self, alpha):
        assert (alpha >= 0)
        self.alpha = alpha
        self.n_alpha = alpha * self.no_obs

    def init_svd(self, data, restriction=None):
        """
        Initialize the solution with svd decomposition
        :param data: data for fitting, must be of dimension N times P
        :param restriction:  restrictions. If is None, then no restriction is imposed. Otherwise, must be P times p
        :return: NSPCASolution consisting of initial value
        """
        if data.shape[0] != self.no_obs:
            print "Number of observations inconsistent"
            return None
        elif data.shape[1] != self.no_var:
            print "Number of variables mismatches"
            return None

        init_transformed_score = self.normalize_transformed_score(data)
        [u, diag, v] = np.linalg.svd(data, full_matrices=False)
        column_number = self.reduced_dim
        init_principal_score = u[:, 0:(self.reduced_dim)]

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
        init_transformed_score = np.copy(data.astype(np.float64))
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

    def count_incidence(self, data):
        self.incidence_count = np.zeros((3, self.no_var))
        for row in range(0, data.shape[0]):
            for col in range(0, data.shape[1]):
                if data[row, col] == -1:
                    self.incidence_count[0, col] += 1
                elif data[row, col] == 0:
                    self.incidence_count[1, col] += 1
                else:
                    self.incidence_count[2, col] += 1

    def count_incidence_score(self, data):
        self.incidence_count_score = np.zeros((3, self.no_var))
        for row in range(0, self.no_obs):
            for col in range(0, self.no_var):
                if data[row, col] == -1:
                    self.incidence_count_score[0, col] += self.AP[row, col]
                elif data[row, col] == 0:
                    self.incidence_count_score[1, col] += self.AP[row, col]
                else:
                    self.incidence_count_score[2, col] += self.AP[row, col]

    def copy_initial_values(self, initial_value):
        assert (isinstance(NSPCASolutionPy, initial_value)), "Must pass NSPCASolutionPy as initial values"
        self.transformed_score = np.copy(initial_value['Transformed_Score'])
        self.principal_score = np.copy(initial_value['Principal_Score'])
        self.component_loading = np.copy(initial_value['Component_Loading'])

    def get_solution(self):
        return NSPCASolutionPy(self.transformed_score, self.principal_score, self.component_loading)

    def copy_old_component_loading(self):
        self.old_component_loading = np.copy(self.component_loading)

    def solve_principal_score(self):
        self.ZPT = self.transformed_score.dot(self.component_loading.T)
        # print "ZPT from python is "
        # print self.ZPT
        [self.U, diag, self.V] = np.linalg.svd(self.ZPT, full_matrices=False)
        # print "U from python is "
        # print self.U
        # print "V transpose from python is"
        # print self.V.T
        # print "V inverse is"
        # print np.linalg.inv(self.V)
        # print self.square_n
        # print self.scale
        self.principal_score = self.square_n * self.scale * self.U.dot(np.linalg.inv(self.V))
        # print self.principal_score

    def solve_component_loading(self):
        self.ATZ = self.principal_score.T.dot(self.transformed_score)
        # print "From Python"
        # print self.ATZ
        for row in range(0, self.reduced_dim):
            for col in range(0, self.no_var):
                self.solve_component_loading_ij(row, col)

    def solve_component_loading_ij(self, row, col):
        temp_t = self.ATZ[row, col]
        if self.n_alpha - 2.0 * temp_t < 0:
            self.component_loading[row, col] = -(self.n_alpha - 2.0 * temp_t) / (2.0 * self.n_scale_square)
        elif self.n_alpha + 2.0 * temp_t < 0:
            self.component_loading[row, col] = (self.n_alpha + 2.0 * temp_t) / (2.0 * self.n_scale_square)
        else:
            self.component_loading[row, col] = 0.0

    def solve_transformed_score(self, data):
        self.AP = self.principal_score.dot(self.component_loading)
        self.count_incidence_score(data)
        for col in range(0, self.no_var):
            result = self.solve_transformed_score_j(col)

            self.write_transformed_score(col, data, result)

    def write_transformed_score(self, col, data, result):
        for row in range(0, self.no_obs):
            if data[row, col] == -1:
                self.transformed_score[row, col] = result.a0
            elif data[row, col] == 0:
                self.transformed_score[row, col] = result.a1
            else:
                self.transformed_score[row, col] = result.a2 + result.a0

    def solve_transformed_score_j(self, col):
        n = self.incidence_count[0, col]
        e = self.incidence_count[1, col]
        p = self.incidence_count[2, col]

        tn = self.incidence_count_score[0, col]
        te = self.incidence_count_score[1, col]
        tp = self.incidence_count_score[2, col]

        s = self.scale
        N = self.no_obs
        sqrt_n = self.square_n
        upper = sqrt((n + p) * N * s * s / (n * p))
        # Check the solution for a2 = 0.0
        a2 = 0.0
        sol1 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2)
        a2 = upper
        sol2 = get_sol_from_a2_upper(N, n, e, p, tn, te, tp, s, upper, a2)
        sol1 = max_sv(sol1, sol2)
        a2 = NSPCASolverImpl.a2_1(sqrt_n, N, n, e, p, tn, te, tp, s, upper)
        sol3 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2)
        sol1 = max_sv(sol1, sol3)

        a2 = NSPCASolverImpl.a2_2(sqrt_n, N, n, e, p, tn, te, tp, s, upper)
        sol4 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2)
        sol1 = max_sv(sol1, sol4)

        a2 = NSPCASolverImpl.a2_3(sqrt_n, N, n, e, p, tn, te, tp, s, upper)
        sol5 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2)
        sol1 = max_sv(sol1, sol5)

        a2 = NSPCASolverImpl.a2_4(sqrt_n, N, n, e, p, tn, te, tp, s, upper)
        sol6 = get_sol_from_a2(N, n, e, p, tn, te, tp, s, upper, a2)
        sol1 = max_sv(sol1, sol6)

        # print n*sol1.a0+e*sol1.a1+p*(sol1.a0+sol1.a2)
        # print n*sol1.a0*sol1.a0+e*sol1.a1*sol1.a1+p*(sol1.a0+sol1.a2)*(sol1.a0+sol1.a2)
        return sol1
        #
        # solution1.solution = 0.0;
        #
        # double a2 = solution1.solution;
        # getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution1, a2);
        # a2 = upper;
        # getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution2, a2);
        # solution1 = compareSolution(solution1, solution2);
        #
        # a2 = geta2sol1(N, n, e, p, tn, te, tp, s, upper);
        # getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution3, a2);
        # solution1 = compareSolution(solution1, solution3);
        # a2 = geta2Sol2(N, n, e, p, tn, te, tp, s, upper);
        # getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution4, a2);
        # solution1 = compareSolution(solution1, solution4);
        #
        # geta2sol3(N, n, e, p, tn, te, tp, s, upper);
        # getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution5, a2);
        # solution1 = compareSolution(solution1, solution5);
        #
        # a2 = geta2Sol4(N, n, e, p, tn, te, tp, s, upper);
        # getSolutionFromGivenA2(N, n, e, p, tn, te, tp, s, upper, solution6, a2);
        # solution1 = compareSolution(solution1, solution2);
        #
        # solutions_for_z_host(0, j) = solution1.solution1;
        # solutions_for_z_host(1, j) = solution1.solution2;
        # solutions_for_z_host(2, j) = solution1.solution;

    @staticmethod
    def a2_1(sqrt_n, N, n, e, p, tn, te, tp, s, upper):
        a2 = 0.0
        upperside = (e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp +
                     e * e * n * p * tp * tp + 2 * e * n * n * p * te * tn + 2 * e * n * n * p * te * tp +
                     N * e * n * n * tp * tp + 2 * e * n * p * p * te * tn +
                     2 * e * n * p * p * te * tp - 2 * N * e * n * p * tn * tp + N * e * p * p * tn * tn +
                     n * n * n * p * te * te + 2 * n * n * p * p * te * te + n * p * p * p * te * te);
        if upperside <= 0:
            a2 = 0.0
        else:
            temp = (sqrt_n * s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                  2 * e * e * n * p * tn * tp +
                                                                                  e * e * n * p * tp * tp +
                                                                                  2 * e * n * n * p * te * tn +
                                                                                  2 * e * n * n * p * te * tp +
                                                                                  N * e * n * n * tp * tp +
                                                                                  2 * e * n * p * p * te * tn +
                                                                                  2 * e * n * p * p * te * tp - 2 *
                                                                                  N * e *
                                                                                  n * p *
                                                                                  tn * tp +
                                                                                  N * e * p * p * tn * tn +
                                                                                  n * n * n * p * te * te +
                                                                                  2 * n * n * p * p * te * te +
                                                                                  n * p * p * p * te * te));
            if temp < 0 or temp > upper:
                a2 = 0.0
            else:
                a2 = temp

        return a2;

    @staticmethod
    def a2_2(sqrt_n, N, n, e, p, tn, te, tp, s, upper):
        a2 = 0.0
        upperside = N * e * n * p * (n + p) * (
            e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp +
            2 * e * n * n * p * te * tn + 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp +
            2 * e * n * p * p * te * tn + 2 * e * n * p * p * te * tp - 2 *
            N * e * n * p * tn * tp +
            N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
            n * p * p * p * te * te);
        if upperside <= 0:
            a2 = 0.0
        else:
            temp = -(sqrt_n * s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                   2 * e * e * n * p * tn * tp +
                                                                                   e * e * n * p * tp * tp +
                                                                                   2 * e * n * n * p * te * tn +
                                                                                   2 * e * n * n * p * te * tp +
                                                                                   N * e * n * n * tp * tp +
                                                                                   2 * e * n * p * p * te * tn +
                                                                                   2 * e * n * p * p * te * tp - 2 *
                                                                                   N * e *
                                                                                   n * p *
                                                                                   tn * tp +
                                                                                   N * e * p * p * tn * tn +
                                                                                   n * n * n * p * te * te +
                                                                                   2 * n * n * p * p * te * te +
                                                                                   n * p * p * p * te * te));
            if temp < 0 or temp > upper:
                a2 = 0.0
            else:
                a2 = temp

        return a2;

    @staticmethod
    def a2_3(sqrt_n, N, n, e, p, tn, te, tp, s, upper):
        a2 = 0.0
        upperside = N * e * n * p * (n + p) * (
            e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp -
            2 * e * n * n * p * te * tn - 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp -
            2 * e * n * p * p * te * tn - 2 * e * n * p * p * te * tp - 2 *
            N * e * n * p * tn * tp +
            N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
            n * p * p * p * te * te);
        if upperside <= 0:
            a2 = 0.0
        else:
            temp = sqrt_n * (s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                  2 * e * e * n * p * tn * tp +
                                                                                  e * e * n * p * tp * tp -
                                                                                  2 * e * n * n * p * te * tn -
                                                                                  2 * e * n * n * p * te * tp +
                                                                                  N * e * n * n * tp * tp -
                                                                                  2 * e * n * p * p * te * tn -
                                                                                  2 * e * n * p * p * te * tp - 2 *
                                                                                  N * e *
                                                                                  n * p *
                                                                                  tn * tp +
                                                                                  N * e * p * p * tn * tn +
                                                                                  n * n * n * p * te * te +
                                                                                  2 * n * n * p * p * te * te +
                                                                                  n * p * p * p * te * te));
            if temp < 0 or temp > upper:
                a2 = 0.0
            else:
                a2 = temp

        return a2;

    @staticmethod
    def a2_4(sqrt_n, N, n, e, p, tn, te, tp, s, upper):
        a2 = 0.0
        upperside = N * e * n * p * (n + p) * (
            e * e * n * p * tn * tn + 2 * e * e * n * p * tn * tp + e * e * n * p * tp * tp -
            2 * e * n * n * p * te * tn - 2 * e * n * n * p * te * tp + N * e * n * n * tp * tp -
            2 * e * n * p * p * te * tn - 2 * e * n * p * p * te * tp - 2 *
            N * e * n * p * tn * tp +
            N * e * p * p * tn * tn + n * n * n * p * te * te + 2 * n * n * p * p * te * te +
            n * p * p * p * te * te);
        if upperside <= 0:
            a2 = 0.0
        else:
            temp = -(sqrt_n * s * (n * tp - p * tn) * sqrt(upperside)) / (n * p * (e * e * n * p * tn * tn +
                                                                                   2 * e * e * n * p * tn * tp +
                                                                                   e * e * n * p * tp * tp -
                                                                                   2 * e * n * n * p * te * tn -
                                                                                   2 * e * n * n * p * te * tp +
                                                                                   N * e * n * n * tp * tp -
                                                                                   2 * e * n * p * p * te * tn -
                                                                                   2 * e * n * p * p * te * tp - 2 *
                                                                                   N * e *
                                                                                   n * p *
                                                                                   tn * tp +
                                                                                   N * e * p * p * tn * tn +
                                                                                   n * n * n * p * te * te +
                                                                                   2 * n * n * p * p * te * te +
                                                                                   n * p * p * p * te * te));
            if temp < 0 or temp > upper:
                a2 = 0.0
            else:
                a2 = temp

        return a2;
