import numpy as np
cimport numpy as np
cimport cython
import cython
from recordtype import recordtype
from libcpp cimport bool


class NSPCASolution:
    def __init__(self, np.ndarray[double, mode='fortran', ndim=2] transformed_score,
                 np.ndarray[double, mode='fortran', ndim=2] principal_score,
                 np.ndarray[double, mode='fortran', ndim=2] component_loading):
        assert (transformed_score.shape[0] == principal_score.shape[0])
        assert (principal_score.shape[1] == component_loading.shape[0])
        assert (transformed_score.shape[1] == component_loading.shape[1])
        self.transformed_score = transformed_score
        self.principal_score = principal_score
        self.component_loading = component_loading
        self.solution_converged = False

    def write_directory(self, path, id):
        np.savetxt(path + '/transformed_score' + str(id) + '.csv', self.transformed_score, delimiter=',')
        np.savetxt(path + '/principal_score' + str(id) + '.csv', self.principal_score, delimiter=',')
        np.savetxt(path + '/component_loading' + str(id) + '.csv', self.component_loading, delimiter=',')

    def set_converge(self):
        self.solution_converged = True
    def set_unconverge(self):
        self.solution_converged = False

cdef extern from "../cxxsrc/Solver.h" namespace "NSPCA":
    cdef cppclass Solver:
        Solver(int no_obs, size_t no_var, size_t reduced_dim, double scale, unsigned int numThreads,
               int numStreams) except+
        void init(const int *data, const int *restriction, double *transformed_score, double *principal_socre,
                  double *component_loading) except+
        bool solve_v2(double *transformed_score, double *principal_score, double *component_loading, double*alpha,
                      double threshold, int max_iter, unsigned int ts_thread1, unsigned int ts_thread2,
                      unsigned int cl_thread,
                      unsigned int cl_block) except+

cdef class NSPCASolver:
    cdef Solver*solver
    cdef size_t no_obs
    cdef size_t no_var
    cdef size_t no_var_after_reduce

    cdef double thres
    cdef int max_iter
    cdef bool initialized_with_data

    def __init__(self, int no_obs, size_t no_var, size_t no_var_after_reduce, double scale,
                 unsigned int no_threads_in_pool,
                 int no_streams_in_pool, thres =0.1, max_iter = 1000):
        self.sanity_check_cons(max_iter, no_obs, no_streams_in_pool, no_threads_in_pool, no_var, no_var_after_reduce,
                               scale,
                               thres)
        try:
            self.solver = new Solver(no_obs, no_var, no_var_after_reduce, scale, no_threads_in_pool, no_streams_in_pool)

            self.no_obs = no_obs
            self.no_var = no_var
            self.no_var_after_reduce = no_var_after_reduce
            self.thres = thres
            self.max_iter = max_iter

            self.initialized_with_data = False

        except:
            raise Exception(
                "Initialization Failed. Please check you have installed appropriate software support and restart python interpreter")



    def __del__(self):
        del self.solver

    def init_with_data(self, np.ndarray[int, mode = 'fortran', ndim=2] data,
                       np.ndarray[int, mode='fortran', ndim=2] restriction,
                       init_values):
        assert isinstance(init_values, NSPCASolution)
        self.init_impl(data, restriction, init_values.transformed_score, init_values.principal_score,
                       init_values.component_loading)
        self.initialized_with_data = True


    def solve_v2(self, np.ndarray[double, ndim=1] alpha, int ts_thread1, int ts_thread2, int cl_thread,
                 int cl_block):

        self.sanity_check_solver(alpha, cl_block, cl_thread, ts_thread1, ts_thread2)

        cdef np.ndarray[double, mode='fortran', ndim=2] ts_sol = np.zeros((self.no_obs, self.no_var),
                                                                          order='F', dtype=np.float64)
        cdef np.ndarray[double, mode='fortran', ndim=2] ps_sol = np.zeros((self.no_obs, self.no_var_after_reduce),
                                                                          order='F', dtype=np.float64)
        cdef np.ndarray[double, mode='fortran', ndim=2] cl_sol = np.zeros(
            (self.no_var_after_reduce, self.no_var), order='F', dtype=np.float64)

        converged = self.solver.solve_v2(&ts_sol[0, 0], &ps_sol[0, 0],
                                         &cl_sol[0, 0], &alpha[0],
                                         self.thres, self.max_iter, ts_thread1, ts_thread2, cl_thread, cl_block)
        result = NSPCASolution(ts_sol, ps_sol, cl_sol)

        if converged:
            result.set_converge()
        else:
            result.set_unconverge()

        return result


    ############################################################################
    ##########################Helper############################################
    ############################################################################

    def sanity_check_cons(self, max_iter, no_obs, no_streams_in_pool, no_threads_in_pool, no_var, no_var_after_reduce,
                          scale,
                          thres):
        assert no_obs > 1
        assert no_var > 1
        assert no_var_after_reduce > 1
        assert no_var_after_reduce < no_var
        assert scale > 0
        assert no_threads_in_pool > 0
        assert no_streams_in_pool > 0
        assert thres > 0
        assert max_iter > 0

    def init_impl(self, np.ndarray[int, mode = 'fortran', ndim=2] data,
                  np.ndarray[int, mode='fortran', ndim=2] restriction,
                  np.ndarray[double, mode='fortran', ndim=2] transformed_score,
                  np.ndarray[double, mode='fortran', ndim=2] principal_score,
                  np.ndarray[double, mode='fortran', ndim=2] component_loading):
        self.solver.init(&data[0, 0], &restriction[0, 0], &transformed_score[0, 0], &principal_score[0, 0],
                         &component_loading[0, 0])

    def sanity_check_solver(self, alpha, cl_block, cl_thread, ts_thread1, ts_thread2):
        assert ts_thread1 > 0
        assert ts_thread2 > 0
        assert cl_thread > 0
        assert cl_block > 0
        assert alpha.dtype == np.float64
        assert alpha.ndim == 1
        assert alpha.shape[0] == self.no_var
    ############################################################################
    ##########################Deprecated########################################
    ############################################################################

    @staticmethod
    def init_svd(np.ndarray[int, mode = 'fortran', ndim=2] data,
                 np.ndarray[int, mode = 'fortran', ndim =2] restriction,
                 np.ndarray[double, mode = 'fortran', ndim=2] transformed_score,
                 np.ndarray[double, mode ='fortran', ndim=2] principal_score,
                 np.ndarray[double, mode = 'fortran', ndim=2] component_loading,
                 double scale):
        init_svd_impl(data, restriction, transformed_score, principal_score, component_loading)
######################################################################
########################Helper Routines###############################
######################################################################
def construct_zeros(nrow, ncol, type = np.float64):
    """
    Construct a fortran contiguous array
    :param nrow: number of rows
    :param ncol: number of cols
    :param type: type of the array
    :return: a fortran contiguous array with zeros
    """
    result = np.asfortranarray(np.zeros(nrow, ncol), type)
    return result

@cython.boundscheck(False)
def is_similar(a, b, thres = 0.1):
    if a.ndim != b.ndim:
        print "Warning: Shape is not the same"
        return False
    for i in range(a.ndim):
        if a.shape[i] != b.shape[i]:
            print "Warning: Shape is not the same"
            return False
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if abs(a[i, j] - b[i, j]) > thres:
                return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
def init_svd_impl(np.ndarray[int, mode = 'fortran', ndim=2] data,
                  np.ndarray[int, mode = 'fortran', ndim =2] restriction,
                  double scale):
    cdef int no_obs = data.shape[0]
    cdef int no_var = data.shape[1]
    cdef int reduce_dim = restriction.shape[0]
    transformed_score = np.asfortranarray(np.zeros((no_obs, no_var), dtype=np.float64))
    principal_score = np.asfortranarray(np.zeros((no_obs, reduce_dim), dtype=np.float64))
    component_loading = np.asfortranarray(np.zeros((reduce_dim, no_var), dtype=np.float64))
    transformed_score = normalize_transformed_score(data, scale)
    # cdef np.ndarray[double, mode='fortran',ndim=2] data_tmp = data.copy()

    [u, diag, v] = np.linalg.svd(data, full_matrices=False)
    cdef int column_number = reduce_dim
    principal_score = np.asfortranarray(u[:, 0:(reduce_dim)])
    # print principal_score
    component_loading = np.asfortranarray(np.diag(diag)[0:column_number, 0:column_number].dot(v[0:column_number, :]))
    # print component_loading
    check_signs(component_loading, restriction)
    return NSPCASolution(transformed_score=transformed_score, principal_score=principal_score,
                         component_loading=component_loading)

@cython.boundscheck(False)
@cython.wraparound(False)
def normalize_transformed_score(np.ndarray[int, mode = 'fortran', ndim=2] data,
                                double scale):
    cdef np.ndarray[double, mode ='fortran', ndim=2] init_transformed_score = np.asfortranarray(
        np.copy(data.astype(np.float64)))
    for i in range(data.shape[1]):
        init_transformed_score[:, i] = (init_transformed_score[:, i] - np.mean(init_transformed_score[:, i])) * (
            scale / np.sqrt(np.var(init_transformed_score[:, i])))

    return init_transformed_score

@cython.boundscheck(False)
@cython.wraparound(False)
def check_signs(np.ndarray[double, mode = 'fortran', ndim=2] init_component_loading,
                np.ndarray[int, mode = 'fortran', ndim=2] restriction):
    cdef int nrow = restriction.shape[0]
    cdef int ncol = restriction.shape[1]

    for row in range(nrow):
        for col in range(ncol):
            if restriction[row, col] == -1 and init_component_loading[row, col] > 0:
                init_component_loading[row, col] = - init_component_loading[row, col]
            elif restriction[row, col] == 1 and init_component_loading[row, col] < 0:
                init_component_loading[row, col] = -init_component_loading[row, col]
            elif restriction[row, col] == 0 and init_component_loading[row, col] != 0:
                init_component_loading[row, col] = 0
