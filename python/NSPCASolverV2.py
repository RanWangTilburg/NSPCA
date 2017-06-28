import numpy as np

from nspcagpu import nspca_solver_impl
from nspcagpu import init_svd_impl
from nspcagpu import NSPCASolution



def get_nspca_solution(transformed_score, principal_score, component_loading):
    check_dim_nspca_solution(component_loading, principal_score, transformed_score)
    return NSPCASolution(transformed_score=transformed_score, principal_score=principal_score,
                         component_loading=component_loading)


def init_svd(data, reduce_dim, restriction=None, scale = 2.0):
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.int32

    if restriction is not None:
        assert isinstance(restriction, np.ndarray)
        assert restriction.dtype == np.int32

    assert data.ndim == 2
    assert np.isfortran(data)

    no_obs = data.shape[0]
    no_var = data.shape[1]

    restriction_tmp = None

    if restriction is None:
        restriction_tmp = np.asfortranarray(np.full((reduce_dim, no_var), 2, dtype=np.int32))
    else:
        # assert isinstance(restriction, np.ndarray)
        restriction_tmp = np.asfortranarray(np.copy(restriction))

    return init_svd_impl(data, restriction_tmp, scale)



class NSPCASolverPy:
    def __init__(self, no_obs, no_var, reduce_dim, scale, num_streams, num_threads):
        NSPCASolverPy.sanity_check_cons(no_obs, no_var, num_streams, num_threads, reduce_dim, scale)
        self.no_obs = no_obs
        self.no_var = no_var
        self.reduce_dim = reduce_dim
        self.scale = scale
        self.nspca_solver_impl = nspca_solver_impl(no_obs, no_var, reduce_dim, scale, num_streams, num_threads)

    def init(self, data, restriction=None, init_values=None):
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert data.shape[0] == self.no_obs and data.shape[1] == self.no_var

        if restriction is not None:
            assert isinstance(restriction, np.ndarray)
            assert restriction.ndim == 2
            assert restriction.shape[0] == self.reduce_dim and restriction.shape[1] == self.no_var
        else:
            restriction = np.asfortranarray(np.full((self.reduce_dim, self.no_var), 2, dtype=np.int32, order='F'))

        if init_values is None:
            init_values = init_svd(data, self.reduce_dim, restriction)
        # print init_values
        self.nspca_solver_impl.init(data, restriction, init_values.transformed_score, init_values.principal_score,
                                    init_values.component_loading)
        # print "From python"
        # print init_values.principal_score.dot(init_values.component_loading)

    def fit(self, data, init_value, alpha, restriction=None, max_iter=100, thres=0.001):
        restriction_temp = self.sanity_check_fit(data, init_value, max_iter, restriction, thres, alpha)
        return self.nspca_solver_impl.fit(data, init_value, restriction_temp, alpha, max_iter, thres)

    def sanity_check_fit(self, data, init_value, max_iter, restriction, thres, alpha):
        assert alpha >= 0
        assert isinstance(max_iter, int)
        assert isinstance(thres, int)
        assert max_iter > 0
        assert thres > 0
        assert isinstance(data, np.ndarray)
        if restriction is not None:
            assert isinstance(restriction, np.ndarray)
        assert isinstance(init_value, NSPCASolution)
        assert np.isfortran(data)
        if restriction is not None:
            assert np.isfortran(restriction)
        assert data.shape[0] == self.no_obs
        assert data.shape[1] == self.no_var
        restriction_temp = None
        if restriction is None:
            restriction_temp = np.asfortranarray(np.zeros(self.no_var, self.reduce_dim))
        else:
            assert restriction.shape[0] == self.no_var
            assert restriction.shape[1] == self.reduce_dim
            restriction_temp = np.asfortranarray(np.copy(restriction))

        assert init_value.transformed_score.shape[0] == self.no_obs
        assert init_value.transformed_score.shape[1] == self.no_var
        assert init_value.principal_score.shape[0] == self.no_obs
        assert init_value.principal_score.shape[1] == self.reduce_dim
        assert init_value.component_loading[0] == self.reduce_dim
        assert init_value.component_loading[1] == self.no_var

        return restriction_temp

    @staticmethod
    def sanity_check_cons(no_obs, no_var, num_streams, num_threads, reduce_dim, scale):
        assert isinstance(no_obs, int)
        assert isinstance(no_var, int)
        assert isinstance(reduce_dim, int)
        assert isinstance(num_streams, int)
        assert isinstance(num_threads, int)
        assert no_obs > 0
        assert no_var > 0
        assert reduce_dim > 0
        assert no_var > reduce_dim
        assert num_streams > 0
        assert num_threads > 0
        assert scale > 0

    def get_transformed_score(self):
        return self.nspca_solver_impl.get_transformed_score()

    def get_principal_score(self):
        return self.nspca_solver_impl.get_principal_score()

    def get_component_loading(self):
        return self.nspca_solver_impl.get_component_loading()

    def transformed_score(self, numThreadsGPU, numThreadsGPUZ):
        self.nspca_solver_impl.transformed_score(numThreadsGPU, numThreadsGPUZ)

    def principal_score(self):
        self.nspca_solver_impl.principal_score()

    def component_loading(self, numThreadsGPU, numBlocksGPU):
        self.nspca_solver_impl.component_loading(numThreadsGPU, numThreadsGPU)

    def get_incidence_count(self):
        return self.nspca_solver_impl.get_incidence_count()

    def get_cumu_score(self):
        return self.nspca_solver_impl.get_cumu_score()

    def set_alpha(self, alpha):
        assert alpha > 0
        self.nspca_solver_impl.set_alpha(alpha)

def check_dim_nspca_solution(component_loading, principal_score, transformed_score):
    assert isinstance(transformed_score, np.ndarray)
    assert isinstance(principal_score, np.ndarray)
    assert isinstance(component_loading, np.ndarray)
    assert transformed_score.ndim == 2
    assert principal_score.ndim == 2
    assert component_loading.ndim == 2
    assert transformed_score.shape[0] == principal_score.shape[0]
    assert transformed_score.shape[1] == component_loading.shape[1]
    assert principal_score.shape[1] == component_loading.shape[0]
