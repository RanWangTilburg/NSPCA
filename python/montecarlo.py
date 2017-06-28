import numpy as np
import scipy.linalg
from collections import namedtuple

monte_carlo_results = namedtuple('MonteCarloResults', 'data, error, principal_score, component_loading')


def monte_carlo_tel(no_obs, beta, upper, lower, sd_error=1.0):
    if not isinstance(beta, np.ndarray):
        raise Exception("Beta must be a numpy array")

    if beta.ndim != 2:
        raise Exception("Beta must be 2d array")
    # print no_obs
    no_vars = beta.shape[1]
    reduced_dim = beta.shape[0]

    score = np.random.randn(no_obs, reduced_dim)
    score = scipy.linalg.orth(score)
    score = np.sqrt(no_obs)*score
    score = np.asfortranarray(score, np.float64)
    error = sd_error * np.random.randn(no_obs, no_vars)
    y = score.dot(beta) + error
    # print y
    data = np.zeros((no_obs, no_vars), dtype=np.int32, order='F')
    for row in range(0, y.shape[0]):
        for col in range(0, y.shape[1]):
            if y[row, col] > upper:
                data[row, col] = 1
            elif y[row, col] < lower:
                data[row, col] = -1
    return monte_carlo_results(data=data, error=error, principal_score=score, component_loading=beta)

