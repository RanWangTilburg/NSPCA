from nspcagpu import NSPCASolution, init_svd_impl, normalize_transformed_score, NSPCASolver
from montecarlo import monte_carlo_tel, monte_carlo_results
import numpy as np
from numpy import *
import os
from nspcasolverimpl import assert_present
import shutil

def run_monte_carlo(out_dir, nrow, true_component_loading, restriction, upper, lower, sd_error, alpha, thres,
                    ts_thread1,
                    ts_thread2, cl_thread, cl_block, iter=1000, rep=1):
    """
    This function runs the Monte Carlo simulation, and write the results into the output directory
    The output directory must not contain a 'true' and 'solution' directory
    :param out_dir: the output directory, must not contain directory named 'true' or 'solution'
    :param nrow: the number of rows in the observations
    :param true_component_loading: the true component loading
    :param restriction: restrictions
    :param upper: the upper bound; a review will be positive only if its true score exceeds this bound
    :param lower: the lower bound; a review will be negative only if its true score falls below this bound
    :param sd_error: standard error of the residual
    :param alpha: the penalty (lambda  in the thesis)
    :param ts_thread1: number of thread in transformed score
    :param ts_thread2: number of thread in transformed score
    :param cl_thread: number of thread in component loading
    :param cl_block: number of thread in component loading
    :param rep: the total Monte Carlo sample size
    """

    assert isinstance(true_component_loading, np.ndarray)
    assert true_component_loading.ndim == 2
    assert true_component_loading.shape[0] < true_component_loading.shape[1]
    assert true_component_loading.shape[1] < nrow

    assert isinstance(restriction, np.ndarray)
    assert restriction.ndim == 2
    assert restriction.shape[0] == true_component_loading.shape[0]
    assert restriction.shape[1] == true_component_loading.shape[1]

    assert nrow > 0
    assert upper > 0
    assert lower < 0
    assert sd_error > 0
    assert rep > 0
    assert isinstance(alpha, np.ndarray)
    assert alpha.ndim == 1
    assert alpha.shape[0] == true_component_loading.shape[1]
    count = 0
    true_solution_dir = out_dir + "true_parameter"
    solution_dir = out_dir + "solution"

    rdim = true_component_loading.shape[0]
    nvar = true_component_loading.shape[1]

    shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    os.mkdir(true_solution_dir)
    os.mkdir(solution_dir)
    solver = NSPCASolver(nrow, nvar, rdim, 1.0, 2, 2, thres, iter)


    pertubation_ratio_in_component_loading_init = 0.01
    noise = pertubation_ratio_in_component_loading_init * np.asfortranarray(np.random.randn(rdim, nvar), dtype=float64)
    while count < rep:
        monte_carlo_sample = monte_carlo_tel(nrow, true_component_loading, upper=upper, lower=lower, sd_error=sd_error)

        if assert_present(monte_carlo_sample.data):
            transformed_score_init = normalize_transformed_score(monte_carlo_sample.data, 1.0)
            component_loading_after_noise = np.asfortranarray(monte_carlo_sample.component_loading + noise,
                                                              dtype=float64)

            init_solution = NSPCASolution(transformed_score_init, monte_carlo_sample.principal_score, component_loading_after_noise)
            true_solution = NSPCASolution(init_solution.transformed_score, monte_carlo_sample.principal_score,
                                          monte_carlo_sample.component_loading)
            solver.init_with_data(monte_carlo_sample.data, restriction, init_solution)
            solution = solver.solve_v2(alpha,  ts_thread1, ts_thread2, cl_thread, cl_block)

            if not solution.solution_converged:
                print "Failed to converge"

            else:
                print "Iteration %d converged " % count
                true_solution.write_directory(true_solution_dir, count)
                solution.write_directory(solution_dir, count)
                count += 1


def main():
    nrow = 1000
    nvar = 50
    rdim = 2

    true_component_loading = np.asfortranarray(np.zeros((rdim, nvar)), dtype=np.float64)
    true_component_loading[0, 0] = 1
    true_component_loading[0, 1] = 1
    true_component_loading[0, 2] = 1
    true_component_loading[0, 3] = 1

    true_component_loading[1, 8] = 1
    true_component_loading[1, 9] = 1


    restriction = np.zeros((rdim, nvar), dtype=np.int32, order='F') + 1
    alpha = np.zeros(nvar, dtype=np.float64) + 0.1

    run_monte_carlo("/home/user/Desktop/test/", nrow, true_component_loading, restriction, 0.2, -0.2, 0.1, alpha, 1.0,
                    2, 2, 2, 2)


if __name__ == "__main__":
    main()
