import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Callable


def gp_regression(x: np.array, y: np.array, x_star: np.array, sigma_n: float, kernel: Callable) -> Tuple[np.array, np.array, float]:
    """Computes the GP's posterior and log marginal likelihood
    
    Args:
        x (np.array): x values of the known data
        y (np.array): y values of the known data (assuming uncertainty ~N(0, sigma))
        x_star (np.array): desired regression values
        sigma_n (float): standard deviation of the uncertainty in y
        kernel (Callable, optional): kernel function
    
    Returns:
        Tuple[np.array, np.array, float]: mean vector and covariance matrix of the
        GP's posterior, p(f*|x,y,x*), as well as log marginal likelihood
    """
    n = x.shape[0]

    noisy_K_X_X = kernel(x, x) + sigma_n**2*np.eye(n)

    K_X_star_X = kernel(x_star, x)
    K_X_X_star = kernel(x, x_star)
    K_X_star_X_star = kernel(x_star, x_star) + 1e-6*np.eye(x_star.shape[0])

    L = np.linalg.cholesky(noisy_K_X_X)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    f_bar_star = np.matmul(K_X_star_X, alpha)

    v = np.linalg.solve(L, K_X_star_X.T)
    cov_f_star = K_X_star_X_star - np.matmul(v.T, v)

    log_marginal_likelihood = -1/2*np.matmul(y.T, alpha) - np.sum(np.log(np.diagonal(L))) - n/2*np.log(2*np.pi)

    # log marginal likelihood should be a float
    log_marginal_likelihood = log_marginal_likelihood.ravel()[0]

    return f_bar_star, cov_f_star, log_marginal_likelihood


def gp_assess_model_performance(y_star: np.array, f_bar_star: np.array, cov_f_star: np.array) -> np.array:
    """Given the groundtruth datapoints (x_star, y_star), and assuming the estimated
    mean and covariance of the GP's posterior at x_star to be given by f_bar_star and
    cov_f_star, estimates the posterior over y_star given x_star and the model:
    
        - log p(y_*|x_*,f_bar_*,cov_f_*) = 1/2log(2pi * sigma_*^2) + (y_* - f_bar_*)^2/(2sigma_*^2)
    
    Args:
        y_star (np.array): groundtruth y data, shape (n, 1)
        f_bar_star (np.array): mean vector of the GP's posterior
        cov_f_star (np.array): covariance matrix of the GP's posterior
    
    Returns:
        np.array: marginal distribution of y_* given x_* and the model, shape (n, 1)
    """
    sigma_star_squared = np.diag(cov_f_star).reshape(-1, 1)
    return 1/2*np.log(2*np.pi*sigma_star_squared) + (y_star - f_bar_star)**2/(2*sigma_star_squared)


def gp_hyperparameter_optimization(x: np.array, y: np.array, x_star: np.array, sigma_n: float, kernel: Callable, initial_params: np.array, bounds: np.array) -> np.array:
    """Perform hyperparameter optimization by maximizing the
    log_marginal_likelihood of the GP's regression.
    
    Args:
        x (np.array): x values of the known data
        y (np.array): y values of the known data (assuming uncertainty ~N(0, sigma))
        x_star (np.array): desired regression values
        sigma_n (float): standard deviation of the uncertainty in y
        kernel (Callable, optional): kernel function with h parameters
        initial_params (np.array): initial parameters of the kernel function, shape (h, 1)
        bounds (np.array): lower and upper bounds of the parameter search, shape (h, 2)
    
    Returns:
        np.array: optimal parameters based on the search, shape (h, 1)
    """
    # restart the iteration count
    global n_evaluations
    n_evaluations = 1

    # the function to be optimized is the log marginal likelihood
    def negative_log_marginal_likelihood_fn(param_vector):
        _, _, log_marginal_likelihood = gp_regression(
            x,
            y,
            x_star,
            sigma_n,
            kernel=lambda x, x_prime: kernel(x, x_prime, param_vector)
        )

        # we want to maximize log marginal likelihood
        return -log_marginal_likelihood

    def print_information(current_params):
        global n_evaluations
        print(f"Iter: {n_evaluations}, params: {current_params}, negative_log_marginal_likelihood: {negative_log_marginal_likelihood_fn(current_params)}")
        n_evaluations += 1

    result = minimize(
        negative_log_marginal_likelihood_fn,
        initial_params,
        callback=print_information,
        bounds=bounds,
        method="L-BFGS-B"
    )

    return result.x

    
def gp_point_sequential_prediction(x: np.array, y: np.array, x_star: np.array, sigma_n: float, kernel: Callable, lookahead: float=0) -> Tuple[np.array, np.array, float]:
    """Perform sequential regression for x_star (shaped (1, 1)), assuming a given
    lookahead, that is, use only data in the training set x such that
        x' + lookahead <= x_star, for all x' in x
    
    Args:
        x (np.array): x values of the known data
        y (np.array): y values of the known data (assuming uncertainty ~N(0, sigma))
        x_star (np.array): desired regression data point, shaped (1, 1) to enforce
            sequence
        sigma_n (float): standard deviation of the uncertainty in y
        kernel (Callable): kernel function
        lookahead (float, optional): separation in x between the most recent reading
            and the x at which predictions are made in the sequential setting
    
    Returns:
        Tuple[np.array, np.array, float]: mean vector and covariance matrix of the GP's
        posterior, p(f*|x,y,x*), as well as log marginal likelihood
    """
    # get the training data based on the lookahead
    x_train = x[x + lookahead <= x_star]
    y_train = y[x + lookahead <= x_star]

    # check if there's any training data
    if x_train.shape[0] == 0:
        raise ValueError("no data is available for the x_star given, cannot obtain the posterior!")

    # perform the regression
    mu, cov, log_marginal_likelihood = gp_regression(
        x_train,
        y_train,
        x_star,
        1e-1,
        kernel=kernel
    )

    return mu, cov, log_marginal_likelihood
