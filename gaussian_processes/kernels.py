import numpy as np

def rbf_kernel(x: np.array, y: np.array, sigma: float, l: float):
    """Returns K(x, x') for K being the RBF kernel
    
    Args:
        x (np.array): first data term, with n x m entries
        y (np.array): second data term, with n x m entries
        sigma (float): maximum allowed covariance
        l (float): lengthscale
    """
    # calculate the squared distances
    x_vector = np.array(x).reshape(-1, 1)
    y_vector = np.array(y).reshape(-1, 1)
    quadratic_dist = np.sum(x_vector**2, 1).reshape(-1, 1) + np.sum(y_vector**2, 1) - 2 * np.dot(x_vector, y_vector.T)

    # compute the kernel using numpy
    return sigma**2*np.exp(-(1/2*l**2)*quadratic_dist)


def periodic_kernel(x: np.array, y: np.array, sigma: float, l: float, p: float):
    """Returns K(x, x') for K being the periodic kernel
    
    Args:
        x (np.array): first data term, with n x m entries
        y (np.array): second data term, with n x m entries
        sigma (float): maximum allowed covariance
        l (float): lengthscale
        p (float): period
    """
    # calculate the squared distances and take the square root
    x_vector = np.array(x).reshape(-1, 1)
    y_vector = np.array(y).reshape(-1, 1)
    dist = np.sqrt(np.sum(x_vector**2, 1).reshape(-1, 1) + np.sum(y_vector**2, 1) - 2 * np.dot(x_vector, y_vector.T))

    # compute the kernel using numpy
    return sigma**2*np.exp(-(2/l**2)*np.sin(np.pi*dist/p)**2)


def rational_quadratic_kernel(x: np.array, y: np.array, sigma: float, alpha: float, l: float):
    """Returns K(x, x') for K being the rational quadratic kernel
    
    Args:
        x (np.array): first data term, with n x m entries
        y (np.array): second data term, with n x m entries
        sigma (float): maximum allowed covariance
        alpha (float): relative weighting of large-scale and small-scale variations
        l (float): lengthscale
    """
    # calculate the squared distances
    x_vector = np.array(x).reshape(-1, 1)
    y_vector = np.array(y).reshape(-1, 1)
    quadratic_dist = np.sum(x_vector**2, 1).reshape(-1, 1) + np.sum(y_vector**2, 1) - 2 * np.dot(x_vector, y_vector.T)

    # compute the kernel using numpy
    return sigma**2*(1 + quadratic_dist/(2*alpha*l))**(-alpha)


def parameter_periodic_rational_quadratic_kernel(x: np.array, y: np.array, parameters: np.array):
    """Kernel for a periodic rational quadratic kernel
    
    Args:
        x (np.array): Description
        y (np.array): Description
        parameters (np.array): (6,) shaped numpy array, with parameters (in order)
            [p_sigma, p_l, p_p, rq_sigma, rq_alpha, rq_l]
    """
    p_sigma, p_l, p_p = parameters[:3]
    rq_sigma, rq_alpha, rq_l = parameters[3:]
    return periodic_kernel(
        x,
        y,
        sigma=p_sigma,
        l=p_l,
        p=p_p
    ) * rational_quadratic_kernel(
        x,
        y,
        sigma=rq_sigma,
        alpha=rq_alpha,
        l=rq_l
    )
