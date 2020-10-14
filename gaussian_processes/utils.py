import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dateutil.parser as dp
from typing import Optional

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12
})


def read_and_process_2D_data(filename: str, y_axis: str="Tide height (m)", true_y_axis: str="True tide height (m)") -> pd.DataFrame:
    """Read CSV file and load the data into a dictionary
    
    Args:
        filename (str): path to the file to read
        y_axis (str, optional): y axis data
    
    Returns:
        pd.DataFrame: dataframe containing the time and y axis information
    """
    data = pd.read_csv(filename)

    initial_time = dp.parse(data["Reading Date and Time (ISO)"][0])

    missing_data = pd.DataFrame(
        [
            [(dp.parse(time) - initial_time).total_seconds()/(3600*24), y, y_true]
            for time, y, y_true in zip(data["Reading Date and Time (ISO)"], data[y_axis], data[true_y_axis])
            if np.isnan(y)
        ],
        columns=["t", "y", "y_true"]
    )

    known_data = pd.DataFrame(
        [
            [(dp.parse(time) - initial_time).total_seconds()/(3600*24), y, y_true]
            for time, y, y_true in zip(data["Reading Date and Time (ISO)"], data[y_axis], data[true_y_axis])
            if not np.isnan(y)
        ],
        columns=["t", "y", "y_true"]
    )

    return known_data, missing_data


def plot_gp(f_bar_star: np.array, cov_f_star: np.array, x_star: np.array, x: np.array, y: np.array, y_mean: float, y_label: str="Tide Height (m)", y_star: Optional[np.array]=None) -> None:
    """Plot the GP's posterior, along with the known data points
    
    Args:
        f_bar_star (np.array): mean vector of the GP's posterior
        cov_f_star (np.array): covariance matrix of the GP's posterior
        x_star (np.array): input regression points
        x (np.array): input given data
        y (np.array): output given data
        y_mean (float): y mean of the known data
        y_label (str, optional): label of the y axis
        y_star (Optional[np.array], optional): GT y values
    """
    if np.linalg.cond(cov_f_star) > 10**10:
        print(f"Covariance matrix is ill-conditioned! Condition number: {np.linalg.cond(cov)}")

    x_star = x_star.ravel()
    f_bar_star = f_bar_star.ravel()
    sigma_star = np.sqrt(np.diag(cov_f_star))

    # function draws
    function_draws = np.random.multivariate_normal(f_bar_star, cov_f_star, 3)
    for i, draw_points in enumerate(function_draws):
        plt.plot(x_star, y_mean + draw_points, lw=1, ls='--', label=f'Draw {i+1}')

    plt.fill_between(x_star, y_mean + f_bar_star + 2*sigma_star, y_mean + f_bar_star - 2*sigma_star, alpha=0.1, label="$\pm2\sigma$")
    plt.fill_between(x_star, y_mean + f_bar_star + sigma_star, y_mean + f_bar_star - sigma_star, alpha=0.2, label="$\pm\sigma$")
    plt.plot(x_star, y_mean + f_bar_star, label="Mean")
    
    if y_star is not None:
        plt.plot(x_star, y_mean + y_star, "rx", markersize=2, label="GT Data")

    plt.plot(x, y_mean + y, "kx", markersize=3, label="Input Data")

    plt.legend()
    plt.xlabel("$\Delta t$ (days)")
    plt.ylabel(y_label)

    plt.show()
