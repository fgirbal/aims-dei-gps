import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dateutil.parser as dp
from typing import Optional

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14
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


def plot_gp(f_bar_star: np.array, cov_f_star: np.array, x_star: np.array, x: np.array, y: np.array, y_mean: float, y_label: str="Tide Height (m)", title: Optional[str]=None, y_gt: Optional[np.array]=None, x_gt: Optional[np.array]=None, draw_samples: bool=True) -> None:
    """Plot the GP's posterior, along with the known data points
    
    Args:
        f_bar_star (np.array): mean vector of the GP's posterior
        cov_f_star (np.array): covariance matrix of the GP's posterior
        x_star (np.array): input regression points
        x (np.array): input given data
        y (np.array): output given data
        y_mean (float): y mean of the known data
        y_label (str, optional): label of the y axis
        title (Optional[str], optional): plot title
        y_gt (Optional[np.array], optional): GT y values
        x_gt (Optional[np.array], optional): GT x values, if different from x_star
        draw_samples (bool, optional): if True, draw function samples and plot them
    """
    if np.linalg.cond(cov_f_star) > 10**10:
        print(f"Covariance matrix is ill-conditioned! Condition number: {np.linalg.cond(cov)}")

    x_star = x_star.ravel()
    f_bar_star = f_bar_star.ravel()
    sigma_star = np.sqrt(np.diag(cov_f_star))

    # function draws
    if draw_samples:
        function_draws = np.random.multivariate_normal(f_bar_star, cov_f_star, 3)
        for i, draw_points in enumerate(function_draws):
            plt.plot(
                x_star,
                y_mean + draw_points,
                lw=1,
                ls='--',
                label=f'Draw {i+1}',
                color=f'C{4+i}'
            )

    plt.fill_between(
        x_star,
        y_mean + f_bar_star + 2*sigma_star,
        y_mean + f_bar_star - 2*sigma_star,
        alpha=0.15,
        label="$\pm2\sigma$",
        color='C0',
        linewidth=0
    )
    plt.fill_between(
        x_star,
        y_mean + f_bar_star + sigma_star,
        y_mean + f_bar_star - sigma_star,
        alpha=0.3,
        label="$\pm\sigma$",
        color='C0',
        linewidth=0
    )
    plt.plot(x_star, y_mean + f_bar_star, label="Mean", color='C2')
    
    if y_gt is not None:
        if x_gt is None:
            x_gt = x_star

        plt.plot(x_gt, y_mean + y_gt, "r.", markersize=3, label="Test Data")

    plt.plot(x, y_mean + y, "k*", markersize=4, label="Train Data")

    plt.legend(prop={'size': 8})
    plt.xlabel("$\Delta t$ (days)")
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()
