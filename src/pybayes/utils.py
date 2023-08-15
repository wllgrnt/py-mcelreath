"""
utils.py

Any function we use more than once in our notebooks ends up here.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


def grid_approximate_binomial(n: int, k: int, grid_size: int, prior=None, plot=True, marker=None) -> np.ndarray:
    """Generate the posterior distribution over the range of possible p values for
    a binomial distribution, given observed n and k, and optional non-uniform prior.
    
        Returns: a numpy array of size (grid_size, 2) with columns (p, posterior).
    """


    p_grid = np.linspace(0,1, grid_size)
    # if prior is None, assume a uniform distribution over the grid.
    if prior is None:
        prior = np.ones(grid_size)
    # evaluate the probability of our observed data given our model.
    # binomial(n, p, k): (n choose k) * p^k * (1-p)^(n-k)
    likelihood = scipy.stats.binom.pmf(n=n, k=k, p=p_grid)   
    posterior_unscaled  = likelihood * prior 
    posterior = posterior_unscaled / posterior_unscaled.sum()
    
    if plot:
        plot_nicely(x_vals=p_grid, y_vals=posterior, marker=marker, ylabel='posterior', xlabel='p')
        
    return np.column_stack((p_grid, posterior))


def plot_nicely(x_vals, y_vals, truncate=True, marker=None, ylabel=None, xlabel=None):
    """Plot the graph in an opinionated way."""
    _, ax = plt.subplots()
    if marker is not None:
        sns.lineplot(x=x_vals, y=y_vals, ax=ax, marker='o')
    else:
        sns.lineplot(x=x_vals, y=y_vals, ax=ax)
    if truncate:
        ax.set_xlim(x_vals.min(), x_vals.max())
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.show()

def hist(samples, ax=None):
    """Plot a histogram of samples."""
    if ax is None:
        _, ax = plt.subplots() 
    sns.histplot(samples, ax=ax, kde=False)
    ax.set_ylabel('frequency')
    plt.show()
    return ax

def load_dataset(data_identifier: str) -> pd.DataFrame:
    """Dataset should be one of ('howell', 'waffle_divorce')"""
    match data_identifier:
        case 'howell':
            url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv"
        case 'waffle_divorce':
            url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
        case _:
            raise ValueError(f"Unknown dataset: {data_identifier}")
    return pd.read_csv(url, sep=';')
