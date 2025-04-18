import numpy as np
import scipy as sp


def compute_ess(x_series: np.ndarray, y_series: np.ndarray) -> float:
    """
    Calculates effective sample size for two time series (https://arxiv.org/pdf/2401.02387).
    :param x_series: X time series data
    :param y_series: Y time series data
    :returns: effective sample size
    """
    n = x_series.shape[0]
    x_dot = np.diff(x_series)
    y_dot = np.diff(y_series)
    ess = n * np.sqrt((np.var(x_dot) + np.var(y_dot)) / (2.0 * np.pi))
    return ess


def correlation_ess_estimation(x_series: np.ndarray, y_series: np.ndarray) -> float:
    """
    Estimates correlation between time series X and Y based on effective sample size (https://arxiv.org/pdf/2401.02387).
    :param x_series: X time series data
    :param y_series: Y time series data
    :returns: test statistic
    """
    n = x_series.shape[0]
    r, _ = sp.stats.pearsonr(x_series, y_series)
    ess = min(compute_ess(x_series, y_series), n)
    return np.sqrt(max(ess - 3, 0)) * np.atanh(r)


def ess_adjusted_pvalue_matrix(time_series: np.ndarray) -> np.ndarray:
    """
    Computes a matrix of ESS-adjusted p-values for pairwise correlations between time series components.
    The p-values test the null hypothesis that the true correlation is zero, accounting for autocorrelation.
    :param time_series: multivariate time series (n_samples, n_components)
    :returns: symmetric matrix of p-values of shape (n_components, n_components)
    """
    n_components = time_series.shape[1]
    pvalue_matrix = np.zeros((n_components, n_components))
    for i in range(n_components):
        for j in range(i + 1, n_components):
            test_stat = correlation_ess_estimation(time_series[:, i], time_series[:, j])
            p_value = 2 * sp.stats.norm.cdf(-np.abs(test_stat))
            pvalue_matrix[i, j] = p_value
            pvalue_matrix[j, i] = p_value

    return pvalue_matrix
