import dgl
import numpy as np

from itertools import product
from estimate import adjusted_pvalue_matrix

def construct_ess(time_series: np.ndarray, alpha: float = 0.05) -> dgl.graph:
    """
    Constructs graph of time series components based on ESS given alpha confidence level.
    :param time_series: multivariate time series (n_samples, n_components)
    :param alpha: confidence level
    :returns: components graph constructed based on ESS
    """
    n_components = time_series.shape[1]
    pvalues_matrix = adjusted_pvalue_matrix(time_series, use_ess=True)
    src_nodes, dst_nodes = np.where(pvalues_matrix <= alpha)
    return dgl.graph((src_nodes, dst_nodes), num_nodes=n_components)

def construct_vanilla(time_series: np.ndarray, alpha: float = 0.05) -> dgl.graph:
    """
    Constructs graph of time series components based on default significance test given alpha confidence level.
    :param time_series: multivariate time series (n_samples, n_components)
    :param alpha: confidence level
    :returns: components graph constructed based on default significance test
    """
    n_components = time_series.shape[1]
    pvalues_matrix = adjusted_pvalue_matrix(time_series, use_ess=False)
    src_nodes, dst_nodes = np.where(pvalues_matrix <= alpha)
    return dgl.graph((src_nodes, dst_nodes), num_nodes=n_components)

def construct_complete(time_series: np.ndarray) -> dgl.graph:
    """
    Constructs complete graph of time series components.
    :param time_series: multivariate time series (n_samples, n_components)
    :returns: components complete graph
    """
    n_components = time_series.shape[1]
    all_edges = np.array(list(product(range(n_components), range(n_components))))
    src_nodes = all_edges[:, 0]
    dst_nodes = all_edges[:, 1]    
    return dgl.graph((src_nodes, dst_nodes), num_nodes=n_components)
