import dgl
import dgl.nn as dglnn

import torch
import numpy as np

from ess import ess_adjusted_pvalue_matrix


def build_graph(time_series: np.ndarray, alpha: float = 0.05) -> dgl.graph:
    """
    Constructs graph of time series components based on alpha confidence level.
    :param time_series: multivariate time series (n_samples, n_components)
    :param alpha: confidence level
    :returns: constructed graph of components
    """
    n_components = time_series.shape[1]
    pvalues_matrix = ess_adjusted_pvalue_matrix(time_series)
    src_nodes, dst_nodes = np.where(pvalues_matrix <= alpha)
    return dgl.graph((src_nodes, dst_nodes), num_nodes=n_components)


def spectral_features(graph: dgl.graph, embed_size: int) -> torch.Tensor:
    """
    Computes Laplacian eigenmaps.
    :param graph: graph to process
    :param embed_size: size of embeddings (must be <= num_nodes)
    :returns: spectral node embeddings of shape (num_nodes, embed_size)
    """
    A = graph.adj().to_dense()
    D = torch.diag(A.sum(dim=1))
    L = D - A
    eig_vecs = torch.linalg.eigh(L)[1]
    return eig_vecs[:, :embed_size]
    

def deepwalk_features() -> torch.Tensor:
    """
    Computes DeepWalk features.
    """
    pass
