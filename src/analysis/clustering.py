"""Group zero-shot models into families based on pairwise prediction agreement."""

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def assign_clusters(
    kappa_matrix: pd.DataFrame,
    n_clusters: int = 5,
    method: str = "ward",
) -> pd.Series:
    """Cluster models by Ward hierarchical clustering on 1 - kappa distance.

    Parameters
    ----------
    kappa_matrix:
        Square, symmetric DataFrame of pairwise Cohen's kappa (index/columns
        are model names). NaN entries (pairs with too little overlap) are
        treated as maximally distant (kappa=0).
    n_clusters:
        Number of flat clusters to cut the dendrogram into.
    method:
        Linkage method passed to scipy.cluster.hierarchy.linkage.

    Returns
    -------
    Series indexed by model name, values "cluster_0".."cluster_{n_clusters-1}".
    """
    models = kappa_matrix.index.tolist()
    kappa_filled = kappa_matrix.fillna(0.0)
    distance = 1.0 - kappa_filled
    np.fill_diagonal(distance.values, 0.0)
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    cluster_ids = pd.Series(
        [f"cluster_{label - 1}" for label in labels],
        index=models,
        name="cluster_id",
    )
    return cluster_ids


def label_clusters_manually(
    cluster_series: pd.Series,
    mapping: Mapping[str, str],
) -> pd.DataFrame:
    """Attach human-readable family names to auto-generated cluster ids.

    Parameters
    ----------
    cluster_series:
        Output of assign_clusters (index=model, values="cluster_N").
    mapping:
        Dict mapping cluster ids (e.g. "cluster_0") to a readable label
        (e.g. "structure_aware"). Every cluster id present in
        cluster_series must have an entry.

    Returns
    -------
    DataFrame indexed by model with columns ["cluster_id", "cluster_label"].
    """
    missing = set(cluster_series.unique()) - set(mapping)
    if missing:
        raise ValueError(f"mapping is missing entries for: {sorted(missing)}")
    out = cluster_series.to_frame("cluster_id")
    out["cluster_label"] = out["cluster_id"].map(mapping)
    out.index.name = "model"
    return out
