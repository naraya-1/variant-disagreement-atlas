"""Visualisation helpers for variant disagreement analysis."""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.analysis.disagreement import variant_disagreement_score


def plot_score_heatmap(
    df: pd.DataFrame,
    dms_id: str,
    models: Optional[Sequence[str]] = None,
    sort_by_disagreement: bool = True,
    n_variants: int = 50,
    figsize: tuple = (12, 8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Variant × model heatmap of z-scored continuous scores for one assay.

    Parameters
    ----------
    df:
        DataFrame from load_dms_scores (must include model columns).
    dms_id:
        Which DMS assay to plot.
    models:
        List of model columns to include. None = all columns detected from df.
    sort_by_disagreement:
        Sort variants by entropy (most disagreed-upon at top).
    n_variants:
        Number of variants to show (top-N by disagreement when sorting).
    figsize:
        Figure size passed to matplotlib.
    ax:
        Existing Axes to draw into. If None, a new Figure is created.
    """
    from src.data.proteingym_loader import get_model_columns

    sub = df[df["dms_id"] == dms_id].copy()
    if sub.empty:
        raise ValueError(f"No rows for dms_id={dms_id!r}")

    if models is None:
        models = get_model_columns(df)

    # Drop model columns with all-NaN for this assay
    valid_models = [m for m in models if sub[m].notna().any()]

    # Z-score each model within the assay
    score_mat = sub[valid_models].copy().astype(float)
    score_mat = (score_mat - score_mat.mean()) / score_mat.std().clip(lower=1e-10)

    if sort_by_disagreement:
        from src.analysis.disagreement import binarize_scores
        sub_b = binarize_scores(sub, valid_models)
        entropy = variant_disagreement_score(sub_b, valid_models)
        order = entropy.sort_values(ascending=False).index
        score_mat = score_mat.loc[order]
        sub = sub.loc[order]

    score_mat = score_mat.iloc[:n_variants]
    sub = sub.iloc[:n_variants]

    row_labels = sub["mutant"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        score_mat,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        yticklabels=row_labels,
        xticklabels=True,
        linewidths=0,
        cbar_kws={"label": "z-score", "shrink": 0.6},
    )
    ax.set_title(f"{dms_id}  —  top {len(score_mat)} variants by disagreement", fontsize=11)
    ax.set_xlabel("Model")
    ax.set_ylabel("Variant")
    ax.tick_params(axis="x", labelsize=5, rotation=90)
    ax.tick_params(axis="y", labelsize=6)

    return fig


def plot_pairwise_agreement(
    kappa_matrix: pd.DataFrame,
    figsize: tuple = (10, 10),
    title: str = "Pairwise model agreement (Cohen's κ)",
) -> sns.matrix.ClusterGrid:
    """Model × model agreement heatmap with hierarchical clustering.

    Models that agree with each other will cluster together (ESM family,
    EVE family, Tranception family, etc.).

    Parameters
    ----------
    kappa_matrix:
        Square DataFrame of kappa values (from pairwise_agreement()).
    figsize:
        Figure size for the clustermap.
    title:
        Title string placed above the clustermap.

    Returns
    -------
    seaborn ClusterGrid object.
    """
    mat = kappa_matrix.copy().astype(float)

    # Fill NaN with 0 for the distance matrix (unknown → neutral assumption)
    mat_filled = mat.fillna(0.0)

    # Convert kappa to a distance: disagreeing models are far apart
    dist = 1.0 - mat_filled
    dist_arr = dist.to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0.0)
    dist = pd.DataFrame(dist_arr, index=dist.index, columns=dist.columns)
    dist = dist.clip(lower=0.0)

    # Condense to 1-D for scipy linkage
    condensed = squareform(dist.values, checks=False)
    condensed = np.clip(condensed, 0.0, None)

    g = sns.clustermap(
        mat,
        row_linkage=linkage(condensed, method="ward"),
        col_linkage=linkage(condensed, method="ward"),
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=1.0,
        center=0.4,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        linewidths=0,
        cbar_kws={"label": "Cohen's κ", "shrink": 0.4},
    )
    g.ax_heatmap.tick_params(axis="x", labelsize=5, rotation=90)
    g.ax_heatmap.tick_params(axis="y", labelsize=5)
    g.fig.suptitle(title, y=1.01, fontsize=13)

    return g
