"""Accuracy of individual models and model-family clusters against wet-lab ground truth."""

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

_METRIC_COLS = ["accuracy", "precision", "recall", "f1", "mcc", "n"]


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if y_true.nunique() > 1 and y_pred.nunique() > 1 else 0.0,
        "n": len(y_true),
    }


def per_model_accuracy(
    df: pd.DataFrame,
    models: Sequence[str],
    label_col: str = "DMS_score_bin",
) -> pd.DataFrame:
    """Accuracy/precision/recall/F1/MCC of each model's binary calls vs. ground truth.

    Rows with a missing label are dropped once, up front. For each model,
    rows where that model's {model}_binary score is missing are additionally
    dropped (per-model, since missingness differs across models).

    Returns
    -------
    DataFrame indexed by model with columns [accuracy, precision, recall, f1, mcc, n].
    """
    labeled = df[df[label_col].notna()]
    rows = {}
    for model in models:
        col = f"{model}_binary"
        sub = labeled[labeled[col].notna()]
        if sub.empty:
            rows[model] = {k: np.nan for k in _METRIC_COLS}
            continue
        rows[model] = _binary_metrics(sub[label_col], sub[col])
    return pd.DataFrame.from_dict(rows, orient="index")[_METRIC_COLS]


def per_model_accuracy_by_assay(
    df: pd.DataFrame,
    models: Sequence[str],
    label_col: str = "DMS_score_bin",
) -> pd.DataFrame:
    """Same as per_model_accuracy, grouped by dms_id.

    Returns
    -------
    Long-format DataFrame with columns [dms_id, model, accuracy, precision, recall, f1, mcc, n].
    """
    parts = []
    for dms_id, group in df.groupby("dms_id"):
        acc = per_model_accuracy(group, models, label_col=label_col)
        acc.insert(0, "dms_id", dms_id)
        acc.index.name = "model"
        parts.append(acc.reset_index())
    return pd.concat(parts, ignore_index=True)


def cluster_majority_vote(
    df_binary: pd.DataFrame,
    cluster_assignments: pd.Series,
) -> pd.DataFrame:
    """Majority binarized call within each model cluster, per variant.

    cluster_assignments: Series indexed by model name, values = cluster label.
    Majority = 1 if the mean of member models' {model}_binary calls > 0.5,
    0 if < 0.5, and NA on an exact tie (or if all members are missing).

    Returns
    -------
    DataFrame aligned to df_binary.index, one column per cluster label.
    """
    votes = {}
    for cluster_label in sorted(cluster_assignments.unique()):
        members = cluster_assignments[cluster_assignments == cluster_label].index.tolist()
        cols = [f"{m}_binary" for m in members if f"{m}_binary" in df_binary.columns]
        mean_vote = df_binary[cols].astype(float).mean(axis=1)
        majority = pd.Series(pd.NA, index=df_binary.index, dtype="Int8")
        majority[mean_vote > 0.5] = 1
        majority[mean_vote < 0.5] = 0
        votes[cluster_label] = majority
    return pd.DataFrame(votes, index=df_binary.index)


def per_cluster_accuracy(
    df: pd.DataFrame,
    cluster_votes: pd.DataFrame,
    label_col: str = "DMS_score_bin",
) -> pd.DataFrame:
    """Accuracy of each cluster's majority vote vs. ground truth, overall and per assay.

    df must share cluster_votes' index and contain `label_col` and `dms_id`.

    Returns
    -------
    Long-format DataFrame with columns [cluster, dms_id, accuracy, precision, recall, f1, mcc, n].
    dms_id == "ALL" holds the pooled-across-assays row for each cluster.
    """
    clusters = cluster_votes.columns.tolist()
    rows = []

    def _score(sub_df, sub_votes, dms_id):
        for cluster in clusters:
            mask = sub_df[label_col].notna() & sub_votes[cluster].notna()
            if mask.sum() == 0:
                rows.append({"cluster": cluster, "dms_id": dms_id, **{k: np.nan for k in _METRIC_COLS}})
                continue
            metrics = _binary_metrics(sub_df.loc[mask, label_col], sub_votes.loc[mask, cluster])
            rows.append({"cluster": cluster, "dms_id": dms_id, **metrics})

    _score(df, cluster_votes, "ALL")
    for dms_id, group in df.groupby("dms_id"):
        _score(group, cluster_votes.loc[group.index], dms_id)

    return pd.DataFrame(rows)[["cluster", "dms_id"] + _METRIC_COLS]


def cluster_pairwise_showdown(
    df: pd.DataFrame,
    cluster_votes: pd.DataFrame,
    label_col: str = "DMS_score_bin",
) -> pd.DataFrame:
    """For each cluster pair, who's right more often on variants where they disagree.

    Only considers variants where both clusters have a majority vote and it
    differs between them, and the ground-truth label is present. Since votes
    and label are binary, exactly one of the two clusters matches the label
    on every such variant, so a_win_pct + b_win_pct == 100.

    Returns
    -------
    DataFrame with columns [cluster_a, cluster_b, n_disagree, a_win_pct, b_win_pct].
    """
    clusters = cluster_votes.columns.tolist()
    label = df[label_col]
    rows = []
    for a, b in combinations(clusters, 2):
        mask = (
            cluster_votes[a].notna()
            & cluster_votes[b].notna()
            & label.notna()
            & (cluster_votes[a] != cluster_votes[b])
        )
        n = int(mask.sum())
        if n == 0:
            rows.append({"cluster_a": a, "cluster_b": b, "n_disagree": 0, "a_win_pct": np.nan, "b_win_pct": np.nan})
            continue
        a_correct = (cluster_votes.loc[mask, a].astype(int) == label.loc[mask].astype(int)).sum()
        a_win_pct = 100.0 * a_correct / n
        rows.append(
            {
                "cluster_a": a,
                "cluster_b": b,
                "n_disagree": n,
                "a_win_pct": a_win_pct,
                "b_win_pct": 100.0 - a_win_pct,
            }
        )
    return pd.DataFrame(rows)


def cluster_showdown_by_assay(
    df: pd.DataFrame,
    cluster_votes: pd.DataFrame,
    label_col: str = "DMS_score_bin",
) -> pd.DataFrame:
    """cluster_pairwise_showdown, stratified by dms_id.

    Returns
    -------
    Long-format DataFrame with columns
    [dms_id, cluster_a, cluster_b, n_disagree, a_win_pct, b_win_pct].
    """
    parts = []
    for dms_id, group in df.groupby("dms_id"):
        showdown = cluster_pairwise_showdown(group, cluster_votes.loc[group.index], label_col=label_col)
        showdown.insert(0, "dms_id", dms_id)
        parts.append(showdown)
    return pd.concat(parts, ignore_index=True)
