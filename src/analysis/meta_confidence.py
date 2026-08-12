"""Does structural context explain model disagreement, and when is the
model consensus reliable? First-cut meta-confidence signal built on top of
Week 1 (disagreement) and Week 2 (cluster accuracy) results."""

from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.accuracy import binary_metrics

_STRUCTURAL_COLS = ["plddt", "sasa", "rsa", "secondary_structure", "contact_density"]

# RSA < 0.25 = buried, >= 0.25 = exposed (Rost & Sander-style threshold,
# consistent with Tien et al. 2013's recommended cutoff for a 2-class split).
_RSA_BURIED_CUTOFF = 0.25


def attach_structural_features(
    df: pd.DataFrame,
    structural_features: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join per-residue structural features onto a per-variant DataFrame.

    Parameters
    ----------
    df:
        Per-variant DataFrame with `protein_id` and `position` columns
        (position is NaN for multi-mutants, which get NaN structural
        features -- a single-mutant structural feature cannot be assigned
        to a multi-site variant).
    structural_features:
        Dict mapping protein_id -> DataFrame indexed by `position` with
        columns [plddt, sasa, rsa, secondary_structure, contact_density]
        (the output of src.data.structural_features.compute_structural_features).
        Proteins missing from this dict (e.g. no AlphaFold model available)
        get NaN structural features for every row.

    Returns
    -------
    Copy of df with the structural feature columns appended, index preserved.
    """
    parts = []
    for protein_id, group in df.groupby("protein_id", dropna=False):
        feats = structural_features.get(protein_id)
        if feats is None:
            merged = group.copy()
            for col in _STRUCTURAL_COLS:
                merged[col] = np.nan
        else:
            merged = group.merge(
                feats[_STRUCTURAL_COLS], left_on="position", right_index=True, how="left"
            )
        parts.append(merged)
    return pd.concat(parts).loc[df.index]


def disagreement_structural_correlation(
    df: pd.DataFrame,
    disagreement_col: str = "disagreement_score",
) -> pd.DataFrame:
    """Spearman correlation of disagreement_col with RSA and contact density.

    Computed per assay and pooled (dms_id == "ALL"). Rows missing either
    variable are dropped per correlation.

    Returns
    -------
    DataFrame with columns [dms_id, n, rsa_spearman_r, rsa_p,
    contact_density_spearman_r, contact_density_p].
    """
    rows = []
    groups = list(df.groupby("dms_id")) + [("ALL", df)]
    for dms_id, group in groups:
        sub = group.dropna(subset=[disagreement_col, "rsa", "contact_density"])
        if len(sub) < 10:
            rows.append(
                {
                    "dms_id": dms_id, "n": len(sub),
                    "rsa_spearman_r": np.nan, "rsa_p": np.nan,
                    "contact_density_spearman_r": np.nan, "contact_density_p": np.nan,
                }
            )
            continue
        r_rsa, p_rsa = stats.spearmanr(sub["rsa"], sub[disagreement_col])
        r_cd, p_cd = stats.spearmanr(sub["contact_density"], sub[disagreement_col])
        rows.append(
            {
                "dms_id": dms_id, "n": len(sub),
                "rsa_spearman_r": r_rsa, "rsa_p": p_rsa,
                "contact_density_spearman_r": r_cd, "contact_density_p": p_cd,
            }
        )
    return pd.DataFrame(rows)[
        ["dms_id", "n", "rsa_spearman_r", "rsa_p", "contact_density_spearman_r", "contact_density_p"]
    ]


def disagreement_by_secondary_structure(
    df: pd.DataFrame,
    disagreement_col: str = "disagreement_score",
) -> pd.DataFrame:
    """Mean/median/std disagreement_col by secondary structure category.

    Reported pooled (dms_id == "ALL") and per assay.

    Returns
    -------
    DataFrame with columns [dms_id, secondary_structure, mean, median, std, count].
    """
    sub = df.dropna(subset=[disagreement_col, "secondary_structure"])

    pooled = sub.groupby("secondary_structure")[disagreement_col].agg(["mean", "median", "std", "count"])
    pooled = pooled.reset_index()
    pooled.insert(0, "dms_id", "ALL")

    per_assay = (
        sub.groupby(["dms_id", "secondary_structure"])[disagreement_col]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )

    return pd.concat([pooled, per_assay], ignore_index=True)[
        ["dms_id", "secondary_structure", "mean", "median", "std", "count"]
    ]


def cluster_accuracy_by_structural_bin(
    df: pd.DataFrame,
    cluster_votes: pd.DataFrame,
    bin_col: str,
    label_col: str = "DMS_score_bin",
) -> pd.DataFrame:
    """Per-cluster majority-vote accuracy, grouped by a structural bin column.

    Mirrors accuracy.per_cluster_accuracy but groups by an arbitrary
    structural column (e.g. "secondary_structure", "rsa_bin") instead of dms_id.
    Rows with a missing bin value are dropped.

    Returns
    -------
    Long-format DataFrame with columns
    [bin_col, cluster, accuracy, precision, recall, f1, mcc, n].
    """
    clusters = cluster_votes.columns.tolist()
    label = df[label_col]
    rows = []
    valid = df[df[bin_col].notna()]
    for bin_value, group in valid.groupby(bin_col):
        idx = group.index
        for cluster in clusters:
            mask = label.loc[idx].notna() & cluster_votes.loc[idx, cluster].notna()
            if mask.sum() == 0:
                rows.append(
                    {
                        bin_col: bin_value, "cluster": cluster,
                        "accuracy": np.nan, "precision": np.nan, "recall": np.nan,
                        "f1": np.nan, "mcc": np.nan, "n": 0,
                    }
                )
                continue
            sub_idx = idx[mask]
            metrics = binary_metrics(label.loc[sub_idx], cluster_votes.loc[sub_idx, cluster])
            rows.append({bin_col: bin_value, "cluster": cluster, **metrics})
    return pd.DataFrame(rows)[[bin_col, "cluster", "accuracy", "precision", "recall", "f1", "mcc", "n"]]


def trust_map(
    df: pd.DataFrame,
    ensemble_vote: pd.Series,
    disagreement_col: str = "disagreement_score",
    label_col: str = "DMS_score_bin",
    n_disagreement_bins: int = 3,
) -> pd.DataFrame:
    """Ensemble-vote accuracy binned by disagreement level x burial (RSA).

    The seed of a meta-confidence signal: cells where accuracy stays high
    despite high disagreement mean disagreement there is "informative but
    not disqualifying"; cells where accuracy craters at high disagreement
    mark where the consensus actually becomes unreliable.

    disagreement_col is split into `n_disagreement_bins` equal-frequency
    bins (qcut); RSA is split at the fixed buried/exposed cutoff (0.25).

    Returns
    -------
    Long-format DataFrame with columns
    [disagreement_bin, rsa_bin, accuracy, mcc, n].
    """
    sub = df.dropna(subset=[disagreement_col, "rsa", label_col]).copy()
    bin_labels = ["low", "mid", "high"][:n_disagreement_bins]
    sub["disagreement_bin"] = pd.qcut(
        sub[disagreement_col], n_disagreement_bins, labels=bin_labels, duplicates="drop"
    )
    sub["rsa_bin"] = pd.cut(
        sub["rsa"], bins=[-np.inf, _RSA_BURIED_CUTOFF, np.inf], labels=["buried", "exposed"]
    )
    vote = ensemble_vote.reindex(sub.index)

    rows = []
    for (dbin, rbin), group in sub.groupby(["disagreement_bin", "rsa_bin"], observed=True):
        idx = group.index
        y_true = sub.loc[idx, label_col]
        y_pred = vote.loc[idx]
        mask = y_true.notna() & y_pred.notna()
        if mask.sum() == 0:
            rows.append({"disagreement_bin": dbin, "rsa_bin": rbin, "accuracy": np.nan, "mcc": np.nan, "n": 0})
            continue
        sub_idx = idx[mask]
        metrics = binary_metrics(y_true.loc[sub_idx], y_pred.loc[sub_idx])
        rows.append(
            {
                "disagreement_bin": dbin, "rsa_bin": rbin,
                "accuracy": metrics["accuracy"], "mcc": metrics["mcc"], "n": metrics["n"],
            }
        )
    return pd.DataFrame(rows)
