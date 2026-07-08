"""Disagreement metrics between zero-shot variant effect predictors."""

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm


def binarize_scores(
    df: pd.DataFrame,
    models: Sequence[str],
    method: str = "per_assay_median",
) -> pd.DataFrame:
    """Add {model}_binary columns using per-assay-per-model median split.

    Each (assay, model) pair gets its own median threshold, so models with
    incompatible score scales (log-likelihoods vs. pseudo-likelihoods vs.
    correlation scores) are treated fairly.

    Binary value: 1 = above median (predicted more fit), 0 = below, NA = missing score.
    """
    if method != "per_assay_median":
        raise ValueError(f"Unsupported method: {method!r}. Only 'per_assay_median' is supported.")

    out = df.copy()
    for model in models:
        if model not in df.columns:
            raise KeyError(f"Model column not found: {model!r}")
        thresholds = df.groupby("dms_id")[model].transform("median")
        binary = (df[model] > thresholds).astype("Int8")
        binary[df[model].isna()] = pd.NA
        out[f"{model}_binary"] = binary
    return out


def pairwise_agreement(
    df_binary: pd.DataFrame,
    models: Sequence[str],
) -> pd.DataFrame:
    """Compute Cohen's kappa for every pair of models.

    Rows where either model has a missing binary score are dropped before
    computing kappa for that pair.

    Returns
    -------
    Symmetric DataFrame (models × models) of kappa values.
    Diagonal = 1.0.  Pairs with fewer than 10 valid rows = NaN.
    """
    models = list(models)
    n = len(models)
    kappa = np.full((n, n), np.nan)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for i, j in tqdm(pairs, desc="Pairwise kappa", unit="pair"):
        col_i = f"{models[i]}_binary"
        col_j = f"{models[j]}_binary"
        mask = df_binary[col_i].notna() & df_binary[col_j].notna()
        if mask.sum() < 10:
            continue
        try:
            k = cohen_kappa_score(
                df_binary.loc[mask, col_i].astype(int),
                df_binary.loc[mask, col_j].astype(int),
            )
        except Exception:
            k = np.nan
        kappa[i, j] = k
        kappa[j, i] = k

    np.fill_diagonal(kappa, 1.0)
    return pd.DataFrame(kappa, index=models, columns=models)


def variant_disagreement_score(
    df_binary: pd.DataFrame,
    models: Sequence[str],
) -> pd.Series:
    """Binary entropy of per-variant model predictions.

    For each variant, p = fraction of models predicting 1.
    H(p) = -p·log₂(p) - (1-p)·log₂(1-p)

    H = 0 when all models agree (p=0 or p=1).
    H = 1 bit when exactly half predict 1 and half predict 0.

    NaN model scores are excluded from the fraction (denominator shrinks).
    Returns a Series aligned to df_binary.index.
    """
    binary_cols = [f"{m}_binary" for m in models]
    data = df_binary[binary_cols].astype(float)  # NaN preserved as float NaN

    p = data.mean(axis=1)  # NaN-aware mean across models
    eps = 1e-10
    entropy = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))
    entropy = entropy.clip(lower=0.0)  # numerical noise can push below 0
    return entropy.rename("disagreement_score")


def score_spread(
    df: pd.DataFrame,
    models: Sequence[str],
) -> pd.Series:
    """Std of per-assay z-scored continuous scores across models per variant.

    1. Z-score each model's continuous scores within each assay (mean 0, std 1),
       removing inter-model scale and inter-assay offset differences.
    2. Take std across the z-scored model columns per variant row.

    Returns a Series aligned to df.index.
    """
    zscored = {}
    for model in models:
        z = df.groupby("dms_id")[model].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-10)
        )
        zscored[model] = z

    z_df = pd.DataFrame(zscored, index=df.index)
    return z_df.std(axis=1).rename("score_spread")
