"""Load ProteinGym DMS substitution zero-shot scores into a tidy DataFrame.

Expected data layout after running download_data.py:
  data/raw/DMS_substitutions.csv          # assay metadata
  data/raw/zero_shot_substitutions_scores/ # one CSV per DMS assay

Each per-assay CSV contains:
  mutant            e.g. "A123V" or "A123V:D456E"
  mutated_sequence  full protein sequence with mutation(s) applied
  DMS_score         experimental fitness (if present in the scores file)
  <model_name>      one column per zero-shot model (45-70+ models)
"""

import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from tqdm import tqdm

_MUTANT_RE = re.compile(r"^([A-Z*])(\d+)([A-Z*])$")

# Columns that are not model scores
_NON_MODEL_COLS = frozenset(
    {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin",
     "dms_id", "protein_id", "position", "wt_aa", "mut_aa", "experimental_fitness"}
)

# Files present in data/raw/ that are not score CSVs
_EXCLUDED_FILES = frozenset({"DMS_substitutions.csv"})


def _parse_single_mutant(mutant: str):
    """Return (wt_aa, position, mut_aa) for 'A123V', or (None, None, None)."""
    if ":" in mutant:
        return None, None, None
    m = _MUTANT_RE.match(mutant)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def _parse_mutants(series: pd.Series) -> pd.DataFrame:
    parsed = series.map(_parse_single_mutant)
    wt   = parsed.map(lambda t: t[0])
    pos  = parsed.map(lambda t: t[1]).astype("Int64")
    mut  = parsed.map(lambda t: t[2])
    return pd.DataFrame({"wt_aa": wt, "position": pos, "mut_aa": mut})


def _find_scores_dir(data_dir: Path) -> Path:
    candidate = data_dir / "zero_shot_substitutions_scores"
    if candidate.exists() and any(candidate.glob("*.csv")):
        return candidate
    # Fallback: CSV files dumped directly into data_dir
    if any(data_dir.glob("*.csv")):
        return data_dir
    raise FileNotFoundError(
        f"No score CSVs found under {data_dir}. Run download_data.py first."
    )


def get_model_columns(
    df: pd.DataFrame,
    exclude: tuple = ("DMS_score_bin",),
) -> list:
    """Return the list of zero-shot model score columns in df.

    Strips fixed metadata columns and any additional names in `exclude`.
    Pass exclude=() to skip no extras beyond the built-in non-model set.
    """
    extra = set(exclude)
    return [c for c in df.columns if c not in _NON_MODEL_COLS and c not in extra]


def load_dms_scores(
    data_dir: str = "data/raw",
    dms_ids: Optional[Sequence[str]] = None,
    n_proteins: Optional[int] = None,
) -> pd.DataFrame:
    """Load zero-shot model scores into a tidy DataFrame.

    Parameters
    ----------
    data_dir:
        Root directory containing DMS_substitutions.csv and the extracted
        zero_shot_substitutions_scores/ folder.
    dms_ids:
        Restrict to these DMS assay IDs (file stems). None = load all.
    n_proteins:
        Load only the first N assay files (alphabetical order). Useful for
        quick exploration when the full dataset (~4.4 GB) is too large.

    Returns
    -------
    DataFrame with columns:
        protein_id, dms_id, mutant, position, wt_aa, mut_aa,
        experimental_fitness, DMS_score_bin,
        <model_col_1>, ..., <model_col_N>

    DMS_score_bin is ProteinGym's curator-assigned binary ground-truth label
    (experimental_fitness thresholded at an assay-specific cutoff — see
    DMS_binarization_cutoff in DMS_substitutions.csv — not a generic median
    split).
    """
    data_dir = Path(data_dir)
    scores_dir = _find_scores_dir(data_dir)
    csv_files = sorted(scores_dir.glob("*.csv"))

    # Exclude known non-score files (e.g. DMS_substitutions.csv when flat layout)
    csv_files = [f for f in csv_files if f.name not in _EXCLUDED_FILES]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {scores_dir}.")

    if dms_ids is not None:
        wanted = set(dms_ids)
        csv_files = [f for f in csv_files if f.stem in wanted]
        if not csv_files:
            raise ValueError(f"None of the requested dms_ids were found: {dms_ids}")

    if n_proteins is not None:
        csv_files = csv_files[:n_proteins]

    # Load per-assay CSVs
    frames = []
    for csv_file in tqdm(csv_files, desc="Loading assays", unit="assay"):
        df = pd.read_csv(csv_file, low_memory=False)
        df.insert(0, "dms_id", csv_file.stem)
        frames.append(df)

    if not frames:
        raise ValueError("No data loaded. Check dms_ids / n_proteins arguments.")

    combined = pd.concat(frames, ignore_index=True)

    # Join reference metadata for protein_id (UniProt_ID) and other fields
    ref_path = data_dir / "DMS_substitutions.csv"
    if ref_path.exists():
        ref = pd.read_csv(ref_path, usecols=["DMS_id", "UniProt_ID"])
        ref = ref.rename(columns={"DMS_id": "dms_id", "UniProt_ID": "protein_id"})
        combined = combined.merge(ref, on="dms_id", how="left")
    else:
        print(
            f"Warning: reference file not found at {ref_path}. "
            "protein_id will be set to dms_id."
        )
        combined["protein_id"] = combined["dms_id"]

    # Parse mutant notation → wt_aa, position, mut_aa
    parsed = _parse_mutants(combined["mutant"])
    combined = pd.concat([combined, parsed], axis=1)

    # Normalise experimental fitness column
    if "DMS_score" in combined.columns:
        combined = combined.rename(columns={"DMS_score": "experimental_fitness"})
    else:
        combined["experimental_fitness"] = pd.NA

    # Identify model columns (everything not in the fixed set)
    model_cols = [c for c in combined.columns if c not in _NON_MODEL_COLS]

    # Drop bulky non-model columns (keep DMS_score_bin — it's ground truth)
    combined = combined.drop(columns=["mutated_sequence"], errors="ignore")
    if "DMS_score_bin" not in combined.columns:
        combined["DMS_score_bin"] = pd.NA
    combined["DMS_score_bin"] = combined["DMS_score_bin"].astype("Int8")

    # Final column order
    fixed = ["protein_id", "dms_id", "mutant", "position", "wt_aa", "mut_aa",
             "experimental_fitness", "DMS_score_bin"]
    combined = combined[fixed + model_cols]

    # Print summary
    n_assays    = combined["dms_id"].nunique()
    n_proteins_ = combined["protein_id"].nunique()
    n_variants  = len(combined)
    print(f"\n{'='*50}")
    print(f"Proteins (UniProt IDs) : {n_proteins_}")
    print(f"DMS assays             : {n_assays}")
    print(f"Zero-shot models       : {len(model_cols)}")
    print(f"Total variant rows     : {n_variants:,}")
    print(f"Models found           : {model_cols[:6]}{'...' if len(model_cols) > 6 else ''}")
    print(f"{'='*50}\n")

    return combined
