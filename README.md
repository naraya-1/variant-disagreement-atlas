# Variant Effect Prediction Disagreement Atlas

A systematic analysis of where state-of-the-art zero-shot variant effect predictors *disagree* — and what those disagreements reveal about the limits of each model family.

We use the [ProteinGym](https://proteingym.org) DMS substitution benchmark (v1.3) as ground truth: 250+ deep mutational scanning assays across diverse proteins, paired with pre-computed scores from 70+ zero-shot models (ESM-1b, EVE, Tranception, ProteinMPNN, and many others).

## Goals

- Map which proteins / positions drive the largest inter-model disagreement
- Cluster models by their disagreement patterns (not just benchmark rank)
- Correlate disagreement with structural features (solvent exposure, contact density)
- Identify variants where consensus is wrong vs. where disagreement is informative

## Project layout

```
download_data.py            # download ProteinGym benchmark data
src/
  data/
    proteingym_loader.py    # load scores into a tidy DataFrame
    alphafold_fetcher.py    # fetch AlphaFold structures (planned)
    structural_features.py  # per-residue structural features (planned)
    uniprot_extractor.py    # UniProt metadata (planned)
  analysis/
    disagreement.py         # pairwise / ensemble disagreement metrics (planned)
    clustering.py           # model clustering (planned)
    meta_confidence.py      # meta-predictor of when consensus is reliable (planned)
  viz/
    heatmaps.py             # disagreement heatmaps (planned)
    structure_3d.py         # 3-D structure overlays (planned)
notebooks/
  01_pilot_exploration.ipynb  # summary stats, sanity checks
app/
  streamlit_app.py            # interactive explorer (planned)
data/
  raw/                        # downloaded ProteinGym files (gitignored)
results/
  figures/                    # output plots
```

## Quickstart

### 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Download data

```bash
python download_data.py
```

This downloads two things into `data/raw/`:

| File | Size | Description |
|------|------|-------------|
| `DMS_substitutions.csv` | ~200 KB | Reference metadata (one row per assay) |
| `zero_shot_substitutions_scores/` | ~4.4 GB | Per-assay CSVs with model predictions |

Downloads are resumable — interrupt and re-run freely.

### 3 — Explore in the notebook

```bash
cd notebooks
jupyter lab 01_pilot_exploration.ipynb
```

Or load programmatically:

```python
from src.data.proteingym_loader import load_dms_scores

# Load first 10 assays (fast, ~few hundred MB)
df = load_dms_scores("data/raw", n_proteins=10)

# Load specific assays
df = load_dms_scores("data/raw", dms_ids=["BLAT_ECOLX_Jacquier_2013"])

# Load everything (requires ~8+ GB RAM)
df = load_dms_scores("data/raw")
```

The returned DataFrame has columns:
`protein_id`, `dms_id`, `mutant`, `position`, `wt_aa`, `mut_aa`,
`experimental_fitness`, then one column per zero-shot model.

## Artifacts & reproducibility

All analysis outputs are saved through `src/utils/persistence.py`. Every notebook
should import and use these helpers — never call `plt.savefig` or `df.to_csv` directly.

| Location | What lives there | Git |
|---|---|---|
| `results/figures/` | PNG (300 dpi) + PDF for every plot | ✅ tracked |
| `results/tables/` | CSV / Parquet for every key table | ✅ tracked |
| `data/processed/` | Intermediate DataFrames and arrays | ❌ gitignored (large) |
| `results/ARTIFACTS_LOG.csv` | Append-only log of every saved artifact | ✅ tracked |

Filenames are **auto-versioned**: if `my_figure.png` already exists, the next save
produces `my_figure_v2.png`. Old versions are never overwritten.

### Usage in notebooks

```python
from src.utils.persistence import save_figure, save_table, save_processed

NB = "02_pilot_disagreement"   # set once per notebook

# figures — accepts matplotlib Figure or seaborn ClusterGrid
save_figure(fig, "pairwise_agreement_clustermap", notebook=NB,
            description="95×95 Cohen's κ clustermap across 5 pilot assays")

# tables
save_table(df, "top_30_disagreement_variants", notebook=NB,
           description="Top-30 most-disagreed variants with key model scores")

# intermediates (DataFrame → parquet, ndarray → .npy, other → pickle)
save_processed(df_binary, "pilot_binarized_scores", notebook=NB,
               description="Binarized predictions for 5 pilot assays")
```

## Data source

**ProteinGym v1.3** — Notin et al., NeurIPS 2023.  
Download URL: `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/zero_shot_substitutions_scores.zip`  
Also on Zenodo: [10.5281/zenodo.15293562](https://doi.org/10.5281/zenodo.15293562)
