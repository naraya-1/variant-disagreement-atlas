"""Fetch and cache AlphaFold DB structures for ProteinGym pilot proteins."""

import re
from pathlib import Path
from typing import Optional

import requests

_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"

# Official UniProt accession pattern (covers both 6-char and 10-char forms)
_ACCESSION_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9])|([A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)


def resolve_uniprot_accession(uniprot_id: str, timeout: float = 15.0) -> Optional[str]:
    """Resolve a ProteinGym UniProt_ID to a canonical UniProt accession.

    ProteinGym uses two conventions for UniProt_ID:
      - reviewed (Swiss-Prot) entries: the mnemonic entry name, e.g. "A4_HUMAN"
      - unreviewed (TrEMBL) entries: "{accession}_{taxon_suffix}", e.g.
        "A0A192B1T2_9HIV1" (the accession is already the leading token)

    Returns the primary accession (e.g. "P05067"), or None if no match is found.
    """
    prefix = uniprot_id.split("_")[0]
    if _ACCESSION_RE.match(prefix):
        return prefix

    resp = requests.get(
        _UNIPROT_SEARCH_URL,
        params={"query": f"id:{uniprot_id}", "fields": "accession,id", "format": "json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None
    return results[0]["primaryAccession"]


def fetch_alphafold_structure(
    uniprot_id: str,
    dest_dir: str = "data/raw/structures",
    timeout: float = 30.0,
) -> Optional[Path]:
    """Download and cache the AlphaFold DB model PDB file for a UniProt ID.

    Looks up the accession (see resolve_uniprot_accession), queries the
    AlphaFold DB prediction API for the latest model version, and downloads
    the PDB file to `dest_dir/{uniprot_id}.pdb`. Per-atom B-factor in the
    downloaded file holds AlphaFold's per-residue pLDDT confidence.

    If a cached file already exists at the destination path, no network
    calls are made and that path is returned immediately.

    Returns
    -------
    Local path to the cached PDB file, or None if no AlphaFold model exists
    for this protein (unresolvable accession, or accession not in AlphaFold DB
    -- e.g. hypervariable / unreviewed sequences without a reference model).
    """
    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir_path / f"{uniprot_id}.pdb"
    if dest_path.exists():
        return dest_path

    accession = resolve_uniprot_accession(uniprot_id, timeout=timeout)
    if accession is None:
        return None

    resp = requests.get(_ALPHAFOLD_API_URL.format(accession=accession), timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    predictions = resp.json()
    if not predictions:
        return None

    pdb_resp = requests.get(predictions[0]["pdbUrl"], timeout=timeout)
    pdb_resp.raise_for_status()
    dest_path.write_text(pdb_resp.text)
    return dest_path
