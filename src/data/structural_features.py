"""Per-residue structural features (solvent exposure, secondary structure,
contact density, confidence) from AlphaFold DB models."""

import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
from scipy.spatial import cKDTree

# Theoretical maximum solvent accessible surface area per residue (A^2),
# Tien et al. 2013 (empirical extended tripeptide scale). Used to normalize
# raw SASA into relative solvent accessibility (RSA, 0-1+).
_MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

_SSE_LABELS = {"a": "helix", "b": "sheet", "c": "coil"}


def compute_structural_features(
    pdb_path: str,
    contact_radius: float = 10.0,
) -> pd.DataFrame:
    """Compute per-residue structural features from an AlphaFold PDB model.

    Parameters
    ----------
    pdb_path:
        Path to an AlphaFold DB model PDB file (single chain; per-atom
        B-factor holds AlphaFold's per-residue pLDDT confidence).
    contact_radius:
        Radius (Angstrom) for counting CA-CA neighbors around each residue's
        CA atom -- a simple local packing / burial proxy (higher = more
        buried/packed core, lower = more exposed/flexible).

    Returns
    -------
    DataFrame indexed by `position` (1-based UniProt/PDB residue number)
    with columns: wt_aa_pdb, plddt, sasa, rsa, secondary_structure,
    contact_density.
    """
    pdb_file = pdb_io.PDBFile.read(pdb_path)
    all_atoms = pdb_io.get_structure(pdb_file, model=1)
    bfactors_all = pdb_file.get_b_factor(model=1)

    aa_mask = struc.filter_amino_acids(all_atoms)
    protein = all_atoms[aa_mask]
    bfactors = bfactors_all[aa_mask]

    ca_mask = protein.atom_name == "CA"
    ca_atoms = protein[ca_mask]
    ca_bfactors = bfactors[ca_mask]

    sasa_atom = struc.sasa(protein, vdw_radii="ProtOr")
    res_ids = protein.res_id
    res_sasa = {
        rid: float(np.nansum(sasa_atom[res_ids == rid])) for rid in np.unique(res_ids)
    }

    sse = struc.annotate_sse(protein)  # one label per residue, ordered like ca_atoms

    tree = cKDTree(ca_atoms.coord)
    contact_density = tree.query_ball_point(ca_atoms.coord, r=contact_radius, return_length=True) - 1

    positions = ca_atoms.res_id
    res_names = ca_atoms.res_name
    sasa_values = np.array([res_sasa[rid] for rid in positions])
    max_asa = np.array([_MAX_ASA.get(name, np.nan) for name in res_names])
    rsa = sasa_values / max_asa

    out = pd.DataFrame(
        {
            "position": positions,
            "wt_aa_pdb": res_names,
            "plddt": ca_bfactors,
            "sasa": sasa_values,
            "rsa": rsa,
            "secondary_structure": [_SSE_LABELS.get(s, "coil") for s in sse],
            "contact_density": contact_density,
        }
    ).set_index("position")
    return out
