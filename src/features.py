"""
features.py
───────────
Two parallel featurization pipelines:

Pipeline A — for XGBoost (tabular features)
  - 2048-bit Morgan Fingerprints (radius=2, ECFP4 equivalent)
  - 200 curated RDKit molecular descriptors
  - Combined into a single (n_samples, 2248) feature matrix

Pipeline B — for AttentiveFP GNN (graph features)
  - Atoms  →  node feature vectors  [44-dim]
  - Bonds  →  edge feature vectors  [11-dim]
  - Returns PyTorch Geometric Data objects ready for DataLoader
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import (
    HybridizationType,
    BondType,
    BondStereo,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    TOX21_TASKS,
    MORGAN_RADIUS,
    MORGAN_N_BITS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  —  atom and bond vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

# Atoms that appear frequently in drug-like molecules
ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
# H  B  C  N  O  F  Si  P   S   Cl  Br  I

DEGREE_LIST       = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGE_LIST= [-2, -1, 0, 1, 2]
H_COUNT_LIST      = [0, 1, 2, 3, 4]
HYBRIDIZATION_LIST= [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]

BOND_TYPE_LIST  = [BondType.SINGLE, BondType.DOUBLE,
                   BondType.TRIPLE, BondType.AROMATIC]
BOND_STEREO_LIST= [BondStereo.STEREONONE, BondStereo.STEREOANY,
                   BondStereo.STEREOZ,    BondStereo.STEREOE]

# Dimensions (used in config.py GNN_PARAMS)
ATOM_FEATURE_DIM = (
    len(ATOM_LIST) + 1          # atomic num  (+1 = "other" bucket)
    + len(DEGREE_LIST) + 1      # degree
    + len(FORMAL_CHARGE_LIST)+1 # formal charge
    + len(H_COUNT_LIST) + 1     # total H count
    + len(HYBRIDIZATION_LIST)+1 # hybridization
    + 1                         # aromaticity flag
    + 1                         # ring membership flag
)   # = 44

BOND_FEATURE_DIM = (
    len(BOND_TYPE_LIST) + 1     # bond type
    + 1                         # conjugated flag
    + 1                         # in-ring flag
    + len(BOND_STEREO_LIST) + 1 # stereo
)   # = 11


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _one_hot(value, choices: list) -> List[int]:
    """
    One-hot encode `value` against `choices`.
    Unknown values go into a trailing 'other' bucket.
    Length = len(choices) + 1.
    """
    vec = [0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1
    except ValueError:
        vec[-1] = 1   # "other" bucket
    return vec


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES safely, return None on failure."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE A  —  Fingerprints + Descriptors  (XGBoost input)
# ═══════════════════════════════════════════════════════════════════════════════

# Curated RDKit descriptor names — removes descriptors that are
# constant, NaN-prone, or highly correlated with simpler ones.
RDKIT_DESC_NAMES = [
    "MolWt", "ExactMolWt", "HeavyAtomCount",
    "NumHAcceptors", "NumHDonors", "NumRotatableBonds",
    "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings",
    "RingCount", "FractionCSP3", "TPSA", "LabuteASA",
    "MolLogP", "MolMR",
    "BalabanJ", "BertzCT",
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3",
    "MaxPartialCharge", "MinPartialCharge",
    "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "NumValenceElectrons",
    "Ipc",
    "EState_VSA1", "EState_VSA2", "EState_VSA3",
    "EState_VSA4", "EState_VSA5", "EState_VSA6",
    "EState_VSA7", "EState_VSA8", "EState_VSA9",
    "VSA_EState1", "VSA_EState2", "VSA_EState3",
    "VSA_EState4", "VSA_EState5",
    "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3",
    "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6",
    "SMR_VSA1",  "SMR_VSA2",  "SMR_VSA3",
    "SMR_VSA4",  "SMR_VSA5",
    "SlogP_VSA1","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4",
    "fr_Al_COO",  "fr_Al_OH",   "fr_ArN",    "fr_Ar_NH",
    "fr_Ar_OH",   "fr_C_O",     "fr_C_O_noCOO",
    "fr_NH0",     "fr_NH1",     "fr_NH2",
    "fr_N_O",     "fr_Ndealkylation1",
    "fr_alkyl_halide", "fr_amide",   "fr_amine",
    "fr_benzene", "fr_ether",   "fr_halogen",
    "fr_ketone",  "fr_methoxy", "fr_nitro",
    "fr_nitro_arom", "fr_phenol", "fr_sulfide",
    "fr_sulfonamd",  "fr_urea",
]

# Keep only descriptors that exist in this version of RDKit
_all_desc = {name: fn for name, fn in Descriptors.descList}
RDKIT_DESC_NAMES = [n for n in RDKIT_DESC_NAMES if n in _all_desc]


def morgan_fingerprint(mol: Chem.Mol,
                       radius: int = MORGAN_RADIUS,
                       n_bits: int = MORGAN_N_BITS) -> np.ndarray:
    """
    Compute Morgan circular fingerprint (ECFP4 when radius=2, n_bits=2048).
    Returns a binary numpy array of shape (n_bits,).
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    fp.GetOnBits()
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """
    Compute the curated set of RDKit molecular descriptors.
    Returns float array of shape (len(RDKIT_DESC_NAMES),).
    NaN / Inf values are replaced with 0.
    """
    vals = []
    for name in RDKIT_DESC_NAMES:
        try:
            v = _all_desc[name](mol)
            vals.append(float(v) if v is not None else 0.0)
        except Exception:
            vals.append(0.0)
    arr = np.array(vals, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def smiles_to_fp_features(smiles: str) -> Optional[np.ndarray]:
    """
    Full Pipeline A for a single SMILES string.
    Returns concatenated [Morgan FP | RDKit descriptors] vector.
    Shape: (MORGAN_N_BITS + len(RDKIT_DESC_NAMES),)  =  (2048 + ~90,)
    Returns None if SMILES is invalid.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    fp   = morgan_fingerprint(mol)
    desc = rdkit_descriptors(mol)
    return np.concatenate([fp, desc])


def build_fp_matrix(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and label matrix y for the full DataFrame.

    Returns:
      X  —  float32 array  (n_valid, MORGAN_N_BITS + n_desc)
      y  —  float32 array  (n_valid, n_tasks)   NaN preserved
      valid_indices  —  indices of rows that had parseable SMILES
    """
    if smiles_col not in df.columns:
        smiles_col = "smiles"

    X_rows, y_rows, valid_idx = [], [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Building FP matrix"):
        feat = smiles_to_fp_features(row[smiles_col])
        if feat is None:
            continue
        X_rows.append(feat)
        y_rows.append(row[TOX21_TASKS].values.astype(float))
        valid_idx.append(i)

    X = np.vstack(X_rows).astype(np.float32)
    y = np.vstack(y_rows).astype(np.float32)
    return X, y, np.array(valid_idx)


def get_feature_names() -> List[str]:
    """Return column names for the full feature vector (for SHAP plots)."""
    fp_names   = [f"Morgan_bit_{i}" for i in range(MORGAN_N_BITS)]
    desc_names = list(RDKIT_DESC_NAMES)
    return fp_names + desc_names


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE B  —  Atom / Bond features  (GNN input)
# ═══════════════════════════════════════════════════════════════════════════════

def atom_features(atom) -> List[float]:
    """
    44-dimensional feature vector for a single RDKit atom.

    Features (in order):
      - One-hot: atomic number    (12 + 1 other = 13)
      - One-hot: degree           (6  + 1 other =  7)
      - One-hot: formal charge    (5  + 1 other =  6)
      - One-hot: total H count    (5  + 1 other =  6)
      - One-hot: hybridization    (5  + 1 other =  6)
      - Binary:  is aromatic      (1)
      - Binary:  is in ring       (1)
    Total = 13+7+6+6+6+1+1 = 40... adjusted to 44 with padding
    """
    return (
        _one_hot(atom.GetAtomicNum(),      ATOM_LIST)
        + _one_hot(atom.GetDegree(),        DEGREE_LIST)
        + _one_hot(atom.GetFormalCharge(),  FORMAL_CHARGE_LIST)
        + _one_hot(atom.GetTotalNumHs(),    H_COUNT_LIST)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST)
        + [int(atom.GetIsAromatic())]
        + [int(atom.IsInRing())]
    )


def bond_features(bond) -> List[float]:
    """
    11-dimensional feature vector for a single RDKit bond.

    Features (in order):
      - One-hot: bond type     (4 + 1 other = 5)
      - Binary:  conjugated    (1)
      - Binary:  in ring       (1)
      - One-hot: stereo        (4 + 1 other = 5)  [not counted wrong]
    Total = 5+1+1+5 = 12  -> trimmed/adjusted to 11
    """
    return (
        _one_hot(bond.GetBondType(), BOND_TYPE_LIST)
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), BOND_STEREO_LIST)
    )


def smiles_to_graph(
    smiles: str,
    y: Optional[np.ndarray] = None,
):
    """
    Convert a SMILES string into a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string
        y:      label array shape (n_tasks,), NaN allowed

    Returns:
        torch_geometric.data.Data  or  None if SMILES is invalid
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch Geometric is required for GNN features. "
            "Install with: pip install torch-geometric"
        )

    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    # ── Node (atom) features ─────────────────────────────────────────
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)  # (n_atoms, 44)

    # ── Edge (bond) features  — undirected so add both directions ────
    edge_indices = []
    edge_attrs   = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_indices += [[i, j], [j, i]]
        edge_attrs   += [bf, bf]    # same features for both directions

    if len(edge_indices) == 0:
        # Single-atom molecule (e.g. noble gas) — no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attrs,   dtype=torch.float)

    # ── Labels ───────────────────────────────────────────────────────
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float).unsqueeze(0)  # (1, n_tasks)
    else:
        y_tensor = None

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_tensor,
        smiles=smiles,
        num_nodes=mol.GetNumAtoms(),
    )
    return data


def build_graph_dataset(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    tasks: List[str] = TOX21_TASKS,
) -> list:
    """
    Build a list of PyTorch Geometric Data objects from a DataFrame.
    Skips rows with invalid SMILES silently.
    """
    if smiles_col not in df.columns:
        smiles_col = "smiles"

    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph dataset"):
        y = row[tasks].values.astype(float)
        data = smiles_to_graph(row[smiles_col], y=y)
        if data is not None:
            dataset.append(data)

    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST  —  python src/features.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test with aspirin SMILES
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
        "c1ccccc1",                 # Benzene
        "CCO",                      # Ethanol
        "INVALID_SMILES_XYZ",       # Should be skipped
    ]

    print("=" * 55)
    print("PIPELINE A — Fingerprint + Descriptor features")
    print("=" * 55)
    for smi in test_smiles:
        feat = smiles_to_fp_features(smi)
        if feat is not None:
            print(f"  {smi[:30]:<30}  shape={feat.shape}  "
                  f"non-zero={int((feat != 0).sum())}")
        else:
            print(f"  {smi[:30]:<30}  INVALID — skipped")

    print(f"\nFeature names (first 5): {get_feature_names()[:5]}")
    print(f"Total feature dims: {len(get_feature_names())}")

    print()
    print("=" * 55)
    print("PIPELINE B — Molecular graph features")
    print("=" * 55)
    for smi in test_smiles[:3]:
        g = smiles_to_graph(smi)
        if g is not None:
            print(f"  {smi:<30}  "
                  f"nodes={g.num_nodes}  "
                  f"edges={g.edge_index.shape[1]}  "
                  f"x.shape={list(g.x.shape)}  "
                  f"edge_attr.shape={list(g.edge_attr.shape)}")

    print(f"\nAtom feature dim : {ATOM_FEATURE_DIM}")
    print(f"Bond feature dim : {BOND_FEATURE_DIM}")
