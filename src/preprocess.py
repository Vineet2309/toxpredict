"""
preprocess.py
─────────────
Full data pipeline for the Tox21 dataset.

Dataset facts (from inspection):
  - 7,831 compounds
  - 14 columns: 12 task columns + mol_id + smiles
  - Significant NaN labels per task (compounds not tested on every assay)
  - SMILES column name: 'smiles' (lowercase)

Pipeline:
  1. Load CSV
  2. Validate SMILES strings with RDKit
  3. Print class imbalance stats for all 12 assays
  4. Bemis-Murcko scaffold split  →  train / val / test
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from loguru import logger

# ── import project config ─────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    TOX21_CSV,
    TOX21_TASKS,
    TEST_SIZE,
    VAL_SIZE,
    SCAFFOLD_SPLIT,
    RANDOM_STATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_tox21(path: Path = TOX21_CSV) -> pd.DataFrame:
    """
    Load tox21.csv and keep only the columns we need.
    Returns DataFrame with columns: smiles, mol_id, + 12 task columns.
    """
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")

    # Confirm expected columns exist
    missing = [t for t in TOX21_TASKS if t not in df.columns]
    if missing:
        raise ValueError(f"Missing task columns: {missing}")
    if "smiles" not in df.columns:
        raise ValueError("No 'smiles' column found in dataset.")

    # Keep smiles + mol_id + all 12 tasks
    keep = ["smiles", "mol_id"] + TOX21_TASKS
    df = df[keep].copy()

    # Cast task columns to float (keeps NaN as NaN)
    for task in TOX21_TASKS:
        df[task] = pd.to_numeric(df[task], errors="coerce")

    logger.info(f"Loaded {len(df)} compounds with {len(TOX21_TASKS)} tasks")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALIDATE SMILES
# ─────────────────────────────────────────────────────────────────────────────

def validate_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where RDKit cannot parse the SMILES string.
    Also adds a canonical_smiles column (RDKit-normalised form).
    """
    logger.info("Validating SMILES strings with RDKit...")
    initial = len(df)

    def _parse(smi: str):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)   # canonical form
        except Exception:
            return None

    df = df.copy()
    df["canonical_smiles"] = df["smiles"].apply(_parse)

    invalid_mask = df["canonical_smiles"].isna()
    n_invalid = invalid_mask.sum()
    df = df[~invalid_mask].reset_index(drop=True)

    logger.info(f"Removed {n_invalid} invalid SMILES  |  Remaining: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASS STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def class_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each of the 12 assay tasks print:
      - total valid labels (non-NaN)
      - positives (toxic) / negatives (non-toxic)
      - positive rate
      - imbalance ratio  (neg / pos)
    """
    rows = []
    for task in TOX21_TASKS:
        col = df[task]
        n_valid   = col.notna().sum()
        n_pos     = (col == 1).sum()
        n_neg     = (col == 0).sum()
        n_missing = col.isna().sum()
        pos_rate  = n_pos / n_valid if n_valid > 0 else 0.0
        imbalance = round(n_neg / n_pos, 1) if n_pos > 0 else float("inf")

        rows.append({
            "task":            task,
            "valid_labels":    int(n_valid),
            "missing":         int(n_missing),
            "toxic(1)":        int(n_pos),
            "non_toxic(0)":    int(n_neg),
            "positive_rate":   round(pos_rate, 4),
            "imbalance_ratio": imbalance,
        })

    stats_df = pd.DataFrame(rows)
    logger.info(f"\n{stats_df.to_string(index=False)}")
    return stats_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCAFFOLD SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def _get_scaffold(smiles: str) -> str:
    """Compute Bemis-Murcko scaffold for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
    except Exception:
        return ""


def scaffold_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float  = VAL_SIZE,
    seed: int        = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bemis-Murcko scaffold split.

    Why scaffold split instead of random split?
    In a random split, structurally similar molecules can appear in both
    train and test — the model just memorises the scaffold, not the chemistry.
    Scaffold split groups molecules by their core ring system and assigns
    entire groups to one split, giving a much harder and fairer evaluation.

    Algorithm:
      1. Compute scaffold for every molecule
      2. Group molecule indices by scaffold
      3. Sort groups largest to smallest (matches DeepChem convention)
      4. Greedily assign groups to train until quota filled, then val, then test
    """
    logger.info("Computing Bemis-Murcko scaffolds...")
    np.random.seed(seed)

    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    scaffolds: dict = {}
    for idx, smi in enumerate(df[smiles_col]):
        scaf = _get_scaffold(smi)
        scaffolds.setdefault(scaf, []).append(idx)

    # Sort by group size descending
    scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)

    train_cutoff = (1.0 - test_size - val_size) * len(df)
    val_cutoff   = (1.0 - test_size) * len(df)

    train_idx, val_idx, test_idx = [], [], []
    running = 0

    for group in scaffold_groups:
        if running + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif running + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
        running += len(group)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"Scaffold split done  ->  "
        f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def random_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float  = VAL_SIZE,
    seed: int        = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fallback random split (less rigorous than scaffold split)."""
    from sklearn.model_selection import train_test_split

    train_val, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    adjusted_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_val, test_size=adjusted_val, random_state=seed)

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    logger.info(
        f"Random split done  ->  "
        f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. LABEL EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_labels(df: pd.DataFrame, tasks: List[str] = TOX21_TASKS) -> np.ndarray:
    """
    Extract label matrix as float numpy array.
    Shape: (n_samples, n_tasks)
    NaN entries remain as np.nan — training code uses masks to ignore them.
    """
    return df[tasks].values.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL PIPELINE (called by train scripts)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    path: Path       = TOX21_CSV,
    scaffold: bool   = SCAFFOLD_SPLIT,
    test_size: float = TEST_SIZE,
    val_size: float  = VAL_SIZE,
    seed: int        = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end preprocessing.
    Returns (train_df, val_df, test_df) — each DataFrame has all original
    columns plus canonical_smiles.
    """
    df = load_tox21(path)
    df = validate_smiles(df)
    class_statistics(df)

    if scaffold:
        return scaffold_split(df, test_size, val_size, seed)
    else:
        return random_split(df, test_size, val_size, seed)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST  —  python src/preprocess.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train, val, test = run_pipeline()

    print("\n── Sample from train set ──")
    print(train[["smiles", "NR-AR", "NR-AhR", "SR-MMP"]].head(5).to_string(index=False))

    print(f"\nLabel matrix shape (train): {get_labels(train).shape}")
    print("\nNaN per task (train):")
    print(pd.DataFrame(get_labels(train), columns=TOX21_TASKS).isna().sum().to_string())
