"""
train_xgb.py
────────────
Multi-task XGBoost training for Tox21 drug toxicity prediction.

One XGBoost classifier is trained per assay (12 total).
Each model handles class imbalance via scale_pos_weight.
SHAP TreeExplainer is used for feature-level explainability.

Saves to:
  models/xgb_multitask.joblib   — dict of {task: trained XGBClassifier}
  models/xgb_shap_values.npy    — SHAP values array (n_test, n_features, n_tasks)
  models/xgb_results.csv        — per-task AUC / F1 / AUPRC metrics
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import joblib
from tqdm import tqdm
from loguru import logger
import xgboost as xgb
import shap

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    TOX21_CSV,
    TOX21_TASKS,
    XGB_PARAMS,
    XGB_MODEL_PATH,
    MODELS_DIR,
    TOXICITY_THRESHOLD,
    SHAP_TOP_N,
    RANDOM_STATE,
)
from preprocess import run_pipeline, get_labels
from features  import build_fp_matrix, get_feature_names


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_scale_pos_weight(y_col: np.ndarray) -> float:
    """
    XGBoost scale_pos_weight = n_negative / n_positive.
    This corrects for the extreme class imbalance in Tox21
    (toxic compounds are rare — typically 3-15% of labeled data).
    We cap at 50 to avoid over-correcting on very sparse assays.
    """
    valid  = y_col[~np.isnan(y_col)]
    n_pos  = (valid == 1).sum()
    n_neg  = (valid == 0).sum()
    if n_pos == 0:
        return 1.0
    weight = n_neg / n_pos
    return float(min(weight, 50.0))


def _mask_task(X: np.ndarray,
               y_col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove rows where the label is NaN.
    Tox21 compounds are not tested on every assay — NaN means 'not tested',
    not 'non-toxic'. We must exclude them from training and evaluation.
    """
    valid_mask = ~np.isnan(y_col)
    return X[valid_mask], y_col[valid_mask].astype(int)


def _evaluate(model, X: np.ndarray,
              y: np.ndarray) -> Dict[str, float]:
    """
    Compute ROC-AUC, AUPRC, and F1 for one task.
    Returns dict with the three metrics.
    """
    X_m, y_m = _mask_task(X, y)
    if len(np.unique(y_m)) < 2:
        return {"roc_auc": float("nan"),
                "auprc":   float("nan"),
                "f1":      float("nan")}

    proba = model.predict_proba(X_m)[:, 1]
    preds = (proba >= TOXICITY_THRESHOLD).astype(int)

    return {
        "roc_auc": round(roc_auc_score(y_m, proba), 4),
        "auprc":   round(average_precision_score(y_m, proba), 4),
        "f1":      round(f1_score(y_m, preds, zero_division=0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_all_tasks(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> Dict[str, xgb.XGBClassifier]:
    """
    Train one XGBClassifier per Tox21 task.

    For each task:
      1. Drop NaN-labelled rows
      2. Compute scale_pos_weight for class imbalance
      3. Train with early stopping on val AUC
      4. Log val ROC-AUC
    """
    models = {}

    for i, task in enumerate(TOX21_TASKS):
        logger.info(f"[{i+1:2d}/{len(TOX21_TASKS)}] Training  {task}")

        X_tr, y_tr = _mask_task(X_train, y_train[:, i])
        X_vl, y_vl = _mask_task(X_val,   y_val[:, i])

        if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
            logger.warning(f"  Skipping {task} — insufficient labelled data")
            continue

        spw = _compute_scale_pos_weight(y_train[:, i])
        logger.info(f"  Samples: {len(X_tr)} train / {len(X_vl)} val  |  "
                    f"scale_pos_weight={spw:.1f}")

        params = {**XGB_PARAMS, "scale_pos_weight": spw}
        model  = xgb.XGBClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False,
        )

        # Quick val AUC
        if len(np.unique(y_vl)) >= 2:
            proba   = model.predict_proba(X_vl)[:, 1]
            val_auc = roc_auc_score(y_vl, proba)
            logger.info(f"  Val ROC-AUC = {val_auc:.4f}")

        models[task] = model

    return models


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    models:  Dict[str, xgb.XGBClassifier],
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> pd.DataFrame:
    """
    Run test-set evaluation for all 12 tasks.
    Returns a DataFrame with columns: task, roc_auc, auprc, f1.
    """
    rows = []
    for i, task in enumerate(TOX21_TASKS):
        if task not in models:
            rows.append({"task": task,
                         "roc_auc": float("nan"),
                         "auprc":   float("nan"),
                         "f1":      float("nan")})
            continue
        metrics = _evaluate(models[task], X_test, y_test[:, i])
        rows.append({"task": task, **metrics})

    results_df = pd.DataFrame(rows)

    # Summary row
    mean_auc   = results_df["roc_auc"].dropna().mean()
    mean_auprc = results_df["auprc"].dropna().mean()
    mean_f1    = results_df["f1"].dropna().mean()

    logger.info("\n" + "="*52)
    logger.info(results_df.to_string(index=False))
    logger.info("="*52)
    logger.info(f"Mean ROC-AUC : {mean_auc:.4f}")
    logger.info(f"Mean AUPRC   : {mean_auprc:.4f}")
    logger.info(f"Mean F1      : {mean_f1:.4f}")
    logger.info("="*52)

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(
    models:  Dict[str, xgb.XGBClassifier],
    X_test:  np.ndarray,
    feat_names: list,
    top_n: int = SHAP_TOP_N,
) -> Dict:
    """
    Compute SHAP values for every task using TreeExplainer.

    Returns a dict:
      {
        "shap_matrix": np.ndarray  (n_test, n_features, n_tasks),
        "top_features": {
            task: [{"feature": str, "mean_abs_shap": float}, ...]
        }
      }

    The shap_matrix is saved to disk for the API to load.
    top_features is the summary used in the frontend bar charts.
    """
    logger.info("Computing SHAP values...")

    n_test     = X_test.shape[0]
    n_features = X_test.shape[1]
    n_tasks    = len(TOX21_TASKS)

    shap_matrix  = np.zeros((n_test, n_features, n_tasks), dtype=np.float32)
    top_features = {}

    for i, task in enumerate(tqdm(TOX21_TASKS, desc="SHAP")):
        if task not in models:
            continue

        explainer  = shap.TreeExplainer(models[task])
        shap_vals  = explainer.shap_values(X_test)

        # shap_values returns list [class0, class1] for binary classifiers
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # use class=1 (toxic) SHAP values

        shap_matrix[:, :, i] = shap_vals.astype(np.float32)

        # Top-N features by mean absolute SHAP value
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]
        top_features[task] = [
            {
                "feature":       feat_names[j],
                "mean_abs_shap": round(float(mean_abs[j]), 6),
            }
            for j in top_idx
        ]

    return {"shap_matrix": shap_matrix, "top_features": top_features}


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-COMPOUND PREDICTION  (used by FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(
    smiles: str,
    models: Dict[str, xgb.XGBClassifier],
    feat_names: list,
    top_n: int = SHAP_TOP_N,
) -> Dict:
    """
    Predict toxicity for a single SMILES string.

    Returns:
      {
        "predictions": {task: {"probability": float, "toxic": bool}},
        "shap_top":    {task: [{"feature": str, "shap_value": float}]}
      }
    """
    from features import smiles_to_fp_features
    feat = smiles_to_fp_features(smiles)
    if feat is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    X = feat.reshape(1, -1)
    predictions = {}
    shap_top    = {}

    for task, model in models.items():
        prob  = float(model.predict_proba(X)[0, 1])
        toxic = prob >= TOXICITY_THRESHOLD
        predictions[task] = {"probability": round(prob, 4), "toxic": toxic}

        # Per-compound SHAP
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv[0]   # shape (n_features,)

        top_idx     = np.argsort(np.abs(sv))[::-1][:top_n]
        shap_top[task] = [
            {"feature": feat_names[j], "shap_value": round(float(sv[j]), 6)}
            for j in top_idx
        ]

    return {"predictions": predictions, "shap_top": shap_top}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("══════════════════════════════════════════")
    logger.info("  ToxPredict — XGBoost Training Pipeline  ")
    logger.info("══════════════════════════════════════════")

    # 1. Preprocessing
    logger.info("\n[1/5] Preprocessing data...")
    train_df, val_df, test_df = run_pipeline(TOX21_CSV)

    # 2. Feature engineering
    logger.info("\n[2/5] Building fingerprint feature matrices...")
    X_train, y_train, _ = build_fp_matrix(train_df)
    X_val,   y_val,   _ = build_fp_matrix(val_df)
    X_test,  y_test,  _ = build_fp_matrix(test_df)
    feat_names = get_feature_names()

    logger.info(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    logger.info(f"  X_val  : {X_val.shape}    y_val  : {y_val.shape}")
    logger.info(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")

    # 3. Train
    logger.info("\n[3/5] Training XGBoost models (one per task)...")
    models = train_all_tasks(X_train, y_train, X_val, y_val)

    # 4. Evaluate
    logger.info("\n[4/5] Evaluating on test set...")
    results_df = evaluate_all(models, X_test, y_test)

    # 5. SHAP
    logger.info("\n[5/5] Computing SHAP values on test set...")
    shap_out = compute_shap(models, X_test, feat_names)

    # ── Save everything ──────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Trained models
    joblib.dump(models, XGB_MODEL_PATH)
    logger.info(f"  Models saved  →  {XGB_MODEL_PATH}")

    # SHAP matrix
    shap_matrix_path = MODELS_DIR / "xgb_shap_values.npy"
    np.save(shap_matrix_path, shap_out["shap_matrix"])
    logger.info(f"  SHAP matrix saved  →  {shap_matrix_path}")

    # Top features JSON (used by frontend directly)
    top_feat_path = MODELS_DIR / "xgb_top_features.json"
    with open(top_feat_path, "w") as f:
        json.dump(shap_out["top_features"], f, indent=2)
    logger.info(f"  Top features saved →  {top_feat_path}")

    # Results CSV
    results_path = MODELS_DIR / "xgb_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"  Results saved  →  {results_path}")

    # Feature names list
    feat_names_path = MODELS_DIR / "feature_names.json"
    with open(feat_names_path, "w") as f:
        json.dump(feat_names, f)
    logger.info(f"  Feature names saved → {feat_names_path}")

    logger.info("\nXGBoost training complete.")


if __name__ == "__main__":
    main()
