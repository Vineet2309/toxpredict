"""
main.py
───────
FastAPI application for ToxPredict.
Designed for Render.com free tier:
  - Port binds IMMEDIATELY (models load in background thread)
  - GNN is optional — works fine with XGBoost only
  - No torch/PyG import at module level (avoids slow import on startup)

Start with:  python run.py
"""

import sys
import json
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# ── Resolve paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))
sys.path.append(str(_ROOT / "src"))

from config import (
    TOX21_TASKS,
    ASSAY_INFO,
    XGB_MODEL_PATH,
    GNN_MODEL_PATH,
    MODELS_DIR,
    TOXICITY_THRESHOLD,
    API_TITLE,
    API_VERSION,
)
from api.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    AssayInfo,
    AssayResult,
    ShapFeature,
    AtomAttention,
    HealthResponse,
)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    xgb_models:   Optional[Dict] = None
    gnn_model:    Optional[object] = None
    feat_names:   Optional[List] = None
    top_features: Optional[Dict] = None
    device:       Optional[object] = None
    loading:      bool = False
    ready:        bool = False

registry = ModelRegistry()


def _load_xgb():
    """Load XGBoost models from disk."""
    try:
        import joblib
        registry.xgb_models = joblib.load(XGB_MODEL_PATH)
        logger.info(f"XGBoost loaded — {len(registry.xgb_models)} tasks")

        feat_path = MODELS_DIR / "feature_names.json"
        if feat_path.exists():
            with open(feat_path) as f:
                registry.feat_names = json.load(f)
            logger.info(f"Feature names loaded — {len(registry.feat_names)} features")

        top_path = MODELS_DIR / "xgb_top_features.json"
        if top_path.exists():
            with open(top_path) as f:
                registry.top_features = json.load(f)
            logger.info("SHAP top features loaded")

    except FileNotFoundError:
        logger.warning(f"XGBoost model not found at {XGB_MODEL_PATH}")
    except Exception as e:
        logger.error(f"XGBoost load error: {e}")


def _load_gnn():
    """Load GNN model — optional, skipped gracefully if not available."""
    if not GNN_MODEL_PATH.exists():
        logger.info("GNN model not found — skipping (XGBoost only mode)")
        return
    try:
        import torch
        from train_gnn import ToxAttentiveFP
        from features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

        registry.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        checkpoint = torch.load(
            GNN_MODEL_PATH,
            map_location=registry.device,
            weights_only=False,
        )
        params = checkpoint.get("gnn_params", {})
        params.update({"in_channels": ATOM_FEATURE_DIM, "edge_dim": BOND_FEATURE_DIM})

        model = ToxAttentiveFP(**params).to(registry.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        registry.gnn_model = model
        logger.info(f"GNN loaded — epoch {checkpoint.get('epoch','?')} on {registry.device}")

    except Exception as e:
        logger.warning(f"GNN load skipped: {e}")


def _load_all_models():
    """Called in background thread — loads everything then marks ready."""
    registry.loading = True
    logger.info("Background thread: loading models...")
    _load_xgb()
    _load_gnn()
    registry.loading = False
    registry.ready   = True
    logger.info("All models ready.")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN  — port binds BEFORE models finish loading
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Yield immediately so uvicorn binds the port right away.
    Models load in a daemon background thread.

    This is critical for Render.com — it scans for an open port
    within seconds of process start. If we block here (loading models),
    Render times out and kills the process.
    """
    logger.info("ToxPredict API — starting, port will bind now...")
    t = threading.Thread(target=_load_all_models, daemon=True)
    t.start()
    logger.info("Background model loading started. Port is open.")
    yield
    logger.info("ToxPredict API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = API_TITLE,
    version     = API_VERSION,
    description = "Drug toxicity prediction across 12 Tox21 assay targets.",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob >= 0.65: return "high"
    if prob >= 0.35: return "medium"
    return "low"

def _overall_risk(n_toxic: int, n_tasks: int) -> str:
    ratio = n_toxic / max(n_tasks, 1)
    if ratio >= 0.4:  return "high"
    if ratio >= 0.15: return "medium"
    return "low"

def _canonical(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None

def _build_assay_results(predictions: Dict) -> List[AssayResult]:
    results = []
    for task in TOX21_TASKS:
        if task not in predictions:
            continue
        pred = predictions[task]
        info = ASSAY_INFO.get(task, {"name": task, "category": "Unknown"})
        prob = pred["probability"]
        results.append(AssayResult(
            task        = task,
            name        = info["name"],
            category    = info["category"],
            probability = prob,
            toxic       = pred["toxic"],
            risk_level  = _risk_level(prob),
        ))
    return results

def _check_models_ready(model_type: str = "xgb"):
    """Raise 503 with helpful message if models aren't loaded yet."""
    if registry.loading and not registry.ready:
        raise HTTPException(
            status_code = 503,
            detail      = "Models are still loading (usually takes 20-30s after startup). Please retry shortly."
        )
    if model_type == "xgb" and registry.xgb_models is None:
        raise HTTPException(503, "XGBoost model not loaded. Check that models/xgb_multitask.joblib exists.")
    if model_type == "gnn" and registry.gnn_model is None:
        raise HTTPException(503, "GNN model not loaded. Train it first with src/train_gnn.py or switch to XGBoost.")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Meta
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Meta"])
def root():
    """Root — confirms service is alive."""
    return {
        "service": "ToxPredict API",
        "status":  "ok" if registry.ready else "loading",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Health check — shows which models are loaded."""
    return HealthResponse(
        status     = "ok" if registry.ready else "loading",
        xgb_loaded = registry.xgb_models is not None,
        gnn_loaded = registry.gnn_model  is not None,
        n_tasks    = len(TOX21_TASKS),
    )


@app.get("/assays", response_model=List[AssayInfo], tags=["Meta"])
def get_assays():
    """Metadata for all 12 Tox21 assay targets."""
    descriptions = {
        "NR-AR":         "Androgen receptor activation — hormone-sensitive cancers",
        "NR-AR-LBD":     "Androgen receptor ligand-binding domain",
        "NR-AhR":        "Aryl hydrocarbon receptor — dioxin-like toxicity",
        "NR-Aromatase":  "Aromatase enzyme inhibition — estrogen synthesis",
        "NR-ER":         "Estrogen receptor activation — breast cancer risk",
        "NR-ER-LBD":     "Estrogen receptor ligand-binding domain",
        "NR-PPAR-gamma": "PPAR-gamma receptor — metabolic effects",
        "SR-ARE":        "Antioxidant response element — oxidative stress",
        "SR-ATAD5":      "DNA damage / genotoxicity",
        "SR-HSE":        "Heat shock response — protein stress",
        "SR-MMP":        "Mitochondrial membrane potential disruption",
        "SR-p53":        "p53 tumour suppressor — DNA damage / cancer",
    }
    return [
        AssayInfo(
            code        = task,
            name        = ASSAY_INFO[task]["name"],
            category    = ASSAY_INFO[task]["category"],
            description = descriptions.get(task, ""),
        )
        for task in TOX21_TASKS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """Predict toxicity for a single SMILES string."""
    _check_models_ready(req.model)

    smiles = req.smiles.strip()
    from rdkit import Chem
    if Chem.MolFromSmiles(smiles) is None:
        raise HTTPException(422, f"Invalid SMILES: '{smiles}'")

    try:
        if req.model == "gnn":
            return _predict_gnn(smiles)
        else:
            return _predict_xgb(smiles)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"Prediction error for '{smiles}': {e}")
        raise HTTPException(500, f"Prediction failed: {e}")


def _predict_xgb(smiles: str) -> PredictResponse:
    from train_xgb import predict_single
    result        = predict_single(smiles, registry.xgb_models, registry.feat_names)
    assay_results = _build_assay_results(result["predictions"])
    n_toxic       = sum(1 for r in assay_results if r.toxic)

    shap_feats: Dict[str, List[ShapFeature]] = {}
    for task, feats in result.get("shap_top", {}).items():
        shap_feats[task] = [ShapFeature(feature=f["feature"], shap_value=f["shap_value"]) for f in feats]

    return PredictResponse(
        smiles           = smiles,
        canonical_smiles = _canonical(smiles),
        model_used       = "xgb",
        assay_results    = assay_results,
        n_toxic          = n_toxic,
        overall_risk     = _overall_risk(n_toxic, len(assay_results)),
        shap_features    = shap_feats or None,
        atom_attention   = None,
    )


def _predict_gnn(smiles: str) -> PredictResponse:
    from train_gnn import predict_single_gnn
    result        = predict_single_gnn(smiles, registry.gnn_model, registry.device)
    assay_results = _build_assay_results(result["predictions"])
    n_toxic       = sum(1 for r in assay_results if r.toxic)
    atom_attn     = [AtomAttention(atom_idx=i, weight=round(w, 4))
                     for i, w in enumerate(result.get("atom_attention", []))]
    return PredictResponse(
        smiles           = smiles,
        canonical_smiles = _canonical(smiles),
        model_used       = "gnn",
        assay_results    = assay_results,
        n_toxic          = n_toxic,
        overall_risk     = _overall_risk(n_toxic, len(assay_results)),
        shap_features    = None,
        atom_attention   = atom_attn or None,
    )


@app.post("/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def batch_predict(req: BatchPredictRequest):
    """Batch predict for a list of SMILES (max 500)."""
    results, failed = [], []
    for smiles in req.smiles_list:
        try:
            results.append(predict(PredictRequest(smiles=smiles.strip(), model=req.model)))
        except Exception:
            failed.append(smiles)
    return BatchPredictResponse(
        total=len(req.smiles_list), valid=len(results),
        invalid=len(failed), results=results, failed=failed,
    )


@app.post("/upload-batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def upload_batch(file: UploadFile = File(...), model: str = "xgb"):
    """Upload CSV with a 'smiles' column for batch prediction."""
    import pandas as pd, io
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    smiles_col = next(
        (c for c in ["smiles","SMILES","Smiles","canonical_smiles"] if c in df.columns), None
    )
    if smiles_col is None:
        raise HTTPException(400, "CSV must have a column named 'smiles'")

    smiles_list = df[smiles_col].dropna().astype(str).tolist()
    if len(smiles_list) > 500:
        raise HTTPException(400, "Maximum 500 compounds per batch")

    return batch_predict(BatchPredictRequest(smiles_list=smiles_list, model=model))


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — Visualization
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/viz/shap-summary", tags=["Visualization"])
def shap_summary():
    if registry.top_features is None:
        raise HTTPException(503, "SHAP summary not available")
    return registry.top_features


@app.post("/viz/shap-compound", tags=["Visualization"])
def shap_compound(req: PredictRequest):
    _check_models_ready("xgb")
    from rdkit import Chem
    smiles = req.smiles.strip()
    if Chem.MolFromSmiles(smiles) is None:
        raise HTTPException(422, f"Invalid SMILES: '{smiles}'")
    try:
        from train_xgb import predict_single
        result = predict_single(smiles, registry.xgb_models, registry.feat_names)
        return {"smiles": smiles, "shap_by_task": result.get("shap_top", {})}
    except Exception as e:
        raise HTTPException(500, f"SHAP failed: {e}")


@app.get("/viz/training-curve", tags=["Visualization"])
def training_curve():
    log_path = MODELS_DIR / "gnn_training_log.json"
    if not log_path.exists():
        raise HTTPException(503, "Training log not found")
    with open(log_path) as f:
        log = json.load(f)
    return {
        "epochs":       [e["epoch"]        for e in log],
        "train_loss":   [e["train_loss"]   for e in log],
        "val_loss":     [e["val_loss"]     for e in log],
        "mean_val_auc": [e["mean_val_auc"] for e in log],
    }


@app.get("/viz/model-results", tags=["Visualization"])
def model_results():
    import pandas as pd
    def _load(path):
        if not path.exists(): return {}
        df = pd.read_csv(path)
        return {
            row["task"]: {
                "roc_auc": None if pd.isna(row["roc_auc"]) else float(row["roc_auc"]),
                "auprc":   None if pd.isna(row["auprc"])   else float(row["auprc"]),
                "f1":      None if pd.isna(row["f1"])       else float(row["f1"]),
            }
            for _, row in df.iterrows()
        }
    xgb_r = _load(MODELS_DIR / "xgb_results.csv")
    gnn_r = _load(MODELS_DIR / "gnn_results.csv")
    if not xgb_r and not gnn_r:
        raise HTTPException(503, "No results found")
    return {"tasks": TOX21_TASKS, "xgb": xgb_r, "gnn": gnn_r}


@app.get("/viz/dataset-stats", tags=["Visualization"])
def dataset_stats():
    import pandas as pd
    from config import TOX21_CSV
    if not TOX21_CSV.exists():
        raise HTTPException(503, "tox21.csv not found")
    df = pd.read_csv(TOX21_CSV)
    stats = []
    for task in TOX21_TASKS:
        if task not in df.columns: continue
        col = df[task]
        n_valid = col.notna().sum()
        n_pos   = int((col == 1).sum())
        n_neg   = int((col == 0).sum())
        stats.append({
            "task":          task,
            "name":          ASSAY_INFO[task]["name"],
            "category":      ASSAY_INFO[task]["category"],
            "n_toxic":       n_pos,
            "n_non_toxic":   n_neg,
            "n_missing":     int(col.isna().sum()),
            "positive_rate": round(n_pos / n_valid, 4) if n_valid > 0 else 0,
        })
    return {"tasks": TOX21_TASKS, "stats": stats}
