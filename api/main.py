"""
main.py
───────
FastAPI application for ToxPredict.

Routes:
  GET  /health          — health check, confirms models are loaded
  GET  /assays          — metadata for all 12 Tox21 assay targets
  POST /predict         — single SMILES → toxicity prediction
  POST /batch           — list of SMILES → batch predictions
  POST /upload-batch    — CSV file upload → batch predictions

Run with:
  uvicorn api.main:app --reload --port 8000
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

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
# MODEL REGISTRY  —  loaded once at startup, shared across all requests
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Holds all loaded models and feature metadata.
    Loaded once during startup via the lifespan context manager.
    Using a class (instead of globals) keeps things clean and testable.
    """
    xgb_models:   Optional[Dict]  = None   # {task: XGBClassifier}
    gnn_model:    Optional[object]= None   # ToxAttentiveFP
    feat_names:   Optional[List]  = None   # feature name list for SHAP
    top_features: Optional[Dict]  = None   # pre-computed SHAP summary
    device:       Optional[object]= None   # torch device
    gnn_params:   Optional[Dict]  = None   # GNN architecture params


registry = ModelRegistry()


def _load_xgb():
    """Load XGBoost models and feature metadata from disk."""
    try:
        import joblib
        registry.xgb_models = joblib.load(XGB_MODEL_PATH)
        logger.info(f"XGBoost models loaded  ({len(registry.xgb_models)} tasks)")

        feat_path = MODELS_DIR / "feature_names.json"
        if feat_path.exists():
            with open(feat_path) as f:
                registry.feat_names = json.load(f)
            logger.info(f"Feature names loaded  ({len(registry.feat_names)} features)")

        top_path = MODELS_DIR / "xgb_top_features.json"
        if top_path.exists():
            with open(top_path) as f:
                registry.top_features = json.load(f)
            logger.info("Pre-computed SHAP top features loaded")

    except FileNotFoundError:
        logger.warning(f"XGBoost model not found at {XGB_MODEL_PATH} — train first")
    except Exception as e:
        logger.error(f"Failed to load XGBoost: {e}")


def _load_gnn():
    """Load AttentiveFP GNN model from checkpoint."""
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
        params.update({
            "in_channels": ATOM_FEATURE_DIM,
            "edge_dim":    BOND_FEATURE_DIM,
        })
        registry.gnn_params = params

        model = ToxAttentiveFP(**params).to(registry.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        registry.gnn_model = model

        logger.info(f"GNN model loaded  (device={registry.device}, "
                    f"epoch={checkpoint.get('epoch', '?')})")

    except FileNotFoundError:
        logger.warning(f"GNN model not found at {GNN_MODEL_PATH} — train first")
    except Exception as e:
        logger.error(f"Failed to load GNN: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN  —  startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models in a background thread so the port binds immediately.
    Render.com times out if startup blocks the port for > 60s.
    Models will be ready within a few seconds of the first request.
    """
    import threading
    logger.info("ToxPredict API starting — port binding now...")

    def _load_all():
        logger.info("Background: loading models...")
        _load_xgb()
        _load_gnn()
        logger.info("Background: all models ready.")

    t = threading.Thread(target=_load_all, daemon=True)
    t.start()

    logger.info("Port bound. Models loading in background.")
    yield
    logger.info("Shutting down ToxPredict API.")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = API_TITLE,
    version     = API_VERSION,
    description = (
        "Drug toxicity prediction across 12 Tox21 assay targets. "
        "Supports XGBoost (with SHAP explainability) and "
        "AttentiveFP GNN (with atom-level attention heatmaps)."
    ),
    lifespan    = lifespan,
)

# CORS — allows the HTML frontend to call this API from any origin
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
    """Convert probability to a risk tier for the frontend."""
    if prob >= 0.65:
        return "high"
    if prob >= 0.35:
        return "medium"
    return "low"


def _overall_risk(n_toxic: int, n_tasks: int) -> str:
    ratio = n_toxic / max(n_tasks, 1)
    if ratio >= 0.4:
        return "high"
    if ratio >= 0.15:
        return "medium"
    return "low"


def _canonical(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def _build_assay_results(predictions: Dict) -> List[AssayResult]:
    """Convert raw predictions dict → list of AssayResult objects."""
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


def _build_shap_response(
    shap_top: Dict,
) -> Dict[str, List[ShapFeature]]:
    out = {}
    for task, features in shap_top.items():
        out[task] = [
            ShapFeature(
                feature    = f["feature"],
                shap_value = f["shap_value"],
            )
            for f in features
        ]
    return out


def _build_atom_attention(weights: List[float]) -> List[AtomAttention]:
    return [
        AtomAttention(atom_idx=i, weight=round(w, 4))
        for i, w in enumerate(weights)
    ]


def _predict_with_gnn(smiles: str) -> PredictResponse:
    from train_gnn import predict_single_gnn

    result = predict_single_gnn(
        smiles,
        registry.gnn_model,
        registry.device,
    )
    assay_results = _build_assay_results(result["predictions"])
    n_toxic       = sum(1 for r in assay_results if r.toxic)

    return PredictResponse(
        smiles           = smiles,
        canonical_smiles = _canonical(smiles),
        model_used       = "gnn",
        assay_results    = assay_results,
        n_toxic          = n_toxic,
        overall_risk     = _overall_risk(n_toxic, len(assay_results)),
        shap_features    = None,
        atom_attention   = _build_atom_attention(result.get("atom_attention", [])),
    )


def _predict_with_xgb(smiles: str) -> PredictResponse:
    from train_xgb import predict_single

    if registry.xgb_models is None:
        raise HTTPException(503, "XGBoost model not loaded")
    if registry.feat_names is None:
        raise HTTPException(503, "Feature names not loaded")

    result = predict_single(
        smiles,
        registry.xgb_models,
        registry.feat_names,
    )
    assay_results = _build_assay_results(result["predictions"])
    n_toxic       = sum(1 for r in assay_results if r.toxic)

    return PredictResponse(
        smiles           = smiles,
        canonical_smiles = _canonical(smiles),
        model_used       = "xgb",
        assay_results    = assay_results,
        n_toxic          = n_toxic,
        overall_risk     = _overall_risk(n_toxic, len(assay_results)),
        shap_features    = _build_shap_response(result.get("shap_top", {})),
        atom_attention   = None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Check API status and confirm which models are loaded."""
    return HealthResponse(
        status     = "ok" if registry.xgb_models is not None else "loading",
        xgb_loaded = registry.xgb_models is not None,
        gnn_loaded = registry.gnn_model  is not None,
        n_tasks    = len(TOX21_TASKS),
    )


@app.get("/", tags=["Meta"])
def root():
    """Root endpoint — confirms API is alive (used by Render health check)."""
    return {
        "service": "ToxPredict API",
        "status":  "ok",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/assays", response_model=List[AssayInfo], tags=["Meta"])
def get_assays():
    """Return metadata for all 12 Tox21 assay targets."""
    descriptions = {
        "NR-AR":         "Measures activation of the androgen receptor — key in hormone-sensitive cancers",
        "NR-AR-LBD":     "Tests binding to the androgen receptor ligand-binding domain",
        "NR-AhR":        "Aryl hydrocarbon receptor activation — linked to dioxin-like toxicity",
        "NR-Aromatase":  "Inhibition of aromatase enzyme affecting estrogen synthesis",
        "NR-ER":         "Estrogen receptor activation — relevant to breast cancer risk",
        "NR-ER-LBD":     "Binding to estrogen receptor ligand-binding domain",
        "NR-PPAR-gamma": "PPAR-gamma receptor activation — metabolic and developmental effects",
        "SR-ARE":        "Antioxidant response element — indicator of oxidative stress",
        "SR-ATAD5":      "DNA damage / genotoxicity assay via ATAD5 upregulation",
        "SR-HSE":        "Heat shock response — marker of protein misfolding stress",
        "SR-MMP":        "Mitochondrial membrane potential disruption — mitochondrial toxicity",
        "SR-p53":        "p53 tumour suppressor activation — DNA damage and cancer risk",
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


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict toxicity for a single SMILES string.

    - model='gnn'  → AttentiveFP GNN with atom attention heatmap
    - model='xgb'  → XGBoost with SHAP feature importance

    Returns probabilities for all 12 Tox21 assay targets.
    """
    # Guard: return 503 if models are still loading in background
    if registry.xgb_models is None and registry.gnn_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please retry in a few seconds."
        )

    smiles = req.smiles.strip()

    # Validate SMILES before running model
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(
            status_code = 422,
            detail      = f"Invalid SMILES string: '{smiles}'"
        )

    try:
        if req.model == "gnn":
            if registry.gnn_model is None:
                raise HTTPException(503, "GNN model not loaded — train first")
            return _predict_with_gnn(smiles)
        else:
            return _predict_with_xgb(smiles)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error for '{smiles}': {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def batch_predict(req: BatchPredictRequest):
    """
    Predict toxicity for a list of SMILES strings (max 500).
    Returns per-compound results and a list of any that failed.
    """
    results = []
    failed  = []

    for smiles in req.smiles_list:
        smiles = smiles.strip()
        try:
            single_req = PredictRequest(smiles=smiles, model=req.model)
            result     = predict(single_req)
            results.append(result)
        except HTTPException:
            failed.append(smiles)
        except Exception:
            failed.append(smiles)

    return BatchPredictResponse(
        total   = len(req.smiles_list),
        valid   = len(results),
        invalid = len(failed),
        results = results,
        failed  = failed,
    )


@app.post("/upload-batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def upload_batch(
    file:  UploadFile = File(...),
    model: str        = "gnn",
):
    """
    Upload a CSV file with a 'smiles' column for batch prediction.
    Returns predictions for all valid compounds in the file.
    """
    import pandas as pd
    import io

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    smiles_col = None
    for candidate in ["smiles", "SMILES", "Smiles", "canonical_smiles"]:
        if candidate in df.columns:
            smiles_col = candidate
            break

    if smiles_col is None:
        raise HTTPException(
            400,
            "CSV must have a column named 'smiles' (case-insensitive)"
        )

    smiles_list = df[smiles_col].dropna().astype(str).tolist()
    if len(smiles_list) > 500:
        raise HTTPException(400, "Maximum 500 compounds per batch upload")

    req = BatchPredictRequest(smiles_list=smiles_list, model=model)
    return batch_predict(req)



# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  —  VISUALIZATION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/viz/shap-summary", tags=["Visualization"])
def shap_summary():
    """
    Pre-computed SHAP top-N features for all 12 assays.
    Frontend uses this to render the global feature importance bar chart.
    """
    if registry.top_features is None:
        raise HTTPException(503, "SHAP summary not available — run train_xgb.py first")
    return registry.top_features


@app.post("/viz/shap-compound", tags=["Visualization"])
def shap_compound(req: PredictRequest):
    """
    Per-compound SHAP values for a single SMILES using XGBoost.
    Returns top-N influential features per assay with signed SHAP values.
    Frontend uses this to render per-compound SHAP bar charts.
    """
    if registry.xgb_models is None:
        raise HTTPException(503, "XGBoost model not loaded")
    if registry.feat_names is None:
        raise HTTPException(503, "Feature names not loaded")

    from rdkit import Chem
    smiles = req.smiles.strip()
    if Chem.MolFromSmiles(smiles) is None:
        raise HTTPException(422, f"Invalid SMILES: '{smiles}'")

    try:
        from train_xgb import predict_single
        result = predict_single(smiles, registry.xgb_models, registry.feat_names)
        return {"smiles": smiles, "shap_by_task": result.get("shap_top", {})}
    except Exception as e:
        raise HTTPException(500, f"SHAP computation failed: {e}")


@app.get("/viz/training-curve", tags=["Visualization"])
def training_curve():
    """
    GNN training loss curve — epoch vs train_loss vs val_loss vs mean_val_auc.
    Frontend uses this to render the training progress chart.
    """
    log_path = MODELS_DIR / "gnn_training_log.json"
    if not log_path.exists():
        raise HTTPException(503, "Training log not found — run train_gnn.py first")

    with open(log_path) as f:
        log = json.load(f)

    return {
        "epochs":       [e["epoch"]        for e in log],
        "train_loss":   [e["train_loss"]   for e in log],
        "val_loss":     [e["val_loss"]      for e in log],
        "mean_val_auc": [e["mean_val_auc"] for e in log],
    }


@app.get("/viz/model-results", tags=["Visualization"])
def model_results():
    """
    Test-set AUC/AUPRC/F1 for both XGBoost and GNN side by side.
    Frontend uses this for the model comparison table.
    """
    import pandas as pd

    def _load(path):
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        return {
            row["task"]: {
                "roc_auc": float(row["roc_auc"]) if not pd.isna(row["roc_auc"]) else None,
                "auprc":   float(row["auprc"])   if not pd.isna(row["auprc"])   else None,
                "f1":      float(row["f1"])       if not pd.isna(row["f1"])       else None,
            }
            for _, row in df.iterrows()
        }

    xgb_r = _load(MODELS_DIR / "xgb_results.csv")
    gnn_r  = _load(MODELS_DIR / "gnn_results.csv")

    if not xgb_r and not gnn_r:
        raise HTTPException(503, "No results found — run training scripts first")

    return {"tasks": TOX21_TASKS, "xgb": xgb_r, "gnn": gnn_r}


@app.get("/viz/dataset-stats", tags=["Visualization"])
def dataset_stats():
    """
    Class imbalance stats for all 12 assays — toxic vs non-toxic counts.
    Frontend uses this for the EDA bar chart.
    """
    import pandas as pd
    from config import TOX21_CSV

    if not TOX21_CSV.exists():
        raise HTTPException(503, "tox21.csv not found in data/")

    df = pd.read_csv(TOX21_CSV)
    stats = []
    for task in TOX21_TASKS:
        if task not in df.columns:
            continue
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

# ─────────────────────────────────────────────────────────────────────────────
# RUN DIRECTLY  —  python api/main.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT

    uvicorn.run(
        "api.main:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = True,
    )