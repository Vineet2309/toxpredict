"""
config.py
─────────
Single source of truth for paths, constants, and model settings.
Every other module imports from here — never hardcode paths elsewhere.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SRC_DIR    = ROOT_DIR / "src"

# ── Dataset ───────────────────────────────────────────────────────────────────
TOX21_CSV  = DATA_DIR / "tox21.csv"

# ── All 12 Tox21 assay targets ───────────────────────────────────────────────
TOX21_TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

# ── Human-readable assay info (used in API responses + frontend cards) ────────
ASSAY_INFO = {
    "NR-AR":        {"name": "Androgen Receptor",            "category": "Nuclear Receptor"},
    "NR-AR-LBD":    {"name": "Androgen Receptor LBD",        "category": "Nuclear Receptor"},
    "NR-AhR":       {"name": "Aryl Hydrocarbon Receptor",    "category": "Nuclear Receptor"},
    "NR-Aromatase": {"name": "Aromatase Enzyme",             "category": "Nuclear Receptor"},
    "NR-ER":        {"name": "Estrogen Receptor",            "category": "Nuclear Receptor"},
    "NR-ER-LBD":    {"name": "Estrogen Receptor LBD",        "category": "Nuclear Receptor"},
    "NR-PPAR-gamma":{"name": "PPAR-gamma Receptor",          "category": "Nuclear Receptor"},
    "SR-ARE":       {"name": "Antioxidant Response Element", "category": "Stress Response"},
    "SR-ATAD5":     {"name": "DNA Damage / Genotoxicity",    "category": "Stress Response"},
    "SR-HSE":       {"name": "Heat Shock Response",          "category": "Stress Response"},
    "SR-MMP":       {"name": "Mitochondrial Membrane",       "category": "Stress Response"},
    "SR-p53":       {"name": "p53 Tumor Suppressor",         "category": "Stress Response"},
}

# ── Feature engineering ───────────────────────────────────────────────────────
MORGAN_RADIUS      = 2          # ECFP4 equivalent
MORGAN_N_BITS      = 2048       # fingerprint vector length
RANDOM_STATE       = 42

# ── Train / val / test split ratios ──────────────────────────────────────────
TEST_SIZE          = 0.10       # 10% test
VAL_SIZE           = 0.10       # 10% validation
SCAFFOLD_SPLIT     = True       # Bemis-Murcko scaffold split (recommended)

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "auc",
    "tree_method":      "hist",   # fast on CPU
    "random_state":     RANDOM_STATE,
}

# ── GNN hyperparameters ───────────────────────────────────────────────────────
GNN_PARAMS = {
    "in_channels":      44,       # atom feature dim (defined in features.py)
    "hidden_channels":  200,
    "out_channels":     len(TOX21_TASKS),
    "edge_dim":         11,       # bond feature dim
    "num_layers":       6,
    "num_timesteps":    2,
    "dropout":          0.2,
}

GNN_TRAIN_PARAMS = {
    "epochs":           100,
    "batch_size":       64,
    "lr":               1e-3,
    "weight_decay":     1e-5,
}

# ── Saved model filenames ─────────────────────────────────────────────────────
XGB_MODEL_PATH = MODELS_DIR / "xgb_multitask.joblib"
GNN_MODEL_PATH = MODELS_DIR / "attentivefp.pt"

# ── API settings ──────────────────────────────────────────────────────────────
API_HOST  = "0.0.0.0"
API_PORT  = 8000
API_TITLE = "ToxPredict API"
API_VERSION = "1.0.0"

# Risk threshold for classifying a compound as toxic
TOXICITY_THRESHOLD = 0.5

# SHAP: how many top features to return
SHAP_TOP_N = 15
