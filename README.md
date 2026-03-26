# ToxPredict 🧪

> **CodeCure AI Hackathon — Track A: Drug Toxicity Prediction**
> Predicting drug toxicity from molecular structure using Graph Neural Networks, XGBoost, and SHAP explainability.

---

## What it does

ToxPredict takes a **SMILES string** (a text representation of any chemical compound) and predicts its toxicity probability across **12 biological assay targets** from the Tox21 dataset. It tells you *which molecular features* are driving the toxicity risk using SHAP values, and highlights *which atoms* on the molecule are responsible using GNN attention weights.

---

## Tech stack

| Layer | Technology |
|---|---|
| ML — Baseline | XGBoost + Morgan Fingerprints + RDKit descriptors |
| ML — Primary | AttentiveFP GNN (PyTorch Geometric) |
| Explainability | SHAP TreeExplainer + atom attention weights |
| Backend | FastAPI (Python) |
| Frontend | HTML / CSS / JavaScript (single file) |
| Deployment | Render.com (API) + Netlify (frontend) |

---

## Project structure

```
codecure/
├── config.py              ← all paths, constants, hyperparams
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── preprocess.py      ← data loading, SMILES validation, scaffold split
│   ├── features.py        ← Morgan FP + RDKit descriptors + graph builder
│   ├── train_xgb.py       ← XGBoost multi-task training + SHAP
│   └── train_gnn.py       ← AttentiveFP GNN training
│
├── api/
│   ├── main.py            ← FastAPI app, routes
│   └── schemas.py         ← Pydantic request / response models
│
├── frontend/
│   └── index.html         ← complete UI (HTML + CSS + JS, single file)
│
├── data/
│   └── tox21.csv          ← download from Kaggle (not committed)
│
├── models/
│   ├── xgb_multitask.joblib
│   └── attentivefp.pt
│
└── notebooks/
    └── EDA.ipynb
```

---

## Quickstart

### 1 — Clone and install

```bash
git clone https://github.com/your-team/toxpredict.git
cd toxpredict
pip install -r requirements.txt
```

### 2 — Get the dataset

Download `tox21.csv` from [Kaggle](https://www.kaggle.com/datasets/epicskills/tox21-dataset) and place it in `data/`.

### 3 — Train models

```bash
# XGBoost baseline (fast, ~2 min)
python src/train_xgb.py

# AttentiveFP GNN (recommended, ~20 min on GPU)
python src/train_gnn.py
```

### 4 — Start the API

```bash
uvicorn api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### 5 — Open the frontend

```bash
# Just open in any browser
open frontend/index.html

# Or serve locally
python -m http.server 3000 --directory frontend
```

---

## API reference

| Method | Route | Description |
|---|---|---|
| `POST` | `/predict` | Predict toxicity for a single SMILES |
| `POST` | `/batch` | Upload CSV of SMILES, get results |
| `GET` | `/assays` | Get metadata for all 12 assay targets |
| `GET` | `/health` | Health check |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
```

---

## Tox21 assay targets

| Code | Target | Category |
|---|---|---|
| NR-AR | Androgen Receptor | Nuclear Receptor |
| NR-AR-LBD | Androgen Receptor LBD | Nuclear Receptor |
| NR-AhR | Aryl Hydrocarbon Receptor | Nuclear Receptor |
| NR-Aromatase | Aromatase Enzyme | Nuclear Receptor |
| NR-ER | Estrogen Receptor | Nuclear Receptor |
| NR-ER-LBD | Estrogen Receptor LBD | Nuclear Receptor |
| NR-PPAR-gamma | PPAR-gamma Receptor | Nuclear Receptor |
| SR-ARE | Antioxidant Response Element | Stress Response |
| SR-ATAD5 | DNA Damage / Genotoxicity | Stress Response |
| SR-HSE | Heat Shock Response | Stress Response |
| SR-MMP | Mitochondrial Membrane | Stress Response |
| SR-p53 | p53 Tumor Suppressor | Stress Response |

---

## Deliverables checklist

- [x] GitHub repository
- [ ] ML model predicting drug toxicity (Step 4 + 5)
- [ ] Feature importance / SHAP analysis (Step 4 + 7)
- [ ] Visualizations — molecular property plots (Step 7)
- [ ] Prediction interface for new compounds (Step 8 + 9)

---

## References

- Tox21 Dataset — [EPA/NIH Tox21 Program](https://tox21.gov/)
- AttentiveFP — Xiong et al. (2020) *J. Med. Chem.* [DOI:10.1021/acs.jmedchem.9b00959](https://doi.org/10.1021/acs.jmedchem.9b00959)
- SHAP — Lundberg & Lee (2017) *NeurIPS*
- RDKit — [rdkit.org](https://www.rdkit.org/)
- PyTorch Geometric — [pyg.org](https://pyg.org/)

---

*Built for CodeCure AI Hackathon — Track A*
