"""
schemas.py
──────────
Pydantic models for all FastAPI request and response bodies.
Strict typing ensures the frontend always gets predictable JSON shapes.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single-compound prediction request."""
    smiles: str = Field(
        ...,
        description="SMILES string of the compound to predict",
        examples=["CC(=O)Oc1ccccc1C(=O)O"],
    )
    model: str = Field(
        default="gnn",
        description="Which model to use: 'gnn' or 'xgb'",
    )

    @field_validator("smiles")
    @classmethod
    def smiles_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("SMILES string cannot be empty")
        return v

    @field_validator("model")
    @classmethod
    def model_must_be_valid(cls, v: str) -> str:
        if v not in ("gnn", "xgb"):
            raise ValueError("model must be 'gnn' or 'xgb'")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class AssayResult(BaseModel):
    """Prediction result for one assay."""
    task:        str   = Field(..., description="Assay code e.g. NR-AR")
    name:        str   = Field(..., description="Full assay name")
    category:    str   = Field(..., description="Nuclear Receptor or Stress Response")
    probability: float = Field(..., description="Toxicity probability 0–1")
    toxic:       bool  = Field(..., description="True if probability >= threshold")
    risk_level:  str   = Field(..., description="low / medium / high")


class ShapFeature(BaseModel):
    """One SHAP feature importance entry."""
    feature:    str   = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value (positive = toxic)")


class AtomAttention(BaseModel):
    """Per-atom attention weight from GNN."""
    atom_idx: int   = Field(..., description="Atom index in molecule")
    weight:   float = Field(..., description="Normalised attention weight 0–1")


class PredictResponse(BaseModel):
    """Full prediction response for one compound."""
    smiles:           str              = Field(..., description="Input SMILES")
    canonical_smiles: Optional[str]    = Field(None, description="RDKit canonical form")
    model_used:       str              = Field(..., description="gnn or xgb")
    assay_results:    List[AssayResult]= Field(..., description="Results for all 12 assays")
    n_toxic:          int              = Field(..., description="Number of assays predicted toxic")
    overall_risk:     str              = Field(..., description="low / medium / high")
    shap_features:    Optional[Dict[str, List[ShapFeature]]] = Field(
        None, description="Top SHAP features per assay (XGBoost only)"
    )
    atom_attention:   Optional[List[AtomAttention]] = Field(
        None, description="Per-atom attention weights (GNN only)"
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction for multiple SMILES."""
    smiles_list: List[str] = Field(
        ...,
        description="List of SMILES strings",
        min_length=1,
        max_length=500,
    )
    model: str = Field(default="gnn")


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    total:    int                   = Field(..., description="Total compounds submitted")
    valid:    int                   = Field(..., description="Successfully predicted")
    invalid:  int                   = Field(..., description="Failed to parse")
    results:  List[PredictResponse] = Field(..., description="Per-compound results")
    failed:   List[str]             = Field(..., description="SMILES that failed")


class AssayInfo(BaseModel):
    """Metadata for one assay target."""
    code:        str = Field(..., description="Assay code e.g. NR-AR")
    name:        str = Field(..., description="Full human-readable name")
    category:    str = Field(..., description="Nuclear Receptor or Stress Response")
    description: str = Field(..., description="What this assay measures")


class HealthResponse(BaseModel):
    """API health check response."""
    status:     str  = Field(..., description="ok or degraded")
    xgb_loaded: bool = Field(..., description="XGBoost model loaded")
    gnn_loaded: bool = Field(..., description="GNN model loaded")
    n_tasks:    int  = Field(..., description="Number of assay tasks")
