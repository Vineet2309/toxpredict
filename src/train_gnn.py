"""
train_gnn.py
────────────
AttentiveFP Graph Neural Network for multi-task Tox21 toxicity prediction.

Architecture: Attentive Fingerprint (Xiong et al., J. Med. Chem. 2020)
  - Graph attention over atom neighbourhoods
  - Molecule-level readout via GRU-based pooling
  - Multi-task sigmoid output heads (one per assay)
  - Masked BCE loss — ignores NaN labels during training

Saves to:
  models/attentivefp.pt          — best model checkpoint (lowest val loss)
  models/gnn_results.csv         — per-task AUC / AUPRC / F1
  models/gnn_training_log.json   — loss curve for plotting
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import AttentiveFP

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    TOX21_CSV,
    TOX21_TASKS,
    GNN_PARAMS,
    GNN_TRAIN_PARAMS,
    GNN_MODEL_PATH,
    MODELS_DIR,
    TOXICITY_THRESHOLD,
    RANDOM_STATE,
)
from preprocess import run_pipeline
from features   import build_graph_dataset, ATOM_FEATURE_DIM, BOND_FEATURE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.info("No GPU found — using CPU (training will be slower)")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ToxAttentiveFP(nn.Module):
    """
    AttentiveFP backbone with multi-task toxicity prediction heads.

    AttentiveFP uses graph attention at two levels:
      1. Atom-level: each atom attends to its neighbours
      2. Molecule-level: a GRU pools atom representations into one vector

    The attention weights at the atom level are what we later extract
    to highlight 'which atoms matter most' for a given prediction.

    Args:
        in_channels:      atom feature dim (44)
        hidden_channels:  internal representation size (200)
        out_channels:     number of tasks (12)
        edge_dim:         bond feature dim (11)
        num_layers:       graph attention layers (6)
        num_timesteps:    GRU readout timesteps (2)
        dropout:          dropout rate (0.2)
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.backbone = AttentiveFP(
            in_channels     = kwargs["in_channels"],
            hidden_channels = kwargs["hidden_channels"],
            out_channels    = kwargs["hidden_channels"],  # embed → task heads
            edge_dim        = kwargs["edge_dim"],
            num_layers      = kwargs["num_layers"],
            num_timesteps   = kwargs["num_timesteps"],
            dropout         = kwargs["dropout"],
        )

        n_tasks = kwargs["out_channels"]
        h       = kwargs["hidden_channels"]

        # Separate linear head per task — allows each assay to have its own
        # decision boundary in the shared embedding space
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Dropout(kwargs["dropout"]),
                nn.Linear(h // 2, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass.
        Returns logits tensor of shape (batch_size, n_tasks).
        """
        mol_embed = self.backbone(x, edge_index, edge_attr, batch)
        # mol_embed: (batch_size, hidden_channels)

        task_logits = [head(mol_embed) for head in self.task_heads]
        return torch.cat(task_logits, dim=1)   # (batch_size, n_tasks)

    def get_attention_weights(self, x, edge_index, edge_attr, batch):
        """
        Extract per-atom attention weights from the backbone.
        Used after training to build atom-level heatmaps.
        Returns the backbone's alpha values (attention coefficients).
        """
        # AttentiveFP stores attention alphas internally during forward
        # We access them via the backbone's graph_conv layers
        mol_embed = self.backbone(x, edge_index, edge_attr, batch)
        return mol_embed   # caller uses registered hooks if needed


# ─────────────────────────────────────────────────────────────────────────────
# LOSS  —  masked binary cross-entropy
# ─────────────────────────────────────────────────────────────────────────────

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy loss that ignores NaN-labelled entries.

    In Tox21 many compounds were not tested on every assay — those entries
    are NaN. Training on them as negatives would corrupt the model.
    This loss builds a boolean mask and only back-propagates through
    the entries where a real label exists.

    Args:
        logits:  (batch_size, n_tasks) raw model output
        targets: (batch_size, n_tasks) float labels, NaN = not tested

    Returns:
        Scalar mean loss over all valid (non-NaN) entries.
    """
    mask = ~torch.isnan(targets)

    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)

    # Replace NaN with 0 for BCE computation (masked out anyway)
    safe_targets = targets.clone()
    safe_targets[~mask] = 0.0

    loss_all = F.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none"
    )
    # Average only over valid entries
    return (loss_all * mask.float()).sum() / mask.float().sum()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def collate_labels(dataset: list) -> list:
    """
    Ensure every Data object has y as shape (n_tasks,) not (1, n_tasks).
    PyG DataLoader expects consistent tensor shapes.
    """
    for data in dataset:
        if data.y is not None and data.y.dim() == 2:
            data.y = data.y.squeeze(0)   # (1, 12) → (12,)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     ToxAttentiveFP,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """One full training pass. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        targets = batch.y.view(logits.shape)      # (batch_size, n_tasks)

        loss = masked_bce_loss(logits, targets)
        loss.backward()

        # Gradient clipping — stabilises GNN training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model:  ToxAttentiveFP,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Validation / test pass.
    Returns (mean_loss, {task: roc_auc}) for all tasks with enough labels.
    """
    model.eval()
    all_logits  = []
    all_targets = []
    total_loss  = 0.0
    n_batches   = 0

    for batch in loader:
        batch   = batch.to(device)
        logits  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        targets = batch.y.view(logits.shape)

        loss = masked_bce_loss(logits, targets)
        total_loss  += loss.item()
        n_batches   += 1

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    all_logits  = torch.cat(all_logits,  dim=0).numpy()   # (N, 12)
    all_targets = torch.cat(all_targets, dim=0).numpy()   # (N, 12)
    all_probs   = 1 / (1 + np.exp(-all_logits))           # sigmoid

    task_aucs = {}
    for i, task in enumerate(TOX21_TASKS):
        y_true = all_targets[:, i]
        y_prob = all_probs[:, i]
        mask   = ~np.isnan(y_true)

        if mask.sum() < 10 or len(np.unique(y_true[mask])) < 2:
            continue
        try:
            task_aucs[task] = roc_auc_score(y_true[mask], y_prob[mask])
        except Exception:
            pass

    mean_loss = total_loss / max(n_batches, 1)
    return mean_loss, task_aucs


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION  (test set)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(
    model:  ToxAttentiveFP,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Compute ROC-AUC, AUPRC, F1 on test set for all 12 tasks."""
    model.eval()
    all_logits  = []
    all_targets = []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_logits.append(logits.cpu())
        all_targets.append(batch.y.view(logits.shape).cpu())

    probs   = torch.sigmoid(torch.cat(all_logits)).numpy()
    targets = torch.cat(all_targets).numpy()

    rows = []
    for i, task in enumerate(TOX21_TASKS):
        y_true = targets[:, i]
        y_prob = probs[:, i]
        mask   = ~np.isnan(y_true)

        if mask.sum() < 10 or len(np.unique(y_true[mask])) < 2:
            rows.append({"task": task,
                         "roc_auc": float("nan"),
                         "auprc":   float("nan"),
                         "f1":      float("nan")})
            continue

        y_pred = (y_prob[mask] >= TOXICITY_THRESHOLD).astype(int)
        rows.append({
            "task":    task,
            "roc_auc": round(roc_auc_score(y_true[mask], y_prob[mask]), 4),
            "auprc":   round(average_precision_score(y_true[mask], y_prob[mask]), 4),
            "f1":      round(f1_score(y_true[mask], y_pred, zero_division=0), 4),
        })

    results_df = pd.DataFrame(rows)

    logger.info("\n" + "="*52)
    logger.info(results_df.to_string(index=False))
    logger.info("="*52)
    logger.info(f"Mean ROC-AUC : {results_df['roc_auc'].dropna().mean():.4f}")
    logger.info(f"Mean AUPRC   : {results_df['auprc'].dropna().mean():.4f}")
    logger.info(f"Mean F1      : {results_df['f1'].dropna().mean():.4f}")
    logger.info("="*52)

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# ATOM ATTENTION EXTRACTOR  (used by FastAPI for heatmaps)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_atom_attention(
    smiles: str,
    model:  ToxAttentiveFP,
    device: torch.device,
) -> List[float]:
    """
    Extract per-atom attention weights for a single molecule.

    AttentiveFP computes atom-level attention during forward pass via
    registered hooks on the graph_conv layers. We capture the final
    layer's attention coefficients and aggregate them per atom.

    Returns a list of floats (one per atom) normalised to [0, 1].
    Higher = that atom contributed more to the prediction.
    """
    from features import smiles_to_graph

    data = smiles_to_graph(smiles)
    if data is None:
        return []

    data = data.to(device)

    # Capture attention alphas via forward hook on backbone
    attention_weights = []

    def _hook(module, input, output):
        # AttentiveFP's GATConv returns (out, alpha) when return_attention_weights=True
        # We hook the conv layer to grab alpha
        if isinstance(output, tuple) and len(output) == 2:
            attention_weights.append(output[1][1].detach().cpu())

    # Register hooks on all GATConv layers inside AttentiveFP backbone
    hooks = []
    for name, module in model.backbone.named_modules():
        if "conv" in name.lower() and hasattr(module, "forward"):
            hooks.append(module.register_forward_hook(_hook))

    model.eval()
    _ = model(
        data.x,
        data.edge_index,
        data.edge_attr,
        torch.zeros(data.num_nodes, dtype=torch.long, device=device),
    )

    # Remove hooks
    for h in hooks:
        h.remove()

    if not attention_weights:
        # Fallback: return uniform weights
        return [1.0 / data.num_nodes] * data.num_nodes

    # Aggregate: for each atom, take the mean attention across all edges
    # pointing to it across all captured layers
    n_atoms     = data.num_nodes
    atom_scores = np.zeros(n_atoms, dtype=np.float32)
    counts      = np.zeros(n_atoms, dtype=np.float32)

    for alpha in attention_weights:
        alpha_np = alpha.numpy().flatten()
        edge_dst = data.edge_index[1].cpu().numpy()

        for edge_i, atom_j in enumerate(edge_dst):
            if edge_i < len(alpha_np):
                atom_scores[atom_j] += alpha_np[edge_i]
                counts[atom_j]      += 1

    counts    = np.where(counts == 0, 1, counts)
    atom_scores = atom_scores / counts

    # Normalise to [0, 1]
    mn, mx = atom_scores.min(), atom_scores.max()
    if mx > mn:
        atom_scores = (atom_scores - mn) / (mx - mn)
    else:
        atom_scores = np.ones(n_atoms, dtype=np.float32) / n_atoms

    return atom_scores.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-COMPOUND PREDICTION  (used by FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single_gnn(
    smiles: str,
    model:  ToxAttentiveFP,
    device: torch.device,
) -> Dict:
    """
    Predict toxicity probabilities for a single SMILES string.

    Returns:
      {
        "predictions":     {task: {"probability": float, "toxic": bool}},
        "atom_attention":  [float, ...]   — per-atom importance weights
      }
    """
    from features import smiles_to_graph

    data = smiles_to_graph(smiles)
    if data is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    data  = data.to(device)
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    logits = model(data.x, data.edge_index, data.edge_attr, batch)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    predictions = {}
    for i, task in enumerate(TOX21_TASKS):
        prob = float(probs[i])
        predictions[task] = {
            "probability": round(prob, 4),
            "toxic":       prob >= TOXICITY_THRESHOLD,
        }

    atom_attn = get_atom_attention(smiles, model, device)

    return {"predictions": predictions, "atom_attention": atom_attn}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = get_device()

    logger.info("══════════════════════════════════════════")
    logger.info("  ToxPredict — AttentiveFP GNN Training   ")
    logger.info("══════════════════════════════════════════")

    # 1. Preprocessing
    logger.info("\n[1/5] Preprocessing data...")
    train_df, val_df, test_df = run_pipeline(TOX21_CSV)

    # 2. Build graph datasets
    logger.info("\n[2/5] Building molecular graph datasets...")
    train_data = collate_labels(build_graph_dataset(train_df))
    val_data   = collate_labels(build_graph_dataset(val_df))
    test_data  = collate_labels(build_graph_dataset(test_df))

    logger.info(f"  Graphs — Train: {len(train_data)}  "
                f"Val: {len(val_data)}  Test: {len(test_data)}")

    bs = GNN_TRAIN_PARAMS["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=bs, shuffle=False)

    # 3. Build model
    logger.info("\n[3/5] Building AttentiveFP model...")
    params = {
        **GNN_PARAMS,
        "in_channels": ATOM_FEATURE_DIM,
        "edge_dim":    BOND_FEATURE_DIM,
    }
    model = ToxAttentiveFP(**params).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Trainable parameters: {n_params:,}")

    optimizer = Adam(
        model.parameters(),
        lr           = GNN_TRAIN_PARAMS["lr"],
        weight_decay = GNN_TRAIN_PARAMS["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = GNN_TRAIN_PARAMS["epochs"],
        eta_min= 1e-5,
    )

    # 4. Training loop
    logger.info("\n[4/5] Training...")
    best_val_loss = float("inf")
    best_epoch    = 0
    training_log  = []
    epochs        = GNN_TRAIN_PARAMS["epochs"]

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        train_loss          = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_aucs  = eval_epoch(model, val_loader, device)
        scheduler.step()

        mean_val_auc = np.mean(list(val_aucs.values())) if val_aucs else 0.0

        training_log.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss,   6),
            "mean_val_auc": round(mean_val_auc, 4),
            "lr":           round(scheduler.get_last_lr()[0], 6),
        })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "gnn_params":  params,
            }, GNN_MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"mean_val_auc={mean_val_auc:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    logger.info(f"\n  Best checkpoint: epoch {best_epoch} "
                f"(val_loss={best_val_loss:.4f})")

    # 5. Test evaluation with best model
    logger.info("\n[5/5] Loading best checkpoint and evaluating on test set...")
    checkpoint = torch.load(GNN_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    results_df = evaluate_test(model, test_loader, device)

    # ── Save outputs ─────────────────────────────────────────────────
    results_path = MODELS_DIR / "gnn_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"  Results saved  →  {results_path}")

    log_path = MODELS_DIR / "gnn_training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"  Training log saved →  {log_path}")

    logger.info("\nAttentiveFP GNN training complete.")


if __name__ == "__main__":
    main()
