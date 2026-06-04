"""
Layer 5 — CNN Surrogate Model for Saccharomyces boulardii CNCM I-745
Generates FBA training data, trains a 1D-CNN growth-rate predictor,
evaluates it, and validates against fresh FBA runs.
"""

import io
import json
import csv
import random
import math
from pathlib import Path
from contextlib import redirect_stderr

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import cobra
from cobra.io import read_sbml_model

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path("/home/nsdeshmukh306/digital-twin")
GEM_IN       = BASE / "data/gem/cncm_i745_regulated.xml"
CSV_TRAIN    = BASE / "data/ml_datasets/fba_training_data.csv"
MODEL_PT     = BASE / "data/ml_datasets/surrogate_model.pt"
SCALER_JSON  = BASE / "data/ml_datasets/surrogate_scaler.json"
REPORT_TXT   = BASE / "logs/layer5_report.txt"

SEP = "=" * 60
log_lines: list[str] = []

def log(msg: str = "") -> None:
    print(msg)
    log_lines.append(msg)

def section(title: str) -> None:
    log("")
    log(SEP)
    log(f"STEP {title}")
    log(SEP)

def load_silent(path) -> cobra.Model:
    buf = io.StringIO()
    with redirect_stderr(buf):
        return read_sbml_model(str(path))

# ── Exchange reaction IDs (confirmed in Layer 2/3) ────────────────────────────
RXN_GLUCOSE = "r_1714"   # D-glucose exchange
RXN_OXYGEN  = "r_1992"   # oxygen exchange

# ═════════════════════════════════════════════════════════════════════════════
# 1. GENERATE FBA TRAINING DATASET
# ═════════════════════════════════════════════════════════════════════════════
section("1 — GENERATE FBA TRAINING DATASET")

log(f"  Loading GEM: {GEM_IN}")
base_model = load_silent(GEM_IN)

GLUCOSE_VALS  = [-0.5, -1.0, -2.0, -5.0, -10.0]
OXYGEN_VALS   = [-0.5, -1.0, -2.0, -5.0, -10.0, -20.0]
PH_FACTORS    = [0.5, 0.7, 0.9, 1.0, 1.1]
N_TOTAL       = len(GLUCOSE_VALS) * len(OXYGEN_VALS) * len(PH_FACTORS)  # 150

log(f"  Sweep: {len(GLUCOSE_VALS)} glucose × {len(OXYGEN_VALS)} O₂ × {len(PH_FACTORS)} pH_factor = {N_TOTAL} conditions")

# ── First pass: collect all flux vectors to identify top-20 variable reactions
all_records: list[dict] = []
all_fluxes:  list[dict] = {}   # idx → {rxn_id: flux}

log("  Running FBA sweep …")
n_feasible = 0
for i, glc in enumerate(GLUCOSE_VALS):
    for j, o2 in enumerate(OXYGEN_VALS):
        for k, ph in enumerate(PH_FACTORS):
            with base_model:
                base_model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = glc
                base_model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = o2
                # pH factor: scale all exchange lower bounds (excluding glucose/O2)
                for rxn in base_model.exchanges:
                    if rxn.id not in (RXN_GLUCOSE, RXN_OXYGEN):
                        if rxn.lower_bound < 0:
                            rxn.lower_bound = rxn.lower_bound * ph
                sol = base_model.optimize()
                gr  = sol.objective_value if sol.status == "optimal" else 0.0
                if sol.status == "optimal":
                    n_feasible += 1
                    all_fluxes[len(all_records)] = dict(sol.fluxes)
                else:
                    all_fluxes[len(all_records)] = {}
            all_records.append({
                "glucose_uptake": glc,
                "oxygen_uptake":  o2,
                "pH_factor":      ph,
                "growth_rate":    gr,
            })

log(f"  Feasible solutions : {n_feasible} / {N_TOTAL}")

# ── Identify top-20 most variable reactions (std dev across feasible runs) ──
feasible_ids = [i for i, r in enumerate(all_records) if r["growth_rate"] > 0]
if feasible_ids:
    rxn_ids = list(next(v for v in all_fluxes.values() if v).keys())
    flux_matrix = np.zeros((len(feasible_ids), len(rxn_ids)))
    for row, idx in enumerate(feasible_ids):
        fv = all_fluxes[idx]
        for col, rid in enumerate(rxn_ids):
            flux_matrix[row, col] = fv.get(rid, 0.0)
    stds = flux_matrix.std(axis=0)
    top20_idx  = np.argsort(stds)[-20:][::-1]
    top20_rxns = [rxn_ids[i] for i in top20_idx]
    top20_names = [base_model.reactions.get_by_id(r).name[:35] for r in top20_rxns]
else:
    top20_rxns = []
    top20_names = []

log(f"  Top-20 variable reactions identified: {len(top20_rxns)}")
for rid, rn in zip(top20_rxns[:5], top20_names[:5]):
    log(f"    {rid}  {rn}")

# ── Write CSV ─────────────────────────────────────────────────────────────────
CSV_TRAIN.parent.mkdir(parents=True, exist_ok=True)
fieldnames = ["glucose_uptake", "oxygen_uptake", "pH_factor", "growth_rate"] + top20_rxns
with open(CSV_TRAIN, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for idx, rec in enumerate(all_records):
        row = dict(rec)
        fv  = all_fluxes.get(idx, {})
        for rid in top20_rxns:
            row[rid] = fv.get(rid, 0.0)
        writer.writerow(row)

log(f"\n  Dataset saved → {CSV_TRAIN}")
log(f"  Shape          : {len(all_records)} rows × {len(fieldnames)} columns")

gr_vals = [r["growth_rate"] for r in all_records]
log(f"  Growth rate    : min={min(gr_vals):.4f}  max={max(gr_vals):.4f}  "
    f"mean={np.mean(gr_vals):.4f}  std={np.std(gr_vals):.4f}")
log(f"  Zero-growth    : {sum(1 for g in gr_vals if g == 0.0)} / {len(gr_vals)}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. BUILD & TRAIN CNN SURROGATE MODEL
# ═════════════════════════════════════════════════════════════════════════════
section("2 — BUILD & TRAIN CNN SURROGATE MODEL")

# ── Features: [glucose, oxygen, pH_factor] → normalise ────────────────────
X_raw = np.array([[r["glucose_uptake"], r["oxygen_uptake"], r["pH_factor"]]
                  for r in all_records], dtype=np.float32)
y_raw = np.array([r["growth_rate"] for r in all_records], dtype=np.float32)

feat_mean = X_raw.mean(axis=0)
feat_std  = X_raw.std(axis=0) + 1e-8
X_norm    = (X_raw - feat_mean) / feat_std

scaler_params = {
    "feature_mean": feat_mean.tolist(),
    "feature_std":  feat_std.tolist(),
    "feature_names": ["glucose_uptake", "oxygen_uptake", "pH_factor"],
}

X_tr, X_te, y_tr, y_te = train_test_split(
    X_norm, y_raw, test_size=0.20, random_state=SEED
)

log(f"  Train samples : {len(X_tr)}   Test samples : {len(X_te)}")
log(f"  Feature means : {feat_mean.tolist()}")
log(f"  Feature stds  : {[f'{v:.4f}' for v in feat_std.tolist()]}")

# ── CNN architecture ──────────────────────────────────────────────────────────
REPEAT  = 8    # each feature repeated 8× → length-24 sequence
SEQ_LEN = 3 * REPEAT   # 24

class GrowthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(32 * SEQ_LEN, 128)
        self.drop  = nn.Dropout(0.2)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x: (B, 3)  →  expand  →  (B, 1, 24)
        x = x.unsqueeze(1).repeat(1, 1, REPEAT).reshape(x.size(0), 1, SEQ_LEN)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)

model_nn = GrowthCNN()
log(f"\n  Architecture:")
log(f"    Input → repeat(8×) → (1, 24) sequence")
log(f"    Conv1d(1→32,k=3) → Conv1d(32→64,k=3) → Conv1d(64→32,k=3)")
log(f"    Flatten → Linear(768→128) → Dropout(0.2) → Linear(128→64) → Linear(64→1)")
n_params = sum(p.numel() for p in model_nn.parameters())
log(f"    Parameters: {n_params:,}")

# ── DataLoaders ───────────────────────────────────────────────────────────────
def make_loader(X, y, batch_size=16, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_tr, y_tr)
test_loader  = make_loader(X_te, y_te, shuffle=False)

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3)
loss_fn   = nn.MSELoss()

EPOCHS     = 100
BATCH_SIZE = 16

log(f"\n  Training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr=0.001, Adam+MSELoss")
log(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Test Loss':>12}")
log(f"  {'-'*6}  {'-'*12}  {'-'*12}")

train_losses: list[float] = []
test_losses:  list[float] = []

for epoch in range(1, EPOCHS + 1):
    model_nn.train()
    ep_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model_nn(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= len(X_tr)
    train_losses.append(ep_loss)

    model_nn.eval()
    with torch.no_grad():
        te_loss = sum(
            loss_fn(model_nn(xb), yb).item() * len(xb)
            for xb, yb in test_loader
        ) / len(X_te)
    test_losses.append(te_loss)

    if epoch % 10 == 0:
        log(f"  {epoch:>6}  {ep_loss:>12.6f}  {te_loss:>12.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. EVALUATE SURROGATE
# ═════════════════════════════════════════════════════════════════════════════
section("3 — EVALUATE SURROGATE")

model_nn.eval()
with torch.no_grad():
    y_pred_te = model_nn(torch.from_numpy(X_te)).numpy()

mse  = mean_squared_error(y_te, y_pred_te)
rmse = math.sqrt(mse)
mae  = mean_absolute_error(y_te, y_pred_te)
r2   = r2_score(y_te, y_pred_te)

log(f"  Test-set metrics:")
log(f"    MSE  : {mse:.6f}")
log(f"    RMSE : {rmse:.6f}")
log(f"    MAE  : {mae:.6f}")
log(f"    R²   : {r2:.4f}")

log(f"\n  Sample predictions (10 from test set):")
log(f"  {'#':>3}  {'Actual':>10}  {'Predicted':>10}  {'Error':>10}")
log(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}")
for i in range(min(10, len(y_te))):
    act  = y_te[i]
    pred = float(y_pred_te[i])
    err  = pred - act
    log(f"  {i+1:>3}  {act:>10.6f}  {pred:>10.6f}  {err:>+10.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. VALIDATE SURROGATE VS FBA
# ═════════════════════════════════════════════════════════════════════════════
section("4 — VALIDATE SURROGATE vs FBA")

N_VALIDATE = 20
rng = np.random.default_rng(SEED + 1)
val_glc = rng.uniform(-10.0, -0.5, N_VALIDATE)
val_o2  = rng.uniform(-20.0, -0.5, N_VALIDATE)
val_ph  = rng.uniform(0.5, 1.1, N_VALIDATE)

log(f"  Running {N_VALIDATE} random validation conditions …")
log(f"\n  {'#':>3}  {'Glucose':>8}  {'O2':>7}  {'pHf':>5}  "
    f"{'FBA_growth':>12}  {'CNN_growth':>12}  {'Err%':>8}")
log(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*8}")

val_errors: list[float] = []
for i in range(N_VALIDATE):
    glc_v = float(val_glc[i])
    o2_v  = float(val_o2[i])
    ph_v  = float(val_ph[i])

    # FBA
    with base_model:
        base_model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = glc_v
        base_model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = o2_v
        for rxn in base_model.exchanges:
            if rxn.id not in (RXN_GLUCOSE, RXN_OXYGEN):
                if rxn.lower_bound < 0:
                    rxn.lower_bound = rxn.lower_bound * ph_v
        sol = base_model.optimize()
        fba_gr = sol.objective_value if sol.status == "optimal" else 0.0

    # Surrogate
    feat = np.array([[glc_v, o2_v, ph_v]], dtype=np.float32)
    feat_n = (feat - feat_mean) / feat_std
    model_nn.eval()
    with torch.no_grad():
        cnn_gr = float(model_nn(torch.from_numpy(feat_n)).item())
    cnn_gr = max(0.0, cnn_gr)   # growth cannot be negative

    if fba_gr > 0:
        err_pct = abs(cnn_gr - fba_gr) / fba_gr * 100
    else:
        err_pct = abs(cnn_gr) * 100
    val_errors.append(err_pct)

    log(f"  {i+1:>3}  {glc_v:>8.2f}  {o2_v:>7.2f}  {ph_v:>5.2f}  "
        f"{fba_gr:>12.6f}  {cnn_gr:>12.6f}  {err_pct:>7.1f}%")

log(f"\n  Mean absolute error% : {np.mean(val_errors):.2f}%")
log(f"  Max absolute error%  : {np.max(val_errors):.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# 5. SAVE MODEL
# ═════════════════════════════════════════════════════════════════════════════
section("5 — SAVE MODEL & REPORT")

MODEL_PT.parent.mkdir(parents=True, exist_ok=True)
torch.save(model_nn.state_dict(), MODEL_PT)
log(f"  Model weights → {MODEL_PT}  ({MODEL_PT.stat().st_size:,} bytes)")

with open(SCALER_JSON, "w") as fh:
    json.dump(scaler_params, fh, indent=2)
log(f"  Scaler params → {SCALER_JSON}")

# ── Full report ───────────────────────────────────────────────────────────────
REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
report_header = [
    SEP,
    "LAYER 5 SURROGATE MODEL REPORT",
    "Organism : Saccharomyces boulardii CNCM I-745",
    "Input GEM: cncm_i745_regulated.xml",
    SEP,
    "",
    "── Dataset ──",
    f"  Training samples  : {len(all_records)}  (150 FBA sweep conditions)",
    f"  Feasible          : {n_feasible}",
    f"  Growth range      : {min(gr_vals):.4f} – {max(gr_vals):.4f} h⁻¹",
    f"  Saved to          : {CSV_TRAIN}",
    "",
    "── CNN Architecture ──",
    f"  Expand 3 features × 8 repeats → (1, 24) 1D sequence",
    f"  Conv1d(1→32) → Conv1d(32→64) → Conv1d(64→32) → FC(768→128→64→1)",
    f"  Trainable parameters: {n_params:,}",
    "",
    "── Training ──",
    f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Optimizer: Adam(lr=0.001)",
    f"  Final train loss : {train_losses[-1]:.6f}",
    f"  Final test  loss : {test_losses[-1]:.6f}",
    "",
    "── Test-Set Metrics ──",
    f"  MSE  : {mse:.6f}",
    f"  RMSE : {rmse:.6f}",
    f"  MAE  : {mae:.6f}",
    f"  R²   : {r2:.4f}",
    "",
    "── Validation vs FBA (20 random conditions) ──",
    f"  Mean |error%| : {np.mean(val_errors):.2f}%",
    f"  Max  |error%| : {np.max(val_errors):.2f}%",
    "",
    "── Output Files ──",
    f"  Training data  : {CSV_TRAIN}",
    f"  Model weights  : {MODEL_PT}",
    f"  Scaler params  : {SCALER_JSON}",
    f"  This report    : {REPORT_TXT}",
    "",
    SEP,
    "",
]
with open(REPORT_TXT, "w") as fh:
    fh.write("\n".join(report_header))
    fh.write("\nFull console output:\n")
    fh.write("\n".join(log_lines))

log(f"  Report → {REPORT_TXT}")
log("")
log("LAYER 5 COMPLETE")
