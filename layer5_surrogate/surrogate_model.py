"""
Layer 5 — CNN Surrogate Model for Saccharomyces boulardii CNCM I-745
2000-sample LHS dataset, 5-fold cross-validation CNN, MC Dropout UQ.
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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import cobra
from cobra.io import read_sbml_model

try:
    from pyDOE3 import lhs
except ImportError:
    from pyDOE2 import lhs

import sys
sys.path.insert(0, str(Path("/home/nsdeshmukh306/digital-twin")))
from references import REFERENCES

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path("/home/nsdeshmukh306/digital-twin")
GEM_IN      = BASE / "data/gem/cncm_i745_gut.xml"
CSV_OUT     = BASE / "data/ml_datasets/fba_lhs_2000.csv"
MODEL_PT    = BASE / "data/ml_datasets/surrogate_model.pt"
SCALER_JSON = BASE / "data/ml_datasets/surrogate_scaler.json"
REPORT_TXT  = BASE / "logs/layer5_report_v2.txt"

SEP = "=" * 65
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

RXN_GLUCOSE = "r_1714"
RXN_OXYGEN  = "r_1992"
RXN_NH4     = "r_1654"
RXN_PI      = "r_2005"

# ═════════════════════════════════════════════════════════════════════════════
# 1. GENERATE LHS DATASET (2000 samples)
# ═════════════════════════════════════════════════════════════════════════════
section("1 — GENERATE LHS DATASET (2000 samples)")

log(f"  Loading GEM: {GEM_IN}")
base_model = load_silent(GEM_IN)
log(f"  Model loaded: {len(base_model.reactions)} reactions, {len(base_model.genes)} genes")

N_SAMPLES = 2000
log(f"  Generating {N_SAMPLES}-sample Latin Hypercube Design ...")

# LHS in [0,1]^4, then scale to physiological ranges
lhs_design = lhs(4, samples=N_SAMPLES, criterion="maximin", random_state=SEED)

# Ranges: glucose [-0.1, -20], oxygen [0, -20], nh4 [-0.1, -5], pi [-0.1, -2]
glucose_vals = -0.1 + lhs_design[:, 0] * (-20.0 - (-0.1))   # [-0.1, -20]
oxygen_vals  = 0.0  + lhs_design[:, 1] * (-20.0 - 0.0)       # [0, -20]
nh4_vals     = -0.1 + lhs_design[:, 2] * (-5.0  - (-0.1))    # [-0.1, -5]
pi_vals      = -0.1 + lhs_design[:, 3] * (-2.0  - (-0.1))    # [-0.1, -2]

log(f"  Running {N_SAMPLES} FBA simulations ...")
records = []
n_feasible = 0

for i in range(N_SAMPLES):
    if i % 200 == 0:
        log(f"    Progress: {i}/{N_SAMPLES} ({100*i//N_SAMPLES}%)")
    glc = float(glucose_vals[i])
    o2  = float(oxygen_vals[i])
    nh4 = float(nh4_vals[i])
    pi  = float(pi_vals[i])
    with base_model:
        base_model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = glc
        base_model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = o2
        base_model.reactions.get_by_id(RXN_NH4).lower_bound     = nh4
        base_model.reactions.get_by_id(RXN_PI).lower_bound      = pi
        sol = base_model.optimize()
        feasible = sol.status == "optimal"
        gr = float(sol.objective_value) if feasible else 0.0
    if feasible:
        n_feasible += 1
    records.append({
        "glucose": glc, "oxygen": o2, "nh4": nh4, "pi": pi,
        "growth_rate": gr, "feasible": int(feasible)
    })

gr_vals = [r["growth_rate"] for r in records]
log(f"\n  Dataset shape : {len(records)} rows × 6 columns")
log(f"  Feasible      : {n_feasible} / {N_SAMPLES}")
log(f"  Growth rate   : min={min(gr_vals):.4f}  max={max(gr_vals):.4f}  "
    f"mean={np.mean(gr_vals):.4f}  std={np.std(gr_vals):.4f}")

CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(CSV_OUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["glucose","oxygen","nh4","pi","growth_rate","feasible"])
    writer.writeheader()
    writer.writerows(records)
log(f"  Saved → {CSV_OUT}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. CNN SURROGATE WITH 5-FOLD CROSS VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
section("2 — CNN SURROGATE WITH 5-FOLD CROSS VALIDATION")

X_raw = np.array([[r["glucose"], r["oxygen"], r["nh4"], r["pi"]]
                  for r in records], dtype=np.float32)
y_raw = np.array([r["growth_rate"] for r in records], dtype=np.float32)

feat_mean = X_raw.mean(axis=0)
feat_std  = X_raw.std(axis=0) + 1e-8
X_norm    = (X_raw - feat_mean) / feat_std

log(f"  Features: glucose, oxygen, nh4, pi (4 inputs)")
log(f"  Feature means : {feat_mean.tolist()}")
log(f"  Feature stds  : {feat_std.tolist()}")

# CNN architecture: 4 features → expand to (32, 1) sequence
SEQ_LEN = 32

class GrowthCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.fc1   = nn.Linear(64 * SEQ_LEN, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(128, 1)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x: (B, 4) → repeat to (B, 1, 32)
        x = x.unsqueeze(1).repeat(1, 1, SEQ_LEN // x.size(1)).reshape(x.size(0), 1, SEQ_LEN)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x).squeeze(1)

def train_model(X_tr, y_tr, epochs=150, batch_size=32):
    m = GrowthCNN()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    m.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(m(xb), yb)
            loss.backward()
            opt.step()
    return m

n_params = sum(p.numel() for p in GrowthCNN().parameters())
log(f"\n  Architecture: 4 → repeat(8×) → (1,32) → Conv64→128→64 → FC256→128→1")
log(f"  BatchNorm1d after each conv, Dropout(0.3/0.2) after FC")
log(f"  Parameters: {n_params:,}")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_r2_list: list[float] = []
fold_rmse_list: list[float] = []

log(f"\n  5-Fold Cross Validation (150 epochs, batch=32, Adam lr=0.001):")
log(f"  {'Fold':>4}  {'R²':>8}  {'RMSE':>10}")
log(f"  {'-'*4}  {'-'*8}  {'-'*10}")

for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_norm)):
    X_tr_f = X_norm[tr_idx]
    y_tr_f = y_raw[tr_idx]
    X_val_f = X_norm[val_idx]
    y_val_f = y_raw[val_idx]

    fold_model = train_model(X_tr_f, y_tr_f, epochs=150)
    fold_model.eval()
    with torch.no_grad():
        y_pred_f = fold_model(torch.from_numpy(X_val_f)).numpy()

    fold_r2   = r2_score(y_val_f, y_pred_f)
    fold_rmse = math.sqrt(mean_squared_error(y_val_f, y_pred_f))
    fold_r2_list.append(fold_r2)
    fold_rmse_list.append(fold_rmse)
    log(f"  {fold_idx+1:>4}  {fold_r2:>8.4f}  {fold_rmse:>10.6f}")

cv_r2_mean = float(np.mean(fold_r2_list))
cv_r2_std  = float(np.std(fold_r2_list))
log(f"\n  Mean R² ± std : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
log(f"  Mean RMSE     : {np.mean(fold_rmse_list):.6f}")

# Final model on full dataset
log(f"\n  Training final model on full dataset (200 epochs) ...")
final_model = train_model(X_norm, y_raw, epochs=200, batch_size=32)
final_model.eval()

with torch.no_grad():
    y_pred_all = final_model(torch.from_numpy(X_norm)).numpy()
full_r2   = r2_score(y_raw, y_pred_all)
full_rmse = math.sqrt(mean_squared_error(y_raw, y_pred_all))
log(f"  Full-dataset R² : {full_r2:.4f}   RMSE : {full_rmse:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. MC DROPOUT UNCERTAINTY QUANTIFICATION
# ═════════════════════════════════════════════════════════════════════════════
section("3 — MC DROPOUT UNCERTAINTY QUANTIFICATION")

N_MC = 50

def mc_predict(model, X_tensor, n_passes=50):
    model.train()  # enable dropout at inference
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(X_tensor).numpy())
    model.eval()
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)

# Use first 5 test samples for illustration
test_X = torch.from_numpy(X_norm[:5])
mc_mean, mc_std = mc_predict(final_model, test_X, n_passes=N_MC)

log(f"  MC Dropout: {N_MC} forward passes per sample")
log(f"  95% CI = mean ± 1.96*std")
log(f"\n  {'#':>3}  {'True':>8}  {'MC Mean':>10}  {'95% CI Lower':>12}  {'95% CI Upper':>12}")
log(f"  {'-'*3}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*12}")
for i in range(5):
    ci_lo = mc_mean[i] - 1.96 * mc_std[i]
    ci_hi = mc_mean[i] + 1.96 * mc_std[i]
    log(f"  {i+1:>3}  {y_raw[i]:>8.4f}  {mc_mean[i]:>10.4f}  {max(0, ci_lo):>12.4f}  {ci_hi:>12.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. SAVE MODEL AND SCALER
# ═════════════════════════════════════════════════════════════════════════════
section("4 — SAVE MODEL & SCALER")

MODEL_PT.parent.mkdir(parents=True, exist_ok=True)
torch.save(final_model.state_dict(), MODEL_PT)
log(f"  Model saved → {MODEL_PT}  ({MODEL_PT.stat().st_size:,} bytes)")

scaler_data = {
    "feature_names": ["glucose", "oxygen", "nh4", "pi"],
    "mean": feat_mean.tolist(),
    "std": feat_std.tolist(),
    "feature_mean": feat_mean.tolist(),
    "feature_std": feat_std.tolist(),
    "seq_len": SEQ_LEN,
    "cross_val_r2_mean": cv_r2_mean,
    "cross_val_r2_std": cv_r2_std,
    "full_r2": full_r2,
    "full_rmse": full_rmse,
    "n_samples": N_SAMPLES,
    "n_feasible": n_feasible,
}
with open(SCALER_JSON, "w") as fh:
    json.dump(scaler_data, fh, indent=2)
log(f"  Scaler saved → {SCALER_JSON}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. SAVE REPORT
# ═════════════════════════════════════════════════════════════════════════════
section("5 — SAVE REPORT")

report_lines = [
    SEP,
    "LAYER 5 SURROGATE MODEL REPORT v2",
    "Organism : Saccharomyces boulardii CNCM I-745",
    "Input GEM: cncm_i745_gut.xml",
    SEP,
    "",
    "── Dataset ──",
    f"  Samples       : {N_SAMPLES} (Latin Hypercube Sampling)",
    f"  Feasible FBA  : {n_feasible} / {N_SAMPLES}",
    f"  Features      : glucose, oxygen, nh4, pi",
    f"  Growth range  : {min(gr_vals):.4f} – {max(gr_vals):.4f} h⁻¹",
    f"  Saved to      : {CSV_OUT}",
    "",
    "── CNN Architecture ──",
    f"  Input: 4 features → expand(8×) → (1, 32) sequence",
    f"  Conv1d(1→64)→BN→ReLU → Conv1d(64→128)→BN→ReLU → Conv1d(128→64)→BN→ReLU",
    f"  Flatten → Linear(2048→256)→ReLU→Dropout(0.3)",
    f"  → Linear(256→128)→ReLU→Dropout(0.2) → Linear(128→1)",
    f"  Parameters: {n_params:,}",
    "",
    "── 5-Fold Cross Validation ──",
]
for i, (r2, rmse) in enumerate(zip(fold_r2_list, fold_rmse_list)):
    report_lines.append(f"  Fold {i+1}: R²={r2:.4f}  RMSE={rmse:.6f}")
report_lines += [
    f"  Mean R² : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}",
    f"  Mean RMSE: {np.mean(fold_rmse_list):.6f}",
    "",
    "── Final Model (200 epochs, full dataset) ──",
    f"  R²   : {full_r2:.4f}",
    f"  RMSE : {full_rmse:.6f}",
    "",
    "── MC Dropout UQ ──",
    f"  Passes: {N_MC}  |  95% CI = mean ± 1.96*std",
    "",
    "── References ──",
]
for ref in REFERENCES["layer5_surrogate"]:
    report_lines.append(f"  • {ref}")
report_lines += ["", SEP]

REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_TXT, "w") as fh:
    fh.write("\n".join(report_lines))
    fh.write("\n\nFull console output:\n")
    fh.write("\n".join(log_lines))

log(f"  Report saved → {REPORT_TXT}")
log("")
log("LAYER 5 v2 COMPLETE")
