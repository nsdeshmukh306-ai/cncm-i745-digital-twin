"""
Task 2.5 — CNN Surrogate v2 using CNCM I-745 strain-specific GEM.
Regenerates 2000-sample LHS dataset and retrains CNN.
Saves to surrogate_model_v2.pt and surrogate_scaler_v2.json
"""

import io, json, csv, random, math
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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE         = Path("/home/nsdeshmukh306/digital-twin")
GEM_IN       = BASE / "data/gem/cncm_i745_strain_specific.xml"
CSV_OUT      = BASE / "data/ml_datasets/fba_lhs_2000_v2.csv"
MODEL_PT     = BASE / "data/ml_datasets/surrogate_model_v2.pt"
SCALER_JSON  = BASE / "data/ml_datasets/surrogate_scaler_v2.json"
REPORT_TXT   = BASE / "logs/layer5_report_v3.txt"

SEP = "=" * 65
log_lines = []

def log(msg=""):
    print(msg); log_lines.append(str(msg))

def load_silent(path):
    buf = io.StringIO()
    with redirect_stderr(buf):
        return read_sbml_model(str(path))

RXN_GLUCOSE = "r_1714"
RXN_OXYGEN  = "r_1992"
RXN_NH4     = "r_1654"
RXN_PI      = "r_2005"
SEQ_LEN     = 32

# ─── CNN architecture (identical to v1 for fair comparison) ───────────────────
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
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(m(xb), yb); loss.backward(); opt.step()
    return m

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GENERATE LHS DATASET
# ═══════════════════════════════════════════════════════════════════════════════
log(SEP); log("STEP 1 — Generate LHS Dataset (strain-specific model)"); log(SEP)

log(f"  Loading: {GEM_IN}")
base_model = load_silent(GEM_IN)
log(f"  Model: {len(base_model.reactions)} reactions, {len(base_model.genes)} genes")

N_SAMPLES = 2000
lhs_design  = lhs(4, samples=N_SAMPLES, criterion="maximin", random_state=SEED)
glucose_vals = -0.1 + lhs_design[:, 0] * (-20.0 - (-0.1))
oxygen_vals  =  0.0 + lhs_design[:, 1] * (-20.0 - 0.0)
nh4_vals     = -0.1 + lhs_design[:, 2] * (-5.0  - (-0.1))
pi_vals      = -0.1 + lhs_design[:, 3] * (-2.0  - (-0.1))

log(f"  Running {N_SAMPLES} FBA simulations ...")
records = []
n_feasible = 0

for i in range(N_SAMPLES):
    if i % 400 == 0: log(f"    {i}/{N_SAMPLES}")
    glc, o2, nh4, pi = float(glucose_vals[i]), float(oxygen_vals[i]), float(nh4_vals[i]), float(pi_vals[i])
    with base_model:
        base_model.reactions.get_by_id(RXN_GLUCOSE).lower_bound = glc
        base_model.reactions.get_by_id(RXN_OXYGEN).lower_bound  = o2
        base_model.reactions.get_by_id(RXN_NH4).lower_bound     = nh4
        base_model.reactions.get_by_id(RXN_PI).lower_bound      = pi
        sol = base_model.optimize()
        feasible = sol.status == "optimal"
        gr = float(sol.objective_value) if feasible else 0.0
    if feasible: n_feasible += 1
    records.append({"glucose": glc, "oxygen": o2, "nh4": nh4, "pi": pi,
                    "growth_rate": gr, "feasible": int(feasible)})

gr_vals = [r["growth_rate"] for r in records]
log(f"\n  Feasible: {n_feasible}/{N_SAMPLES}")
log(f"  Growth range: {min(gr_vals):.4f} – {max(gr_vals):.4f}  mean={np.mean(gr_vals):.4f}")

CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(CSV_OUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["glucose","oxygen","nh4","pi","growth_rate","feasible"])
    writer.writeheader(); writer.writerows(records)
log(f"  Saved → {CSV_OUT}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 5-FOLD CROSS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
log(""); log(SEP); log("STEP 2 — 5-Fold CV"); log(SEP)

X_raw = np.array([[r["glucose"], r["oxygen"], r["nh4"], r["pi"]] for r in records], dtype=np.float32)
y_raw = np.array([r["growth_rate"] for r in records], dtype=np.float32)
feat_mean = X_raw.mean(axis=0); feat_std = X_raw.std(axis=0) + 1e-8
X_norm = (X_raw - feat_mean) / feat_std

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_r2, fold_rmse = [], []
log(f"  {'Fold':>4}  {'R²':>8}  {'RMSE':>10}")
for fi, (tri, vali) in enumerate(kf.split(X_norm)):
    fm = train_model(X_norm[tri], y_raw[tri], epochs=150)
    fm.eval()
    with torch.no_grad():
        yp = fm(torch.from_numpy(X_norm[vali])).numpy()
    fr2 = r2_score(y_raw[vali], yp)
    fr = math.sqrt(mean_squared_error(y_raw[vali], yp))
    fold_r2.append(fr2); fold_rmse.append(fr)
    log(f"  {fi+1:>4}  {fr2:>8.4f}  {fr:>10.6f}")

cv_r2_mean = float(np.mean(fold_r2)); cv_r2_std = float(np.std(fold_r2))
log(f"\n  Mean R² ± std : {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
log(f"  Mean RMSE     : {np.mean(fold_rmse):.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FINAL MODEL + SAVE
# ═══════════════════════════════════════════════════════════════════════════════
log(""); log(SEP); log("STEP 3 — Final model + save"); log(SEP)

final_model = train_model(X_norm, y_raw, epochs=200, batch_size=32)
final_model.eval()
with torch.no_grad():
    y_pred_all = final_model(torch.from_numpy(X_norm)).numpy()
full_r2   = r2_score(y_raw, y_pred_all)
full_rmse = math.sqrt(mean_squared_error(y_raw, y_pred_all))
log(f"  Full-dataset R²  : {full_r2:.4f}   RMSE: {full_rmse:.6f}")

MODEL_PT.parent.mkdir(parents=True, exist_ok=True)
torch.save(final_model.state_dict(), MODEL_PT)
log(f"  Model saved → {MODEL_PT}")

scaler_data = {
    "feature_names": ["glucose", "oxygen", "nh4", "pi"],
    "mean": feat_mean.tolist(), "std": feat_std.tolist(),
    "feature_mean": feat_mean.tolist(), "feature_std": feat_std.tolist(),
    "seq_len": SEQ_LEN,
    "cross_val_r2_mean": cv_r2_mean, "cross_val_r2_std": cv_r2_std,
    "full_r2": full_r2, "full_rmse": full_rmse,
    "n_samples": N_SAMPLES, "n_feasible": n_feasible,
    "base_model": "cncm_i745_strain_specific.xml",
    "version": "v2",
}
with open(SCALER_JSON, "w") as fh:
    json.dump(scaler_data, fh, indent=2)
log(f"  Scaler saved → {SCALER_JSON}")

# Comparison summary
log("")
log(SEP)
log("SURROGATE COMPARISON")
log(SEP)
v1_json = BASE / "data/ml_datasets/surrogate_scaler.json"
if v1_json.exists():
    with open(v1_json) as fh:
        v1 = json.load(fh)
    log(f"  v1 (Yeast9-based):         R² = {v1.get('cross_val_r2_mean','?'):.4f} ± {v1.get('cross_val_r2_std','?'):.4f}")
log(f"  v2 (CNCM I-745 specific):  R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_TXT, "w") as fh:
    fh.write(SEP + "\nLAYER 5 SURROGATE v3 REPORT\nBase: cncm_i745_strain_specific.xml\n" + SEP + "\n\n")
    fh.write("\n".join(log_lines))
log(f"\n  Report → {REPORT_TXT}")

log(""); log("LAYER 5 v3 COMPLETE")
print("LAYER 5 v3 COMPLETE")
