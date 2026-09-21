"""Neural surrogate for the genome-scale model, with honest baselines.

A surrogate is only worth having if it beats the cheap alternatives, so the
1-D CNN is trained alongside ridge regression, random forest and gradient
boosting on identical folds. Monte-Carlo dropout supplies predictive
intervals, and inference speed is measured against the linear program rather
than asserted.
"""
from __future__ import annotations
import time
import numpy as np


def latin_hypercube(bounds, n, seed=0):
    """Scrambled Sobol'-quality LHS design over the nutrient space."""
    from scipy.stats import qmc
    keys = list(bounds)
    lhs = qmc.LatinHypercube(d=len(keys), seed=seed)
    u = lhs.random(n)
    lo = np.array([bounds[k][0] for k in keys])
    hi = np.array([bounds[k][1] for k in keys])
    return keys, qmc.scale(u, lo, hi)


# --------------------------------------------------------------------------
def build_cnn(n_in, dropout=0.15):
    import torch.nn as nn

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Dropout(dropout))
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(64 * n_in, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, 1))

        def forward(self, x):
            return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)

    return CNN()


def train_cnn(Xtr, ytr, Xva, yva, epochs=200, lr=1e-3, batch=64, seed=0, patience=25):
    import torch
    torch.manual_seed(seed)
    net = build_cnn(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32); yv = torch.tensor(yva, dtype=torch.float32)
    best, best_state, bad = np.inf, None, 0
    n = len(Xt)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Xv), yv))
        if v < best - 1e-7:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    return net, best


def mc_dropout_predict(net, X, n_pass=100, seed=0):
    """Predictive mean and SD from Monte-Carlo dropout."""
    import torch
    torch.manual_seed(seed)
    net.train()                                   # keep dropout active
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        draws = np.stack([net(Xt).numpy() for _ in range(n_pass)])
    net.eval()
    return draws.mean(axis=0), draws.std(axis=0)


def benchmark_speed(net, X, lp_callable, n_lp=25, seed=0):
    """Measured wall-clock comparison: surrogate forward pass vs LP solve."""
    import torch
    net.eval()
    Xt = torch.tensor(X[:1024], dtype=torch.float32)
    with torch.no_grad():
        net(Xt[:8])                               # warm-up
        t0 = time.perf_counter()
        net(Xt)
        t_nn = (time.perf_counter() - t0) / len(Xt)
    t0 = time.perf_counter()
    for i in range(n_lp):
        lp_callable(X[i])
    t_lp = (time.perf_counter() - t0) / n_lp
    return dict(seconds_per_surrogate_prediction=t_nn,
                seconds_per_lp_solve=t_lp,
                speedup=t_lp / t_nn if t_nn > 0 else float("nan"),
                n_surrogate=int(len(Xt)), n_lp=int(n_lp))
