"""Section 14 baselines: analytic physical score and a small MLP, calibration metrics, scene-level bootstrap."""

from __future__ import annotations

import numpy as np

from .features import FEATURE_NAMES

# Analytic score: sign and base weight per feature (robust-standardized on the train split). Group multipliers are
# tuned on the validation split for chosen-candidate success; Platt scaling on the calibration split gives probabilities.
ANALYTIC_WEIGHTS = {
    "env_clear_min": 1.0, "self_clear_min": 0.5, "jl_margin_min": 0.3, "sigma_min": 0.3, "env_clear_conf": 0.2,
    "target_conf": 0.3, "open_clearance": 0.3, "cap_residual_max": -0.5, "track_err_pred_max": -0.5,
    "torque_util_max": -0.3, "duration": -0.4, "cond_max": -0.2,
}
ANALYTIC_GROUPS = {
    "margins": ["env_clear_min", "self_clear_min", "jl_margin_min", "sigma_min", "env_clear_conf"],
    "dynamics": ["cap_residual_max", "track_err_pred_max", "torque_util_max", "cond_max"],
    "cost": ["duration"],
    "grasp": ["target_conf", "open_clearance"],
}
NLL_EPS = 1e-6


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class Scaler:
    def fit(self, X):
        self.med = np.nanmedian(X, 0)
        iqr = np.nanpercentile(X, 75, 0) - np.nanpercentile(X, 25, 0)
        sd = np.nanstd(X, 0)
        self.s = np.where(iqr > 1e-9, iqr, np.where(sd > 1e-9, sd, 1.0))
        return self

    def __call__(self, X):
        return np.nan_to_num(np.clip((X - self.med) / self.s, -5.0, 5.0))


def fit_platt(score, y, iters=100, l2=1.0):
    """p = sigmoid(a * score + b): damped Newton on the logistic NLL with Platt's smoothed targets and a small ridge
    on the slope (keeps the fit finite when the score separates the calibration labels)."""
    n_pos, n_neg = float(np.sum(y)), float(len(y) - np.sum(y))
    t = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

    def obj(a, b):
        z = a * score + b
        return float(np.sum(np.logaddexp(0.0, z) - t * z) + 0.5 * l2 * a * a)

    a, b = 0.0, float(np.log((n_pos + 1.0) / (n_neg + 1.0)))
    f = obj(a, b)
    for _ in range(iters):
        p = sigmoid(a * score + b)
        g = np.array([np.sum((p - t) * score) + l2 * a, np.sum(p - t)])
        w = p * (1 - p)
        H = np.array([[np.sum(w * score * score) + l2, np.sum(w * score)], [np.sum(w * score), np.sum(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        k = 1.0
        while k > 1e-6:
            f_new = obj(a - k * step[0], b - k * step[1])
            if f_new <= f:
                break
            k *= 0.5
        a, b, f_old, f = a - k * step[0], b - k * step[1], f, f_new
        if abs(f_old - f) < 1e-10:
            break
    return a, b


def fit_temperature(logit, y):
    Ts = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    nll = [binary_nll(sigmoid(logit / T), y) for T in Ts]
    return float(Ts[int(np.argmin(nll))])


class AnalyticSelector:
    name = "analytic"

    def __init__(self, feature_names=FEATURE_NAMES, weights=None):
        self.names = list(feature_names)
        w = weights or ANALYTIC_WEIGHTS
        self.idx = [self.names.index(k) for k in w if k in self.names]
        self.base = np.array([w[self.names[i]] for i in self.idx])
        self.group_of = [next((g for g, ks in ANALYTIC_GROUPS.items() if self.names[i] in ks), "other") for i in self.idx]
        self.w = self.base.copy()
        self.multipliers = {g: 1.0 for g in ANALYTIC_GROUPS}

    def fit(self, X, y, X_cal=None, y_cal=None, val_groups=None, label="y_task", grid=(0.5, 1.0, 2.0)):
        self.scaler = Scaler().fit(X[:, self.idx])
        if val_groups:
            import itertools

            best = None
            for combo in itertools.product(grid, repeat=len(ANALYTIC_GROUPS)):
                m = dict(zip(ANALYTIC_GROUPS, combo))
                self.w = self.base * np.array([m.get(g, 1.0) for g in self.group_of])
                sel = selection_table(_ScoreOnly(self), val_groups, label)
                val = float(np.mean([r["chosen"] for r in sel])) if sel else 0.0
                key = (val, -sum(abs(np.log(c)) for c in combo))
                if best is None or key > best[0]:
                    best = (key, m)
            self.multipliers = best[1]
            self.w = self.base * np.array([self.multipliers.get(g, 1.0) for g in self.group_of])
        Xc, yc = (X, y) if X_cal is None or len(X_cal) == 0 else (X_cal, y_cal)
        self.a, self.b = fit_platt(self.score(Xc), yc)
        return self

    def score(self, X):
        return self.scaler(X[:, self.idx]) @ self.w

    def predict(self, X):
        return sigmoid(self.a * self.score(X) + self.b)


class _ScoreOnly:
    def __init__(self, m):
        self.m = m

    def predict(self, X):
        return self.m.score(X)


class MLPSelector:
    """Two hidden layers, ReLU, Adam, early stopping on validation NLL, temperature scaling on the calib split."""

    name = "mlp"

    def __init__(self, hidden=64, lr=1e-3, weight_decay=1e-4, epochs=300, batch=256, patience=25, seed=0):
        self.h, self.lr, self.wd, self.epochs, self.batch, self.patience = hidden, lr, weight_decay, epochs, batch, patience
        self.rng = np.random.default_rng(seed)

    def _init(self, d):
        r = self.rng
        self.P = {"W1": r.normal(0, np.sqrt(2 / d), (d, self.h)), "b1": np.zeros(self.h),
                  "W2": r.normal(0, np.sqrt(2 / self.h), (self.h, self.h)), "b2": np.zeros(self.h),
                  "W3": r.normal(0, np.sqrt(1 / self.h), (self.h, 1)), "b3": np.zeros(1)}
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def _forward(self, X, P=None):
        P = P or self.P
        h1 = np.maximum(X @ P["W1"] + P["b1"], 0.0)
        h2 = np.maximum(h1 @ P["W2"] + P["b2"], 0.0)
        return (h2 @ P["W3"] + P["b3"])[:, 0], (h1, h2)

    def _step(self, X, y):
        z, (h1, h2) = self._forward(X)
        n = len(y)
        dz = (sigmoid(z) - y)[:, None] / n
        g = {"W3": h2.T @ dz, "b3": dz.sum(0)}
        d2 = (dz @ self.P["W3"].T) * (h2 > 0)
        g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ self.P["W2"].T) * (h1 > 0)
        g["W1"], g["b1"] = X.T @ d1, d1.sum(0)
        self.t += 1
        for k in self.P:
            if k.startswith("W"):
                g[k] = g[k] + self.wd * self.P[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9**self.t)
            vh = self.v[k] / (1 - 0.999**self.t)
            self.P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, X, y, X_val=None, y_val=None, X_cal=None, y_cal=None):
        self.scaler = Scaler().fit(X)
        Xs = self.scaler(X)
        self._init(Xs.shape[1])
        Xv, yv = (Xs, y) if X_val is None or len(X_val) == 0 else (self.scaler(X_val), y_val)
        best, best_P, wait = np.inf, None, 0
        self.history = []
        for ep in range(self.epochs):
            perm = self.rng.permutation(len(y))
            for i in range(0, len(y), self.batch):
                b = perm[i:i + self.batch]
                self._step(Xs[b], y[b])
            nll = binary_nll(sigmoid(self._forward(Xv)[0]), yv)
            self.history.append(nll)
            if nll < best - 1e-5:
                best, best_P, wait = nll, {k: v.copy() for k, v in self.P.items()}, 0
            else:
                wait += 1
                if wait > self.patience:
                    break
        self.P = best_P or self.P
        self.epochs_run = ep + 1
        self.T = 1.0
        if X_cal is not None and len(X_cal):
            self.T = fit_temperature(self._forward(self.scaler(X_cal))[0], y_cal)
        return self

    def predict(self, X):
        return sigmoid(self._forward(self.scaler(X))[0] / self.T)


class HeuristicSelector:
    """Non-probabilistic references: shortest duration or largest environment clearance."""

    def __init__(self, kind, feature_names=FEATURE_NAMES):
        self.name = kind
        self.i = list(feature_names).index("duration" if kind == "shortest" else "env_clear_min")
        self.sign = -1.0 if kind == "shortest" else 1.0

    def fit(self, *a, **k):
        return self

    def predict(self, X):
        return sigmoid(self.sign * X[:, self.i] * 10.0)


# ---------------------------------------------------------------------- metrics
def binary_nll(p, y):
    p = np.clip(p, NLL_EPS, 1 - NLL_EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def brier(p, y):
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = k == b
        if m.any():
            rows.append({"bin": b, "p_mean": float(p[m].mean()), "y_mean": float(y[m].mean()), "count": int(m.sum())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece), rows


def auroc(p, y):
    """Mann-Whitney AUROC with average ranks for ties."""
    n_pos, n_neg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks = np.empty(len(p))
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------- dataset plumbing
def rollout_arrays(groups, label="y_task"):
    """Per-rollout samples: X (N, F), y (N,), group index (N,)."""
    X, y, g = [], [], []
    for gi, grp in enumerate(groups):
        for c, ys in enumerate(grp[label]):
            if not grp["mask"][c]:
                continue
            for v in ys:
                X.append(grp["X"][c])
                y.append(v)
                g.append(gi)
    F = groups[0]["X"].shape[1] if groups else len(FEATURE_NAMES)
    return (np.asarray(X, float).reshape(-1, F), np.asarray(y, float), np.asarray(g, int))


def selection_table(model, groups, label="y_task"):
    """Per decision group: success of the chosen candidate, oracle best, random choice, chosen probability,
    chosen planned duration and the failure types seen for the chosen candidate."""
    rows = []
    for grp in groups:
        valid = [c for c in range(len(grp["mask"])) if grp["mask"][c] and len(grp[label][c])]
        if not valid:
            continue
        rate = np.array([np.mean(grp[label][c]) for c in valid])
        p = np.asarray(model.predict(grp["X"][valid].astype(float)))
        j = int(np.argmax(p))
        c = valid[j]
        rows.append({"scene_id": grp["scene_id"], "og_id": grp.get("og_id", ""), "chosen": float(rate[j]),
                     "oracle": float(rate.max()), "random": float(rate.mean()), "n": len(valid), "p_chosen": float(p[j]),
                     "y_chosen": list(grp[label][c]), "duration": float(grp["duration"][c]) if "duration" in grp else float("nan"),
                     "failures": [f for f in grp.get("failure", [[]] * len(grp["mask"]))[c] if f] if label == "y_task" else []})
    return rows


def evaluate(model, groups, label="y_task", bins=10, taus=np.linspace(0.0, 0.9, 10)):
    """Probability quality over all executed candidates and over the selected ones, selection success vs oracle and
    random choice, candidate coverage, abstention trade-off, chosen failure causes, and prediction latency."""
    import time

    X, y, _ = rollout_arrays(groups, label)
    out = {"n_rollouts": int(len(y)), "n_groups": len(groups), "base_rate": float(y.mean()) if len(y) else float("nan"),
           "nll_epsilon": NLL_EPS}
    if len(y):
        t0 = time.time()
        p = model.predict(X)
        out["latency_ms_per_candidate"] = 1000.0 * (time.time() - t0) / len(y)
        out["brier"] = brier(p, y)
        out["nll"] = binary_nll(p, y)
        out["ece"], out["reliability"] = reliability(p, y, bins)
        out["auroc"] = auroc(p, y)
    sel = selection_table(model, groups, label)
    if sel:
        ps = np.concatenate([[r["p_chosen"]] * len(r["y_chosen"]) for r in sel])
        ys = np.concatenate([r["y_chosen"] for r in sel]).astype(float)
        out["selected_brier"] = brier(ps, ys)
        out["selected_nll"] = binary_nll(ps, ys)
        out["chosen_success"] = float(np.mean([r["chosen"] for r in sel]))
        out["oracle_success"] = float(np.mean([r["oracle"] for r in sel]))
        out["random_success"] = float(np.mean([r["random"] for r in sel]))
        out["regret"] = out["oracle_success"] - out["chosen_success"]
        out["coverage"] = float(np.mean([r["oracle"] > 0 for r in sel]))
        out["chosen_duration_s"] = float(np.nanmean([r["duration"] for r in sel]))
        fails = {}
        for r in sel:
            for f in r["failures"]:
                fails[f] = fails.get(f, 0) + 1
        out["chosen_failures"] = fails
        curve = []
        for tau in taus:
            act = [r for r in sel if r["p_chosen"] >= tau]
            curve.append({"tau": float(tau), "act_rate": len(act) / len(sel),
                          "success_when_acting": float(np.mean([r["chosen"] for r in act])) if act else float("nan"),
                          "success_overall": float(np.sum([r["chosen"] for r in act]) / len(sel))})
        out["abstention"] = curve
    return out


def bootstrap(model, groups, label="y_task", B=1000, seed=0, keys=("brier", "nll", "ece", "chosen_success", "regret")):
    """Percentile confidence intervals resampling base scenes (all observation groups of a scene together)."""
    rng = np.random.default_rng(seed)
    per = []
    for g in groups:
        valid = [c for c in np.flatnonzero(g["mask"]) if len(g[label][c])]
        pr = model.predict(g["X"][valid].astype(float)) if valid else np.zeros(0)
        ps = np.concatenate([np.full(len(g[label][c]), pr[j]) for j, c in enumerate(valid)]) if valid else np.zeros(0)
        ys = np.concatenate([np.asarray(g[label][c], float) for c in valid]) if valid else np.zeros(0)
        rates = np.array([np.mean(g[label][c]) for c in valid])
        chosen = rates[int(np.argmax(pr))] if valid else np.nan
        per.append((g["scene_id"], ps, ys, chosen, rates.max() if valid else np.nan))
    scenes = sorted({x[0] for x in per})
    by = {s: [x for x in per if x[0] == s] for s in scenes}
    stats = {k: [] for k in keys}
    for _ in range(B):
        pick = rng.integers(len(scenes), size=len(scenes))
        gs = [x for i in pick for x in by[scenes[i]]]
        ps = np.concatenate([x[1] for x in gs])
        ys = np.concatenate([x[2] for x in gs])
        ch = np.array([x[3] for x in gs])
        orc = np.array([x[4] for x in gs])
        ok = np.isfinite(ch)
        vals = {"brier": brier(ps, ys), "nll": binary_nll(ps, ys), "ece": reliability(ps, ys)[0] if len(ys) else np.nan,
                "chosen_success": float(ch[ok].mean()) if ok.any() else np.nan,
                "regret": float(orc[ok].mean() - ch[ok].mean()) if ok.any() else np.nan}
        for k in keys:
            stats[k].append(vals[k])
    return {k: {"lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5)), "mean": float(np.nanmean(v))}
            for k, v in stats.items()}
