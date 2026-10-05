"""Experiment E1: a Laya-style act head that decides "act or abstain" for the candidate a fixed rule picks.

The fixed rule is the analytic score (default weights, no label fitting). Over the valid candidates of a scene its
softmax gives Laya's four act-head inputs: top1, top1 - top2, normalised entropy and the option count. The gate is
an L2 logistic regression trained on the picked candidate's outcome (one sample per rollout):

    G0  always act (train success rate)
    G1  Laya inputs only
    G2  control signals of the picked candidate (manipulability, margins, clearance, tracking, perception)
    G3  Laya inputs + control signals

Fit on train + val + calib scenes, test on test scenes. Reports failure AUROC, Brier, success vs coverage and
bootstrap intervals (over test scenes) for G3 - G1.

    python scripts/act_gate.py --data data/gate1
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from a0509pp.cli import write_report  # noqa: E402
from a0509pp.recording import load_decision_groups  # noqa: E402
from a0509pp.selectors import AnalyticSelector, Scaler, auroc, binary_nll, brier, sigmoid  # noqa: E402

CONTROL = ["sigma_min", "cond_max", "jl_margin_min", "torque_util_max", "vel_util_max", "acc_util_max", "env_clear_min",
           "self_clear_min", "track_err_pred_max", "cap_residual_max", "target_conf"]
LAYA = ["top1", "margin", "entropy", "k"]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", required=True)
parser.add_argument("--label", default="y_task", choices=["y_task", "y_exec"])
parser.add_argument("--og", default=None, help="only observation groups whose id contains this, e.g. fixed_camera")
parser.add_argument("--l2", type=float, default=1.0)
parser.add_argument("--bootstrap", type=int, default=1000)
args = parser.parse_args()

groups = {s: load_decision_groups(args.data, s, "features", args.og) for s in ("train", "val", "calib", "test")}
fit_groups = groups["train"] + groups["val"] + groups["calib"]
if not fit_groups or not groups["test"]:
    sys.exit("need train and test scenes; collect more scenes")
names = fit_groups[0]["feature_names"]
rule = AnalyticSelector(names)
rule.scaler = Scaler().fit(np.concatenate([g["X"][g["mask"].astype(bool)] for g in fit_groups])[:, rule.idx])
ctrl_idx = [names.index(n) for n in CONTROL]
k_max = len(fit_groups[0]["mask"])


def samples(gs):
    """One row per rollout of the picked candidate: Laya inputs, control signals, label, scene index."""
    L, C, y, scene = [], [], [], []
    for gi, g in enumerate(gs):
        valid = np.flatnonzero(g["mask"])
        if len(valid) == 0:
            continue
        s = rule.score(g["X"][valid])
        q = np.exp(s - s.max())
        q /= q.sum()
        top = np.sort(q)[::-1]
        k = len(q)
        laya = [top[0], top[0] - (top[1] if k > 1 else 0.0), -(q * np.log(q + 1e-12)).sum() / np.log(max(k, 2)), k / k_max]
        pick = valid[int(np.argmax(s))]
        for v in g[args.label][pick]:
            L.append(laya)
            C.append(g["X"][pick, ctrl_idx])
            y.append(v)
            scene.append(gi)
    return np.array(L, float), np.nan_to_num(np.array(C, float)), np.array(y, float), np.array(scene, int)


def fit_logreg(X, y, l2):
    """L2 logistic regression by Newton's method; X already standardised, intercept unpenalised."""
    A = np.c_[X, np.ones(len(X))]
    w = np.zeros(A.shape[1])
    R = l2 * np.diag(np.r_[np.ones(X.shape[1]), 0.0])
    for _ in range(50):
        p = sigmoid(A @ w)
        step = np.linalg.solve(A.T @ (A * (p * (1 - p))[:, None]) + R + 1e-9 * np.eye(len(w)), A.T @ (p - y) + R @ w)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def coverage_curve(p, y, fracs=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)):
    order = np.argsort(-p, kind="mergesort")
    return [float(y[order[:max(1, int(round(f * len(y))))]].mean()) for f in fracs]


def metrics(p, y):
    return {"auroc": auroc(p, y), "brier": brier(p, y), "nll": binary_nll(p, y), "aurc": float(np.mean(coverage_curve(p, y)))}


Ltr, Ctr, ytr, _ = samples(fit_groups)
Lte, Cte, yte, ste = samples(groups["test"])
if len(np.unique(ytr)) < 2:
    sys.exit(f"training labels are all {int(ytr[0])}: the gate has nothing to learn; make the scenes harder")
inputs = {"G1": (Ltr, Lte), "G2": (Ctr, Cte), "G3": (np.c_[Ltr, Ctr], np.c_[Lte, Cte])}
pred = {"G0": np.full(len(yte), ytr.mean())}
weights = {}
for name, (Xa, Xb) in inputs.items():
    sc = Scaler().fit(Xa)
    w = fit_logreg(sc(Xa), ytr, args.l2)
    pred[name] = sigmoid(np.c_[sc(Xb), np.ones(len(Xb))] @ w)
    weights[name] = dict(zip((LAYA if name == "G1" else CONTROL if name == "G2" else LAYA + CONTROL) + ["bias"], w.round(4).tolist()))

print(f"fit: {len(ytr)} rollouts, success {ytr.mean():.3f} | test: {len(yte)} rollouts, {len(np.unique(ste))} scenes, "
      f"success {yte.mean():.3f}")
print(f"\n{'gate':4s} {'AUROC':>6s} {'Brier':>6s} {'NLL':>6s} {'AURC':>6s} | success at coverage 100% 90% ... 20%")
report = {"label": args.label, "og": args.og, "fit_rollouts": len(ytr), "test_rollouts": len(yte), "gates": {}, "weights": weights}
for name, p in pred.items():
    m = metrics(p, yte)
    m["coverage"] = coverage_curve(p, yte)
    report["gates"][name] = m
    print(f"{name:4s} {m['auroc']:6.3f} {m['brier']:6.3f} {m['nll']:6.3f} {m['aurc']:6.3f} | " + " ".join(f"{c:.2f}" for c in m["coverage"]))

rng = np.random.default_rng(0)
scenes = np.unique(ste)
diffs = {"G3-G1": [], "G3-G2": []}
for _ in range(args.bootstrap):
    idx = np.concatenate([np.flatnonzero(ste == s) for s in rng.choice(scenes, len(scenes))])
    if len(np.unique(yte[idx])) < 2:
        continue
    m = {n: metrics(pred[n][idx], yte[idx]) for n in ("G1", "G2", "G3")}
    for d, (a, b) in (("G3-G1", ("G3", "G1")), ("G3-G2", ("G3", "G2"))):
        diffs[d].append([m[a]["auroc"] - m[b]["auroc"], m[a]["aurc"] - m[b]["aurc"]])
print()
for d, v in diffs.items():
    if v:
        lo, hi = np.percentile(np.array(v), [2.5, 97.5], axis=0)
        report[d] = {"auroc_ci95": [float(lo[0]), float(hi[0])], "aurc_ci95": [float(lo[1]), float(hi[1])]}
        print(f"{d}: AUROC diff 95% CI [{lo[0]:+.3f}, {hi[0]:+.3f}], AURC diff 95% CI [{lo[1]:+.3f}, {hi[1]:+.3f}]")
write_report(os.path.join(args.data, "act_gate", "report.json"), report)
