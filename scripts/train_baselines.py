"""Section 14 baselines on a collected dataset: analytic physical score and a small MLP.

Splits are by base scene (train/val/calib/test from the dataset). Reports Brier, NLL, ECE, AUROC, reliability,
chosen-candidate success vs oracle and random choice, with 95% bootstrap intervals over test scenes.
GPU-free; plain Python.

    python scripts/train_baselines.py --data data/pilot
    python scripts/train_baselines.py --data data/pilot --label y_exec --features features_oracle
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from a0509pp.cli import write_report  # noqa: E402
from a0509pp.recording import load_decision_groups  # noqa: E402
from a0509pp.selectors import AnalyticSelector, HeuristicSelector, MLPSelector, bootstrap, evaluate, rollout_arrays  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", required=True)
parser.add_argument("--label", default="y_task", choices=["y_task", "y_exec"])
parser.add_argument("--features", default="features", choices=["features", "features_oracle"])
parser.add_argument("--og", default=None, help="only observation groups whose id contains this, e.g. fixed_camera")
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--out", default=None)
args = parser.parse_args()
out_dir = args.out or os.path.join(args.data, "baselines")

split = {s: load_decision_groups(args.data, s, args.features, args.og) for s in ("train", "val", "calib", "test")}
for s, g in split.items():
    X, y, _ = rollout_arrays(g, args.label) if g else (np.zeros((0, 1)), np.zeros(0), None)
    print(f"{s:5s}: {len(g):4d} decision groups, {len(y):5d} rollouts, base rate {y.mean() if len(y) else float('nan'):.3f}")
if not split["train"] or not split["test"]:
    sys.exit("need train and test scenes; collect more scenes")

X, y, _ = rollout_arrays(split["train"], args.label)
Xv, yv, _ = rollout_arrays(split["val"], args.label) if split["val"] else (None, None, None)
Xc, yc, _ = rollout_arrays(split["calib"], args.label) if split["calib"] else (None, None, None)
names = split["train"][0]["feature_names"]
models = [AnalyticSelector(names).fit(X, y, Xc, yc, val_groups=split["val"], label=args.label),
          MLPSelector(epochs=args.epochs).fit(X, y, Xv, yv, Xc, yc),
          HeuristicSelector("shortest", names), HeuristicSelector("clearance", names)]
report = {"label": args.label, "features": args.features, "og_filter": args.og, "splits": {s: len(g) for s, g in split.items()},
          "models": {}}
print(f"\n{'model':10s} {'Brier':>7s} {'NLL':>7s} {'ECE':>6s} {'AUROC':>6s} {'sel.Brier':>9s} {'chosen':>7s} {'oracle':>7s} "
      f"{'random':>7s} {'coverage':>8s}")
for m in models:
    ev = evaluate(m, split["test"], args.label)
    probabilistic = m.name in ("analytic", "mlp")
    if probabilistic and args.bootstrap:
        ev["bootstrap_95"] = bootstrap(m, split["test"], args.label, B=args.bootstrap)
    if not probabilistic:
        for k in ("brier", "nll", "ece", "reliability", "selected_brier", "selected_nll", "abstention"):
            ev.pop(k, None)
    report["models"][m.name] = ev
    f = lambda k: f"{ev[k]:.3f}" if k in ev and ev[k] == ev[k] else "  -  "
    print(f"{m.name:10s} {f('brier'):>7s} {f('nll'):>7s} {f('ece'):>6s} {f('auroc'):>6s} {f('selected_brier'):>9s} "
          f"{f('chosen_success'):>7s} {f('oracle_success'):>7s} {f('random_success'):>7s} {f('coverage'):>8s}")
    if "bootstrap_95" in ev:
        b = ev["bootstrap_95"]
        print(f"{'':10s} 95% CI: Brier [{b['brier']['lo']:.3f}, {b['brier']['hi']:.3f}]  chosen "
              f"[{b['chosen_success']['lo']:.3f}, {b['chosen_success']['hi']:.3f}]  regret "
              f"[{b['regret']['lo']:.3f}, {b['regret']['hi']:.3f}]")
for name in ("analytic", "mlp"):
    ab = report["models"][name].get("abstention", [])
    if ab:
        print(f"{name} abstention: " + ", ".join(f"tau {r['tau']:.1f}: act {r['act_rate']:.2f}, success {r['success_overall']:.2f}"
                                                for r in ab[::3]))
report["analytic_group_multipliers"] = models[0].multipliers
report["mlp_epochs"] = models[1].epochs_run
report["mlp_temperature"] = models[1].T
report["analytic_platt"] = [models[0].a, models[0].b]
write_report(os.path.join(out_dir, f"baselines_{args.label}_{args.features}.json"), report)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

colors = {"analytic": "#2a78d6", "mlp": "#eb6834"}
fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.2, 6.0), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
                              facecolor="#fcfcfb")
for a in (ax, axh):
    a.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color("#8a8984")
    a.tick_params(colors="#52514e", labelsize=9)
    a.grid(color="#e6e5e0", linewidth=0.6)
ax.plot([0, 1], [0, 1], color="#8a8984", linewidth=1.0, linestyle="--", label="perfect calibration")
width = 0.04
for i, name in enumerate(("analytic", "mlp")):
    rel = report["models"][name].get("reliability", [])
    if not rel:
        continue
    p = [r["p_mean"] for r in rel]
    q = [r["y_mean"] for r in rel]
    ax.plot(p, q, color=colors[name], linewidth=1.5, marker="o", markersize=5,
            label=f"{name} (ECE {report['models'][name]['ece']:.3f})")
    axh.bar(np.array([(r['bin'] + 0.5) / 10 for r in rel]) + (i - 0.5) * width, [r["count"] for r in rel], width=width,
            color=colors[name], edgecolor="#fcfcfb", linewidth=1.0)
ax.set_ylabel("observed success rate", color="#0b0b0b", fontsize=10)
ax.set_title(f"Reliability on test scenes ({args.label}, {args.features})", color="#0b0b0b", fontsize=10, loc="left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=9, labelcolor="#0b0b0b")
axh.set_xlabel("predicted success probability", color="#0b0b0b", fontsize=10)
axh.set_ylabel("rollouts", color="#0b0b0b", fontsize=10)
fig.tight_layout()
png = os.path.join(out_dir, f"reliability_{args.label}_{args.features}.png")
fig.savefig(png, dpi=150)
print(f"[plot] {png}")
