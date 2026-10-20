import json
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FR = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
AULC_MAX = 0.60


def load(path):
    return [json.loads(line) for line in open(path)]


def table(rows):
    d = defaultdict(dict)
    for r in rows:
        d[r["seed"]][r["frac"]] = (r["budget"], r["test_micro_f1"])
    return d


R = table(load("results/random.jsonl"))
U = table(load("results/uncertainty.jsonl"))
seeds = sorted(set(R) & set(U))


def curve(tab, seed, shared_ends):
    xs, ys = [], []
    for f in FR:
        src = R if (shared_ends and f in (0.05, 1.00)) else tab
        b, v = src[seed][f]
        xs.append(b)
        ys.append(v)
    return np.array(xs, float), np.array(ys, float)


def aulc(xs, ys):
    pick = [i for i, f in enumerate(FR) if f <= AULC_MAX]
    x, y = xs[pick], ys[pick]
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def reach(xs, ys, target):
    for i in range(len(xs)):
        if ys[i] >= target:
            if i == 0:
                return 100 * xs[0] / xs[-1]
            frac = (target - ys[i - 1]) / (ys[i] - ys[i - 1])
            return 100 * (xs[i - 1] + frac * (xs[i] - xs[i - 1])) / xs[-1]
    return None


curves = {"random": [], "uncertainty": []}
for s in seeds:
    curves["random"].append(curve(R, s, False))
    curves["uncertainty"].append(curve(U, s, True))

mean = {}
for name, cs in curves.items():
    xs = np.mean([c[0] for c in cs], axis=0)
    ys = np.mean([c[1] for c in cs], axis=0)
    mean[name] = (xs, ys)

ref = mean["random"][1][-1]
print("seeds used:", seeds)
print("reference (random, 100%%): %.4f" % ref)
print()
print("budget  random   uncert   diff")
for i, f in enumerate(FR):
    r, u = mean["random"][1][i], mean["uncertainty"][1][i]
    print("%4.0f%%   %.4f   %.4f   %+.4f" % (100 * f, r, u, u - r))
print()

aulcs = {}
for name, cs in curves.items():
    aulcs[name] = np.array([aulc(c[0], c[1]) for c in cs])
    print("AULC %-12s %.4f +- %.4f" % (name, aulcs[name].mean(), aulcs[name].std(ddof=1)))
d = aulcs["uncertainty"] - aulcs["random"]
print("AULC diff (unc - random): %+.4f, better in %d of %d seeds" % (d.mean(), int((d > 0).sum()), len(d)))
try:
    print("Wilcoxon p = %.4f" % wilcoxon(aulcs["uncertainty"], aulcs["random"]).pvalue)
except ValueError as e:
    print("Wilcoxon not computed:", e)
print()

for p in (0.90, 0.95, 0.98):
    out = []
    for name in ("random", "uncertainty"):
        xs, ys = mean[name]
        v = reach(xs, ys, p * ref)
        out.append("%s %s" % (name, "not reached" if v is None else "%.1f%%" % v))
    print("reach %d%% of reference: %s" % (round(100 * p), " | ".join(out)))

plt.figure(figsize=(7, 4.5))
for name in ("random", "uncertainty"):
    xs, ys = mean[name]
    plt.plot(100 * xs / xs[-1], ys, marker="o", label=name)
plt.axhline(ref, color="grey", linestyle=":")
plt.xlabel("labelled text (% of training words)")
plt.ylabel("test micro-F1")
plt.legend()
plt.tight_layout()
plt.savefig("results/compare_curve.png", dpi=150)
print("saved results/compare_curve.png")