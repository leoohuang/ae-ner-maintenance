import json
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FR = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
AULC_MAX = 0.60
NAMES = ["random", "uncertainty", "hybrid"]


def load(path):
    return [json.loads(line) for line in open(path)]


def table(rows):
    d = defaultdict(dict)
    for r in rows:
        d[r["seed"]][r["frac"]] = (r["budget"], r["test_micro_f1"])
    return d


T = {n: table(load("results/%s.jsonl" % n)) for n in NAMES}
R = T["random"]
seeds = sorted(set.intersection(*[set(T[n]) for n in NAMES]))


def curve(name, seed):
    xs, ys = [], []
    for f in FR:
        src = R if (name != "random" and f in (0.05, 1.00)) else T[name]
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


curves = {n: [curve(n, s) for s in seeds] for n in NAMES}
mean = {n: (np.mean([c[0] for c in cs], axis=0), np.mean([c[1] for c in cs], axis=0))
        for n, cs in curves.items()}
ref = mean["random"][1][-1]

print("seeds used:", seeds)
print("reference (random, 100%%): %.4f" % ref)
print()
print("budget  random   uncert   hybrid   hyb-unc  hyb-rand")
for i, f in enumerate(FR):
    r, u, h = (mean[n][1][i] for n in NAMES)
    print("%4.0f%%   %.4f   %.4f   %.4f   %+.4f   %+.4f" % (100 * f, r, u, h, h - u, h - r))
print()

aulcs = {n: np.array([aulc(c[0], c[1]) for c in cs]) for n, cs in curves.items()}
for n in NAMES:
    print("AULC %-12s %.4f +- %.4f" % (n, aulcs[n].mean(), aulcs[n].std(ddof=1)))
print()
for a, b in (("uncertainty", "random"), ("hybrid", "random"), ("hybrid", "uncertainty")):
    d = aulcs[a] - aulcs[b]
    try:
        p = wilcoxon(aulcs[a], aulcs[b]).pvalue
    except ValueError:
        p = float("nan")
    print("AULC %s - %s: %+.4f, better in %d of %d seeds, Wilcoxon p = %.4f"
          % (a, b, d.mean(), int((d > 0).sum()), len(d), p))
print()

for p in (0.90, 0.95, 0.98):
    out = []
    for n in NAMES:
        xs, ys = mean[n]
        v = reach(xs, ys, p * ref)
        out.append("%s %s" % (n, "not reached" if v is None else "%.1f%%" % v))
    print("reach %d%% of reference: %s" % (round(100 * p), " | ".join(out)))

plt.figure(figsize=(7, 4.5))
for n in NAMES:
    xs, ys = mean[n]
    plt.plot(100 * xs / xs[-1], ys, marker="o", label=n)
plt.axhline(ref, color="grey", linestyle=":")
plt.xlabel("labelled text (% of training words)")
plt.ylabel("test micro-F1")
plt.legend()
plt.tight_layout()
plt.savefig("results/compare_all_curve.png", dpi=150)
print("saved results/compare_all_curve.png")