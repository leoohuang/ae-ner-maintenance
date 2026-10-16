import collections, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.stats import aulc, budget_to_reach

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in open(ROOT / "results/random.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    by[r["frac"]].append(r)
fracs = sorted(by)
seeds = sorted({r["seed"] for r in rows})
print(f"行数 {len(rows)} | 预算点 {len(fracs)} | 种子 {len(seeds)}")
assert all(len(by[f]) == len(seeds) for f in fracs), "有预算点的种子数不齐"
assert all(r["tokens_used"] <= r["budget"] for r in rows)
print("实际 token 与预算的最大差距:", max(r["budget"] - r["tokens_used"] for r in rows))

budget = {f: by[f][0]["budget"] for f in fracs}
total = budget[1.0]


def col(f, key):
    return np.array([r[key] for r in sorted(by[f], key=lambda r: r["seed"])])


M = np.array([col(f, "test_micro_f1") for f in fracs])    # 行：预算点，列：种子
V = np.array([col(f, "val_micro_f1") for f in fracs])
n_seeds = M.shape[1]

print("\n预算 | token | 记录数 | 测试集 F1 均值 ± std | 验证集 F1 均值 ± std")
for k, f in enumerate(fracs):
    print(f"{f:4.0%} | {budget[f]:5d} | {col(f, 'n_records').mean():6.1f} | "
          f"{M[k].mean():.4f} ± {M[k].std(ddof=1):.4f} | {V[k].mean():.4f} ± {V[k].std(ddof=1):.4f}")

ref = M[-1].mean()
x = [budget[f] for f in fracs]
mean_curve = M.mean(axis=1)
print(f"\n满分基准（100% 点，测试集 F1 的 {n_seeds} 个种子均值）: {ref:.4f}")

print("\n达到满分基准的某个比例所需的数据（占训练池 token 的 %）")
for p in (0.90, 0.95, 0.98):
    tgt = p * ref
    b = budget_to_reach(x, mean_curve, tgt)
    per = [budget_to_reach(x, M[:, s], tgt) for s in range(n_seeds)]
    ok = [v for v in per if v is not None]
    med = f"{100 * np.median(ok) / total:.1f}%" if ok else "—"
    b_txt = f"{100 * b / total:.1f}%" if b is not None else "未达到"
    print(f"{p:.0%} (F1 {tgt:.4f}): 平均曲线 {b_txt} | 单个种子达到 {len(ok)}/{n_seeds}，中位数 {med}")

sel = [k for k, f in enumerate(fracs) if f <= 0.60]
a = [aulc([x[k] for k in sel], M[sel, s]) for s in range(n_seeds)]
print(f"\nAULC（5%–60%，按 token 归一化）: {np.mean(a):.4f} ± {np.std(a, ddof=1):.4f}")

xp = [100 * f for f in fracs]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for s in range(n_seeds):
    ax.plot(xp, M[:, s], color="0.8", lw=0.8, label="single seed" if s == 0 else None)
ax.plot(xp, mean_curve, "o-", lw=2, label="random (mean over seeds)")
sd = M.std(axis=1, ddof=1)
ax.fill_between(xp, mean_curve - sd, mean_curve + sd, alpha=0.2, label="+/- 1 std")
for p in (0.90, 0.95, 0.98):
    ax.axhline(p * ref, ls=":", color="gray", lw=0.8)
    ax.text(101, p * ref, f"{p:.0%}", va="center", fontsize=8, color="gray")
ax.axhline(ref, ls="--", color="k", lw=0.8, label="full-data reference")
ax.set_xlim(2, 108)
ax.set_xticks(xp)
ax.set_xticklabels([f"{v:g}" for v in xp], fontsize=8)
ax.set_xlabel("labelled tokens (% of train pool)")
ax.set_ylabel("test micro-F1")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "results/random_curve.png", dpi=150)
print("\n图已保存: results/random_curve.png")
