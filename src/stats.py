import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def aulc(tokens, f1):
    """归一化的学习曲线下面积，范围约 [0, 1]；比较策略时 x 范围必须一致"""
    x, y = np.asarray(tokens, float), np.asarray(f1, float)
    return float(_trapz(y, x) / (x[-1] - x[0]))


def budget_to_reach(tokens, f1, target):
    """线性插值：曲线首次达到 target 所需 token；达不到返回 None"""
    if f1[0] >= target:
        return tokens[0]
    for k in range(1, len(tokens)):
        if f1[k] >= target > f1[k - 1]:
            frac = (target - f1[k - 1]) / (f1[k] - f1[k - 1])
            return tokens[k - 1] + frac * (tokens[k] - tokens[k - 1])
    return None

def paired(a, b):
    """按种子配对比较 a 和 b（RQ3、RQ3b）：平均差 a - b、a 更好的种子数、Wilcoxon 双侧 p。
    所有差都为 0 时 p 为 nan。"""
    from scipy.stats import wilcoxon
    d = np.asarray(a, float) - np.asarray(b, float)
    p = float("nan") if not d.any() else float(wilcoxon(a, b).pvalue)
    return {"diff": float(d.mean()), "better": int((d > 0).sum()), "n": len(d), "p": p}
