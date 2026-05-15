"""P-004 PatternVisualizer — Pattern-Cluster v2

训练末尾自动产出:
- cluster_prototypes.png         — 各 cluster 原型的归一化 close 曲线叠加图
- cluster_year_distribution.png  — 各 cluster 样本年度分布
- cluster_feature_importance.png — 方案 A 各 cluster top-10 特征对比
- calibration_per_cluster.png    — 方案 A 校准曲线
- calibration_single_with_cf.png — 方案 B 校准曲线

matplotlib 可选: 若导入失败, 退化为只写 CSV/JSON, 不画图.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_prototypes(
    medoid_sequences: List[np.ndarray],
    out_dir: Path,
    channel_idx: int = 0,
    channel_name: str = "close/first - 1",
) -> Optional[Path]:
    """画各 cluster 原型的归一化 close 曲线."""
    out_dir = _ensure_dir(Path(out_dir))
    # CSV 总是出
    rows = []
    for cid, seq in enumerate(medoid_sequences):
        for t, v in enumerate(seq[:, channel_idx]):
            rows.append({"cluster_id": cid, "t": t, "value": float(v)})
    csv_path = out_dir / "cluster_prototypes.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    if not _HAS_MPL:
        logger.info(f"matplotlib 不可用, 仅写 {csv_path.name}")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for cid, seq in enumerate(medoid_sequences):
        ax.plot(seq[:, channel_idx], label=f"cluster {cid}", linewidth=2)
    ax.set_xlabel("回看天数 (T-19 → T)")
    ax.set_ylabel(channel_name)
    ax.set_title("形态原型 (各 cluster medoid)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = out_dir / "cluster_prototypes.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_cluster_year_distribution(
    samples_with_cluster: pd.DataFrame,
    out_dir: Path,
    date_col: str = "potential_date",
    cluster_col: str = "cluster_id",
) -> Optional[Path]:
    """画各 cluster 按年的样本数分布."""
    out_dir = _ensure_dir(Path(out_dir))
    df = samples_with_cluster.copy()
    df = df[df[cluster_col] >= 0].copy()
    df["yr"] = pd.to_datetime(df[date_col]).dt.year
    pivot = df.pivot_table(index="yr", columns=cluster_col, values="sample_id" if "sample_id" in df.columns else df.columns[0], aggfunc="count").fillna(0)
    pivot.to_csv(out_dir / "cluster_year_distribution.csv", encoding="utf-8-sig")

    if not _HAS_MPL or pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", stacked=False, ax=ax)
    ax.set_ylabel("样本数")
    ax.set_xlabel("年份")
    ax.set_title("各 cluster 样本年度分布")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png = out_dir / "cluster_year_distribution.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_feature_importance_per_cluster(
    importance_per_cluster: Dict[int, pd.DataFrame],
    out_dir: Path,
    top_n: int = 10,
) -> Optional[Path]:
    """画方案 A 各 cluster 前 top_n 特征的横向对比."""
    out_dir = _ensure_dir(Path(out_dir))
    if not importance_per_cluster:
        return None
    all_feats = set()
    for imp in importance_per_cluster.values():
        if imp is None or imp.empty:
            continue
        all_feats.update(imp.head(top_n)["feature"].tolist())
    if not all_feats:
        return None
    rows = []
    for cid, imp in importance_per_cluster.items():
        if imp is None or imp.empty:
            continue
        s = imp.set_index("feature")["importance"]
        for f in all_feats:
            rows.append({"cluster_id": cid, "feature": f, "importance": float(s.get(f, 0.0))})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="feature", columns="cluster_id", values="importance").fillna(0)
    pivot.to_csv(out_dir / "cluster_feature_importance.csv", encoding="utf-8-sig")

    if not _HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(pivot))))
    pivot.plot(kind="barh", ax=ax)
    ax.set_xlabel("LightGBM importance")
    ax.set_title(f"各 cluster top-{top_n} 特征对比")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out_png = out_dir / "cluster_feature_importance.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
    title: str = "Calibration",
) -> Optional[Path]:
    """画单个模型的校准曲线."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, 1, n_bins + 1)
    digitized = np.digitize(y_proba, bins) - 1
    digitized = np.clip(digitized, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = digitized == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin_low": bins[b], "bin_high": bins[b + 1],
            "mean_pred": float(y_proba[m].mean()),
            "frac_pos": float(y_true[m].mean()),
            "n": int(m.sum()),
        })
    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    if not _HAS_MPL or cal_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.plot(cal_df["mean_pred"], cal_df["frac_pos"], marker="o", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=120)
    plt.close(fig)
    return out_path.with_suffix(".png")
