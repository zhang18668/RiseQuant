"""M-004 模型包 (ModelBundle) — Pattern-Cluster v2

把一次训练产出的全部"可重用资产"打包成一个目录, 让回测脚本能反复 load:

::

    bundle_dir/
        bundle.json                         ← 元信息 + 特征列 + cluster 配置
        prototypes.pkl                      ← {cluster_id: ndarray (T, C)}
        router.pkl                          ← PatternRouter
        cluster_stats.csv                   ← 聚类后的统计 (年份分布、样本数等)
        models/
            cluster_0/model.pkl
            cluster_0/train_metrics.json
            ...
            single_with_cf/model.pkl        ← 方案 B
            single_with_cf/train_metrics.json
        usability_flags.json                ← 哪些 cluster 可用

Bundle 类型 (``bundle_type``):
    - "per_cluster"      — N 个独立 LightGBM (方案 A)
    - "single_with_cf"   — 1 个 LightGBM + cluster_id 类别特征 (方案 B)

一次训练可同时保存两种 bundle, 由 ``BundleArchive`` 写入同一 run 目录.
"""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.model.model_trainer import LimitUpModelTrainer
from src.pattern.pattern_router import PatternRouter
from src.utils.artifact_schema import write_training_manifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _json_default(o: Any):
    if isinstance(o, (datetime, pd.Timestamp)):
        return pd.Timestamp(o).isoformat()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =======================================================================
@dataclass
class BundleArchive:
    """形态聚类多模型 bundle 的写入/读取归档."""

    base_dir: str
    bundle_name: str = "pattern_cluster"
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.base_dir) / self.bundle_name / f"run_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir = Path(self.base_dir) / self.bundle_name / "latest"

    # ------------------------------------------------------------------
    # 共享层
    # ------------------------------------------------------------------
    def save_shared(
        self,
        feature_cols: List[str],
        router: PatternRouter,
        cluster_stats: Optional[pd.DataFrame] = None,
        training_config: Optional[Dict[str, Any]] = None,
        training_data_range: Optional[List[str]] = None,
        cluster_sizes: Optional[Dict[int, int]] = None,
        dist_thresholds: Optional[Dict[int, float]] = None,
    ) -> None:
        shared_dir = self.run_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)

        # router 与原型
        router.save(shared_dir / "router.pkl")
        with open(shared_dir / "prototypes.pkl", "wb") as f:
            pickle.dump({int(k): np.asarray(v) for k, v in router.prototypes.items()}, f)

        # cluster_stats
        if cluster_stats is not None and len(cluster_stats):
            cluster_stats.to_csv(shared_dir / "cluster_stats.csv",
                                 index=False, encoding="utf-8-sig")

        # shared meta
        meta = {
            "bundle_name": f"{self.bundle_name}_run_{self.run_id}",
            "created_at": datetime.now().isoformat(),
            "training_data_range": training_data_range or [],
            "feature_cols": list(feature_cols),
            "n_clusters": len(router.prototypes),
            "cluster_sizes": {int(k): int(v) for k, v in (cluster_sizes or {}).items()},
            "cluster_dist_thresholds": {
                int(k): float(v) for k, v in (dist_thresholds or router.dist_thresholds).items()
            },
            "training_config": training_config or {},
        }
        with open(shared_dir / "shared_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # 方案 A: per_cluster
    # ------------------------------------------------------------------
    def save_per_cluster(
        self,
        trainers: Dict[int, LimitUpModelTrainer],
        metrics_per_cluster: Dict[int, Dict[str, Any]],
        feature_importance_per_cluster: Optional[Dict[int, pd.DataFrame]] = None,
        usability_flags: Optional[Dict[int, bool]] = None,
        default_min_score_per_cluster: Optional[Dict[int, float]] = None,
    ) -> None:
        bundle_dir = self.run_dir / "bundle_per_cluster"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        models_dir = bundle_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: List[dict] = []
        for cid, trainer in trainers.items():
            cdir = models_dir / f"cluster_{int(cid)}"
            cdir.mkdir(parents=True, exist_ok=True)
            trainer.save(str(cdir / "model.pkl"))
            mt = metrics_per_cluster.get(int(cid), {})
            with open(cdir / "train_metrics.json", "w", encoding="utf-8") as f:
                json.dump(mt, f, ensure_ascii=False, indent=2, default=_json_default)
            if feature_importance_per_cluster and int(cid) in feature_importance_per_cluster:
                imp = feature_importance_per_cluster[int(cid)]
                imp.to_csv(cdir / "feature_importance.csv",
                           index=False, encoding="utf-8-sig")
            summary_rows.append({
                "cluster_id": int(cid),
                **{k: v for k, v in mt.items() if isinstance(v, (int, float, str, bool)) or v is None},
            })

        # usability
        usability = {int(k): bool(v) for k, v in (usability_flags or {}).items()}
        with open(bundle_dir / "usability_flags.json", "w", encoding="utf-8") as f:
            json.dump(usability, f, ensure_ascii=False, indent=2)

        # 默认 min_score per cluster
        if default_min_score_per_cluster:
            with open(bundle_dir / "default_min_score.json", "w", encoding="utf-8") as f:
                json.dump(
                    {int(k): float(v) for k, v in default_min_score_per_cluster.items()},
                    f, ensure_ascii=False, indent=2,
                )

        bundle_meta = {
            "bundle_type": "per_cluster",
            "n_models": len(trainers),
            "usable_clusters": [c for c, ok in usability.items() if ok],
            "summary_rows": summary_rows,
        }
        with open(bundle_dir / "bundle_meta.json", "w", encoding="utf-8") as f:
            json.dump(bundle_meta, f, ensure_ascii=False, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # 方案 B: single_with_cf
    # ------------------------------------------------------------------
    def save_single_with_cf(
        self,
        trainer: LimitUpModelTrainer,
        metrics: Dict[str, Any],
        feature_importance: Optional[pd.DataFrame] = None,
        categorical_feature: Optional[List[str]] = None,
    ) -> None:
        bundle_dir = self.run_dir / "bundle_single_with_cf"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        trainer.save(str(bundle_dir / "model.pkl"))
        with open(bundle_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)
        if feature_importance is not None:
            feature_importance.to_csv(bundle_dir / "feature_importance.csv",
                                      index=False, encoding="utf-8-sig")
        bundle_meta = {
            "bundle_type": "single_with_cf",
            "categorical_feature": list(categorical_feature or ["cluster_id"]),
        }
        with open(bundle_dir / "bundle_meta.json", "w", encoding="utf-8") as f:
            json.dump(bundle_meta, f, ensure_ascii=False, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # 训练用样本 / 特征 (可选, Q6 倾向: 存)
    # ------------------------------------------------------------------
    def save_training_data(
        self,
        samples_with_cluster: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None,
    ) -> None:
        shared_dir = self.run_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)
        if samples_with_cluster is not None and len(samples_with_cluster):
            # parquet 可选 — 若无 pyarrow 则 fallback 到 csv
            self._save_table(samples_with_cluster, shared_dir / "samples_with_cluster")
        if features is not None and len(features):
            self._save_table(features, shared_dir / "features")

    @staticmethod
    def _save_table(df: pd.DataFrame, stem: Path) -> None:
        try:
            df.to_parquet(stem.with_suffix(".parquet"), index=False)
        except Exception:
            df.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 训练总结 (含两种 bundle 的对比)
    # ------------------------------------------------------------------
    def save_training_summary(self, summary: Dict[str, Any]) -> None:
        with open(self.run_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
        bundle_types = []
        if (self.run_dir / "bundle_per_cluster").exists():
            bundle_types.append("per_cluster")
        if (self.run_dir / "bundle_single_with_cf").exists():
            bundle_types.append("single_with_cf")
        write_training_manifest(
            self.run_dir,
            strategy=self.bundle_name or "pattern_cluster",
            run_id=str(self.run_id),
            bundle_types=bundle_types,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # latest symlink/copy
    # ------------------------------------------------------------------
    def update_latest(self) -> None:
        try:
            if self.latest_dir.exists() or self.latest_dir.is_symlink():
                if self.latest_dir.is_symlink() or self.latest_dir.is_file():
                    self.latest_dir.unlink()
                else:
                    shutil.rmtree(self.latest_dir)
            # 优先 symlink, Windows 无权限则 fallback 到全量拷贝
            try:
                self.latest_dir.symlink_to(self.run_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                shutil.copytree(self.run_dir, self.latest_dir)
        except Exception as e:
            logger.warning(f"update_latest 失败: {e}")


# =======================================================================
@dataclass
class ModelBundle:
    """加载后的 bundle, 给回测脚本用."""

    bundle_dir: Path
    bundle_type: str                          # "per_cluster" | "single_with_cf"
    feature_cols: List[str]
    router: PatternRouter
    shared_meta: Dict[str, Any]
    # per_cluster 用
    trainers_per_cluster: Optional[Dict[int, LimitUpModelTrainer]] = None
    usability_flags: Optional[Dict[int, bool]] = None
    default_min_score_per_cluster: Optional[Dict[int, float]] = None
    # single_with_cf 用
    trainer_single: Optional[LimitUpModelTrainer] = None
    categorical_feature: Optional[List[str]] = None

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, bundle_dir, bundle_type: str = "per_cluster") -> "ModelBundle":
        bundle_dir = Path(bundle_dir)
        shared_dir = bundle_dir / "shared"
        if not shared_dir.exists():
            raise FileNotFoundError(f"shared/ not found under {bundle_dir}")
        with open(shared_dir / "shared_meta.json", encoding="utf-8") as f:
            shared_meta = json.load(f)
        router = PatternRouter.load(shared_dir / "router.pkl")

        kwargs: Dict[str, Any] = {
            "bundle_dir": bundle_dir,
            "bundle_type": bundle_type,
            "feature_cols": list(shared_meta.get("feature_cols", [])),
            "router": router,
            "shared_meta": shared_meta,
        }

        if bundle_type == "per_cluster":
            pcb = bundle_dir / "bundle_per_cluster"
            if not pcb.exists():
                raise FileNotFoundError(f"bundle_per_cluster/ not found under {bundle_dir}")
            trainers: Dict[int, LimitUpModelTrainer] = {}
            for sub in (pcb / "models").iterdir():
                if not sub.is_dir() or not sub.name.startswith("cluster_"):
                    continue
                cid = int(sub.name.split("_")[1])
                model_pkl = sub / "model.pkl"
                if model_pkl.exists():
                    trainers[cid] = LimitUpModelTrainer.load(str(model_pkl))
            kwargs["trainers_per_cluster"] = trainers
            uf = pcb / "usability_flags.json"
            if uf.exists():
                with open(uf, encoding="utf-8") as f:
                    kwargs["usability_flags"] = {int(k): bool(v) for k, v in json.load(f).items()}
            df = pcb / "default_min_score.json"
            if df.exists():
                with open(df, encoding="utf-8") as f:
                    kwargs["default_min_score_per_cluster"] = {
                        int(k): float(v) for k, v in json.load(f).items()
                    }

        elif bundle_type == "single_with_cf":
            sb = bundle_dir / "bundle_single_with_cf"
            if not sb.exists():
                raise FileNotFoundError(f"bundle_single_with_cf/ not found under {bundle_dir}")
            kwargs["trainer_single"] = LimitUpModelTrainer.load(str(sb / "model.pkl"))
            meta_f = sb / "bundle_meta.json"
            if meta_f.exists():
                with open(meta_f, encoding="utf-8") as f:
                    bm = json.load(f)
                kwargs["categorical_feature"] = list(bm.get("categorical_feature", ["cluster_id"]))
        else:
            raise ValueError(f"unknown bundle_type: {bundle_type}")

        return cls(**kwargs)

    # ------------------------------------------------------------------
    def predict(
        self,
        features_df: pd.DataFrame,
        cluster_ids: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """对一批样本预测正样本概率 ``P(label=1)``.

        Parameters
        ----------
        features_df : 含 ``feature_cols`` 的全部列 (顺序无关, reindex 内部对齐).
        cluster_ids : per_cluster 必传, single_with_cf 可选 (会作为 cluster_id 特征).

        Returns
        -------
        ndarray (n,) — 正样本概率. 不可用 cluster / cluster_id = -1 / 缺模型 一律返回 0.0.
        """
        n = len(features_df)
        scores = np.zeros(n, dtype=np.float64)

        if self.bundle_type == "per_cluster":
            if cluster_ids is None:
                raise ValueError("per_cluster bundle 必须传 cluster_ids")
            assert self.trainers_per_cluster is not None
            for cid, trainer in self.trainers_per_cluster.items():
                if self.usability_flags is not None and not self.usability_flags.get(int(cid), True):
                    continue
                mask = (cluster_ids.values == int(cid))
                if not mask.any():
                    continue
                X = features_df.loc[mask].reindex(columns=self.feature_cols, fill_value=0).fillna(0)
                proba = trainer.predict_proba(X)
                p_pos = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
                scores[mask] = p_pos
        elif self.bundle_type == "single_with_cf":
            assert self.trainer_single is not None
            X = features_df.reindex(columns=self.feature_cols, fill_value=0).fillna(0)
            if cluster_ids is not None and "cluster_id" in self.feature_cols:
                X["cluster_id"] = cluster_ids.values
            proba = self.trainer_single.predict_proba(X)
            scores = proba[:, 1] if proba.shape[1] >= 2 else proba[:, 0]
        else:
            raise ValueError(f"unknown bundle_type: {self.bundle_type}")

        return scores
