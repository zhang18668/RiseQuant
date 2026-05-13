"""DS-007 模型版本注册表

管理模型版本的保存、加载和列表功能。
每个版本包含: 模型文件、元数据、回测结果。

用法:
    registry = ModelRegistry("models")
    version = registry.save_version(
        model=trainer,
        meta={...},
        metrics={...},
        config={...},
    )
    registry.list_versions()
    info = registry.load_version("v001")
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ModelRegistry:
    """模型版本注册表"""

    REGISTRY_FILE = "registry.json"

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.registry_path = self.models_dir / self.REGISTRY_FILE
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        """确保 models 目录和注册表存在"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save_registry({"versions": [], "latest_version": None})

    def _load_registry(self) -> Dict[str, Any]:
        """加载注册表"""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """保存注册表"""
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def _next_version(self) -> str:
        """生成下一个版本号"""
        registry = self._load_registry()
        versions = registry.get("versions", [])
        if not versions:
            return "v001"
        last_version = versions[-1]["version"]
        # 解析 vXXX 并递增
        num = int(last_version[1:]) + 1
        return f"v{num:03d}"

    def save_version(
        self,
        model: Any,
        meta: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        model_filename: str = "limit_up_lgbm.pkl",
    ) -> str:
        """
        保存新版本

        Args:
            model: 模型对象（需要有 save 方法）
            meta: 训练元信息（如 n_train, n_valid, feature_cols 等）
            metrics: 回测指标
            config: 训练时的完整配置快照
            model_filename: 模型文件名

        Returns:
            版本号字符串，如 "v001"
        """
        version = self._next_version()
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        model_path = version_dir / model_filename
        model.save(str(model_path))

        # 保存 meta.json
        meta_path = version_dir / "meta.json"
        meta_data = {
            **meta,
            "version": version,
            "trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        # 保存回测结果
        if metrics is not None:
            bt_path = version_dir / "backtest_results.json"
            bt_data = {
                "version": version,
                "trained_at": meta_data["trained_at"],
                "metrics": metrics,
            }
            with open(bt_path, "w", encoding="utf-8") as f:
                json.dump(bt_data, f, ensure_ascii=False, indent=2)

        # 保存配置快照
        if config is not None:
            config_path = version_dir / "config_snapshot.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

        # 更新注册表
        registry = self._load_registry()
        version_entry = {
            "version": version,
            "trained_at": meta_data["trained_at"],
            "model_file": str(model_path.relative_to(self.models_dir)),
            "meta_file": str(meta_path.relative_to(self.models_dir)),
            "backtest_file": str((version_dir / "backtest_results.json").relative_to(self.models_dir)) if metrics else None,
            "config_file": str((version_dir / "config_snapshot.json").relative_to(self.models_dir)) if config else None,
            "n_train": meta.get("n_train"),
            "n_valid": meta.get("n_valid"),
            "n_test": meta.get("n_test"),
            "train_range": meta.get("train_range"),
            "test_range": meta.get("test_range"),
            "metrics": metrics,
        }
        registry["versions"].append(version_entry)
        registry["latest_version"] = version
        self._save_registry(registry)

        return version

    def load_version(self, version: str) -> Optional[Dict[str, Any]]:
        """
        加载指定版本的信息

        Args:
            version: 版本号，如 "v001"

        Returns:
            版本信息字典，包含 model_path, meta, metrics 等
        """
        registry = self._load_registry()
        for v in registry["versions"]:
            if v["version"] == version:
                result = v.copy()
                # 返回完整路径
                result["model_path"] = str(self.models_dir / v["model_file"])
                if v.get("backtest_file"):
                    bt_path = self.models_dir / v["backtest_file"]
                    with open(bt_path, "r", encoding="utf-8") as f:
                        result["backtest_data"] = json.load(f)
                if v.get("config_file"):
                    config_path = self.models_dir / v["config_file"]
                    with open(config_path, "r", encoding="utf-8") as f:
                        result["config"] = json.load(f)
                meta_path = self.models_dir / v["meta_file"]
                with open(meta_path, "r", encoding="utf-8") as f:
                    result["meta"] = json.load(f)
                return result
        return None

    # ------------------------------------------------------------------
    def update_version_with_backtest(
        self,
        version: str,
        backtester=None,
        metrics: Optional[Dict[str, Any]] = None,
        signals_df=None,
    ) -> Optional[Path]:
        """对已有 version 补充写入回测产物.

        - 落盘文件:
            backtest_results.json    回测核心指标 (含卖出原因分布、盈亏分布)
            equity_curve.csv         每日净值
            trades.csv               交易明细 (含 SELL_STOP_DYNAMIC / SELL_TIME_STOP 等)
            signals.csv (可选)       预测分数

        - 调用方式 (任选其一):
            update_version_with_backtest("v003", backtester=bt)
            update_version_with_backtest("v003", metrics={...})
        """
        import pandas as pd

        version_dir = self.models_dir / version
        if not version_dir.exists():
            return None

        bt_metrics: Dict[str, Any] = dict(metrics or {})
        trades = None
        equity = None

        if backtester is not None:
            bt_metrics = {**bt_metrics, **backtester.get_metrics()}
            trades = backtester.get_trades()
            equity = backtester.equity_curve.copy() if backtester.equity_curve is not None else None

            # 卖出原因分布
            if trades is not None and len(trades):
                r = trades[trades["action"].astype(str).str.startswith("SELL")]
                bt_metrics["sell_reasons"] = r["action"].value_counts().to_dict()
                if len(r):
                    bt_metrics["sell_avg_return"]    = float(r["return_pct"].mean())
                    bt_metrics["sell_median_return"] = float(r["return_pct"].median())
                    bt_metrics["sell_max_loss"]      = float(r["return_pct"].min())
                    bt_metrics["sell_max_gain"]      = float(r["return_pct"].max())

        # 写 backtest_results.json
        bt_path = version_dir / "backtest_results.json"
        bt_data = {
            "version": version,
            "trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "metrics": bt_metrics,
        }
        with open(bt_path, "w", encoding="utf-8") as f:
            json.dump(bt_data, f, ensure_ascii=False, indent=2, default=str)

        # equity / trades / signals
        if equity is not None and len(equity):
            equity.to_csv(version_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        if trades is not None and len(trades):
            trades.to_csv(version_dir / "trades.csv", index=False, encoding="utf-8-sig")
        if signals_df is not None and len(signals_df):
            signals_df.to_csv(version_dir / "signals.csv", index=False, encoding="utf-8-sig")

        # 更新注册表里的 metrics 字段, 方便 list_versions 显示
        registry = self._load_registry()
        for v in registry.get("versions", []):
            if v["version"] == version:
                v["metrics"] = bt_metrics
                v["backtest_file"] = str(bt_path.relative_to(self.models_dir))
                break
        self._save_registry(registry)
        return version_dir

    def list_versions(self) -> None:
        """打印所有版本列表"""
        registry = self._load_registry()
        versions = registry.get("versions", [])
        if not versions:
            print("暂无保存的版本")
            return
        print("\n" + "=" * 70)
        print(f"{'版本':<8} {'训练时间':<20} {'样本数(train)':<12} {'收益率':<10} {'夏普比率':<10} {'最大回撤':<10}")
        print("-" * 70)
        for v in versions:
            metrics = v.get("metrics") or {}
            total_return = metrics.get("total_return", 0)
            sharpe = metrics.get("sharpe_ratio", 0)
            max_dd = metrics.get("max_drawdown", 0)
            n_train = v.get("n_train") or "N/A"
            print(
                f"{v['version']:<8} {v['trained_at']:<20} {n_train:<12} "
                f"{total_return:>8.2%}   {sharpe:>8.2f}   {max_dd:>8.2%}"
            )
        print("=" * 70)
        print(f"最新版本: {registry.get('latest_version', 'N/A')}")
        print()

    def get_latest_version(self) -> Optional[str]:
        """获取最新版本号"""
        registry = self._load_registry()
        return registry.get("latest_version")

    def delete_version(self, version: str) -> bool:
        """
        删除指定版本

        Args:
            version: 版本号

        Returns:
            是否删除成功
        """
        registry = self._load_registry()
        versions = registry.get("versions", [])
        new_versions = [v for v in versions if v["version"] != version]
        if len(new_versions) == len(versions):
            return False

        # 删除文件
        version_dir = self.models_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # 更新注册表
        registry["versions"] = new_versions
        registry["latest_version"] = new_versions[-1]["version"] if new_versions else None
        self._save_registry(registry)
        return True
