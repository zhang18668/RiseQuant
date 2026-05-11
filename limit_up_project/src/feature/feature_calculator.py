"""F-006 特征计算入口

把前面五个特征模块串起来，输出对每个事件样本一行的"特征宽表"：
``sample_id, code, event_date, f_xxx, ...``

调用方式
--------
::

    fc = FeatureCalculator(config)
    feats = fc.calculate(daily_data, events)

其中 ``events`` 至少需要 ``code, event_date`` 两列。
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from src.feature.limit_up_features import LimitUpFeatures
from src.feature.money_flow import MoneyFlowFeatures
from src.feature.pre_trend_features import PreTrendFeatures
from src.feature.price_volume import PriceVolumeFactors
from src.feature.sector_sentiment import SectorSentimentFeatures
from src.utils.logger import get_logger
from src.utils.validator import DataValidator

logger = get_logger(__name__)


class FeatureCalculator:
    """事件层 -> 特征层的整合器。"""

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        feat_cfg = cfg.get("feature", {}) if isinstance(cfg, dict) else {}
        self.config = cfg
        self.price_volume = PriceVolumeFactors()
        self.limit_up = LimitUpFeatures(
            threshold=cfg.get("event", {}).get("limit_up_threshold", 9.9) if isinstance(cfg, dict) else 9.9
        )
        self.sector = SectorSentimentFeatures()
        self.money_flow = MoneyFlowFeatures()
        self.pre_trend = PreTrendFeatures(
            window_1m=feat_cfg.get("pre_trend_window_1m", 20),
            window_2m=feat_cfg.get("pre_trend_window_2m", 40),
        )
        self._pv_cfg = {
            "returns_n": feat_cfg.get("returns_n", [1, 3, 5, 10, 20]),
            "volume_n": feat_cfg.get("volume_rolling_n", [5, 10]),
            "ma_n": feat_cfg.get("ma_n", [5, 10, 20]),
            "volatility_n": feat_cfg.get("volatility_n", [5, 10, 20]),
        }

    # ------------------------------------------------------------------
    def calculate(
        self,
        daily_data: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """按事件批量提特征。

        Parameters
        ----------
        daily_data : DataFrame
            列 ``date, code, open, high, low, close, volume, change_pct`` (turnover 可选)。
        events : DataFrame
            至少包含 ``code, event_date`` 两列。允许其他列被一并保留。
        """
        DataValidator.check_required_columns(
            daily_data, ["date", "code", "open", "high", "low", "close", "volume"],
            raise_error=True,
        )
        DataValidator.check_required_columns(events, ["code", "event_date"], raise_error=True)

        if events.empty:
            return pd.DataFrame()

        rows: List[pd.Series] = []
        # 仅遍历事件涉及到的股票，避免计算量爆炸
        codes_needed = events["code"].unique().tolist()
        daily_indexed = {
            code: g.sort_values("date").reset_index(drop=True)
            for code, g in daily_data[daily_data["code"].isin(codes_needed)].groupby("code")
        }

        for _, ev in events.iterrows():
            code = ev["code"]
            ev_date = ev["event_date"]
            sub = daily_indexed.get(code)
            if sub is None:
                continue
            feats = self.pre_trend.calculate_at_event(code, ev_date, sub)
            rows.append(feats)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).reset_index(drop=True)
        df = events.merge(df, on=["code", "event_date"], how="left")
        return df

    # ------------------------------------------------------------------
    def calculate_at_event(
        self,
        code: str,
        event_date,
        daily_data: pd.DataFrame,
    ) -> pd.Series:
        return self.pre_trend.calculate_at_event(code, event_date, daily_data)
