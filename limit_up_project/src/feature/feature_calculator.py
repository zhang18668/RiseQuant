"""F-006 特征计算入口

整合各特征模块, 输出对每个事件样本一行的"特征宽表":
``sample_id, code, event_date, f_xxx, ...``

集成的模块:
- PreTrendFeatures      F-005  涨停前形态
- TechnicalIndicators   F-007  MA/EMA/MACD/RSI/BOLL/KDJ/ATR
- MarketFeatures        F-008  大盘点位/趋势/量能 (横切到所有事件)

调用方式
--------
::

    fc = FeatureCalculator(config)
    feats = fc.calculate(daily_data, events, market_data=None)

其中 ``events`` 至少需要 ``code, event_date`` 两列.
``market_data`` (可选) 若不传, 会用 ``market_proxy_from_daily(daily_data)`` 自动合成.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from src.feature.limit_up_features import LimitUpFeatures
from src.feature.market_features import MarketFeatures, market_proxy_from_daily
from src.feature.money_flow import MoneyFlowFeatures
from src.feature.pre_trend_features import PreTrendFeatures
from src.feature.price_volume import PriceVolumeFactors
from src.feature.sector_sentiment import SectorSentimentFeatures
from src.feature.technical_indicators import TechnicalIndicators
from src.utils.logger import get_logger
from src.utils.validator import DataValidator

logger = get_logger(__name__)


class FeatureCalculator:
    """事件层 -> 特征层的整合器."""

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
        self.technical = TechnicalIndicators()
        self.market = MarketFeatures()
        # 开关 (默认全部开)
        self.use_technical = feat_cfg.get("use_technical", True)
        self.use_market = feat_cfg.get("use_market", True)

    # ------------------------------------------------------------------
    def calculate(
        self,
        daily_data: pd.DataFrame,
        events: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """按事件批量提特征.

        Parameters
        ----------
        daily_data : DataFrame
            列 ``date, code, open, high, low, close, volume, change_pct``.
        events : DataFrame
            至少 ``code, event_date``.
        market_data : DataFrame, optional
            列 ``date, close, volume``. 缺省时从 daily_data 合成代理.
        """
        DataValidator.check_required_columns(
            daily_data, ["date", "code", "open", "high", "low", "close", "volume"],
            raise_error=True,
        )
        DataValidator.check_required_columns(events, ["code", "event_date"], raise_error=True)
        if events.empty:
            return pd.DataFrame()

        codes_needed = events["code"].unique().tolist()
        daily_indexed = {
            code: g.sort_values("date").reset_index(drop=True)
            for code, g in daily_data[daily_data["code"].isin(codes_needed)].groupby("code")
        }

        # 1) per-event 特征: PreTrend + Technical
        rows: List[pd.Series] = []
        for _, ev in events.iterrows():
            code = ev["code"]
            ev_date = ev["event_date"]
            sub = daily_indexed.get(code)
            if sub is None:
                continue
            pt = self.pre_trend.calculate_at_event(code, ev_date, sub).to_dict()
            merged = dict(pt)
            if self.use_technical:
                ti = self.technical.calculate_at_event(code, ev_date, sub).to_dict()
                # 删除重复的 code/event_date
                ti.pop("code", None)
                ti.pop("event_date", None)
                merged.update(ti)
            rows.append(pd.Series(merged))

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).reset_index(drop=True)
        df = events.merge(df, on=["code", "event_date"], how="left")

        # 2) 大盘特征 (按日期横切, 然后 join 回事件)
        if self.use_market:
            if market_data is None:
                logger.info("MarketFeatures: 用全市场代理 (close=median, volume=sum)")
                market_data = market_proxy_from_daily(daily_data)
            mkt_df = self.market.batch_for_events(df, market_data, event_date_col="event_date")
            if not mkt_df.empty:
                df["event_date"] = pd.to_datetime(df["event_date"])
                df = df.merge(mkt_df, on="event_date", how="left")

        return df

    # ------------------------------------------------------------------
    def calculate_at_event(
        self,
        code: str,
        event_date,
        daily_data: pd.DataFrame,
    ) -> pd.Series:
        """单事件: PreTrend + Technical (大盘特征需要批量调用 calculate)."""
        pt = self.pre_trend.calculate_at_event(code, event_date, daily_data)
        if not self.use_technical:
            return pt
        ti = self.technical.calculate_at_event(code, event_date, daily_data)
        merged = pt.to_dict()
        ti_d = ti.to_dict()
        ti_d.pop("code", None)
        ti_d.pop("event_date", None)
        merged.update(ti_d)
        return pd.Series(merged)
