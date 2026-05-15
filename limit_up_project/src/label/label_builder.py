"""L-001 标签构建器 (优化期望收益)

支持期望收益最大化标签:
- label_expectation: 0/1, 基于未来N日收益是否超过阈值

训练目标: E = P(win) * avg_win - P(loss) * avg_loss 最大化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class LabelBuilder:
    """复合标签构建器 - 优化期望收益"""

    LABEL_FAIL: int = 0
    LABEL_WIN: int = 1

    # 持有N日后涨幅阈值 (用于判定是否盈利)
    hold_days: int = 5
    return_threshold: float = 0.02  # 2% 涨幅阈值

    # 止盈止损
    stop_profit: float = 0.10  # 10% 止盈
    stop_loss: float = -0.05  # -5% 止损

    # ------------------------------------------------------------------
    def build_label(
        self,
        daily_data: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        构建期望收益标签

        Args:
            daily_data: 日线数据 (date, code, close, open, high, low)
            events: 事件表 (code, event_date) - 信号产生日

        Returns:
            DataFrame with (code, event_date, label, future_return)
        """
        if events is None or events.empty:
            return pd.DataFrame(columns=["code", "event_date", "label", "future_return"])

        df = events.copy()
        df["event_date"] = pd.to_datetime(df["event_date"])

        # 为每个事件计算未来收益
        labels = []
        for _, ev in df.iterrows():
            code = ev["code"]
            ev_date = ev["event_date"]

            # 获取事件后N日的股价数据
            stock_data = daily_data[
                (daily_data["code"] == code) & (daily_data["date"] > ev_date)
            ].sort_values("date").head(self.hold_days)

            if stock_data.empty:
                labels.append({"code": code, "event_date": ev_date, "label": 0, "future_return": 0.0})
                continue

            # 计算买入价格 (次日开盘)
            first_day = stock_data.iloc[0]
            buy_price = first_day["close"]  # 信号日收盘买入

            # 持有期最高价/最低价
            max_high = stock_data["high"].max()
            min_low = stock_data["low"].min()

            # 计算收益率
            sell_price = stock_data.iloc[-1]["close"]
            total_return = (sell_price / buy_price - 1.0) if buy_price > 0 else 0.0

            # 盘中最高/最低收益
            high_return = (max_high / buy_price - 1.0) if buy_price > 0 else 0.0
            low_return = (min_low / buy_price - 1.0) if buy_price > 0 else 0.0

            # 判定是否止损/止盈出场
            if low_return <= self.stop_loss:
                # 止损出局
                label = 0
                future_return = self.stop_loss
            elif high_return >= self.stop_profit:
                # 止盈出局
                label = 1
                future_return = self.stop_profit
            else:
                # 持有到期
                label = 1 if total_return >= self.return_threshold else 0
                future_return = total_return

            labels.append({
                "code": code,
                "event_date": ev_date,
                "label": label,
                "future_return": future_return,
            })

        result = pd.DataFrame(labels)

        # 合并回原事件表
        df = df.merge(result, on=["code", "event_date"], how="left")
        df["label"] = df["label"].fillna(0).astype(int)
        df["future_return"] = df["future_return"].fillna(0.0)

        return df[["code", "event_date", "label", "future_return"]]

    # ------------------------------------------------------------------
    def calc_label_short(self, row: dict) -> int:
        """简单二板标签"""
        return 1 if bool(row.get("has_second", False)) else 0

    def calc_label_combined(self, row: dict) -> int:
        """复合标签"""
        if not row.get("has_second", False):
            return self.LABEL_FAIL
        period_return = row.get("period_return")
        if period_return is None or pd.isna(period_return):
            return self.LABEL_WIN
        if float(period_return) >= 0.15:
            return 2  # 完美
        return self.LABEL_WIN

    # ------------------------------------------------------------------
    def build(
        self,
        event_data: pd.DataFrame,
        daily_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """兼容旧接口"""
        if daily_data is not None and "close" in daily_data.columns:
            return self.build_label(daily_data, event_data)

        # 旧逻辑
        if event_data is None or event_data.empty:
            return pd.DataFrame(columns=[
                "sample_id", "code", "first_date", "label_short", "label_combined",
            ])

        df = event_data.copy()
        if "has_second" not in df.columns:
            df["has_second"] = df.get("second_date").notna() if "second_date" in df.columns else False
        if "period_return" not in df.columns:
            df["period_return"] = float("nan")
        if "sample_id" not in df.columns:
            df["sample_id"] = df["code"].astype(str) + "_" + pd.to_datetime(df["first_date"]).dt.strftime("%Y%m%d")

        df["label_short"] = df["has_second"].astype(bool).astype(int)
        df["label_combined"] = df.apply(
            lambda r: self.calc_label_combined({
                "has_second": bool(r["has_second"]),
                "period_return": r["period_return"],
            }),
            axis=1,
        )
        return df[["sample_id", "code", "first_date", "label_short", "label_combined"]].reset_index(drop=True)
