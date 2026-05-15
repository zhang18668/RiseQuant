r"""Pattern-Cluster dry-run.

Usage (Windows, single-line):

    set PYTHONPATH=%CD%
    python scripts\wash_pattern_dryrun.py --start 2022-01-01 --end 2024-12-31 --golden-lo 0.10 0.15 0.20 --golden-hi 0.40 0.35 0.30

If --golden-lo / --golden-hi are omitted, defaults to three bins:
  [0.10,0.40] [0.15,0.35] [0.20,0.30].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.tdx_loader import TDXDataLoader
from src.event.data_dryrun import DataDryRun
from src.utils.logger import get_logger, setup_logger

setup_logger(log_level="INFO")
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Pattern-Cluster dry-run 数据诊断")
    p.add_argument("--start", required=True, help="起始日期 YYYY-MM-DD")
    p.add_argument("--end",   required=True, help="结束日期 YYYY-MM-DD")
    p.add_argument("--tdx-path", default=r"C:\new_tdx\vipdoc",
                   help="通达信 vipdoc 目录")
    p.add_argument("--golden-lo", type=float, nargs="+",
                   default=[0.10, 0.15, 0.20])
    p.add_argument("--golden-hi", type=float, nargs="+",
                   default=[0.40, 0.35, 0.30])
    p.add_argument("--horizon", type=int, default=22, help="未来窗口天数")
    p.add_argument("--min-events", type=int, default=300,
                   help="目标区间金标准事件数下限")
    p.add_argument("--target-lo", type=float, default=0.15)
    p.add_argument("--target-hi", type=float, default=0.35)
    p.add_argument("--out", default="", help="可选: 把报告 JSON 写到此路径")
    p.add_argument("--max-gap", type=int, default=30,
                   help="WashSecondDetector.max_gap")
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.golden_lo) != len(args.golden_hi):
        logger.error("--golden-lo 和 --golden-hi 长度必须相同")
        sys.exit(2)
    bins = list(zip(args.golden_lo, args.golden_hi))

    logger.info("加载日线数据 …")
    tdx = TDXDataLoader(args.tdx_path)
    if tdx.data_path is None:
        logger.error(f"找不到通达信目录: {args.tdx_path}")
        sys.exit(2)
    sh = tdx.load_stock_list("sh")
    sz = tdx.load_stock_list("sz")
    codes = [c for c in sh["code"].tolist() + sz["code"].tolist() if tdx.is_main_board(c)]
    logger.info(f"加载 {len(codes)} 只主板股票")
    daily = tdx.load_batch(codes=codes, start_date=args.start, end_date=args.end)
    if daily.empty:
        logger.error("日线数据为空")
        sys.exit(2)
    daily["date"] = pd.to_datetime(daily["date"])

    dry = DataDryRun(
        lo_hi_bins=bins,
        horizon=args.horizon,
        min_events_required=args.min_events,
        target_bin=(args.target_lo, args.target_hi),
        detector_max_gap=args.max_gap,
    )
    rep = dry.summarize(daily)
    print()
    print(rep.pretty_print())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rep.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"报告已写入 {out_path}")

    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
