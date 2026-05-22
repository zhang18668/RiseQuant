"""Validate candidate historical limit-up-pool data sources.

The goal is to verify whether a provider can really return historical rows with
the fields needed by the limit-up pullback research:

code/name/date, first/last seal time, seal amount, turnover, float market cap,
open-board count, consecutive-limit count.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from src.data.zt_pool_loader import ZT_POOL_TYPE, ZtPoolLoader
from src.data.tushare_limit_pool import (
    TushareLimitPoolClient,
    normalize_limit_list_to_zt_pool,
)


DEFAULT_DATES = "20200102,20220615,20241125,20260520"
OUT_DIR = ROOT / "data" / "source_validation"


REQUIRED_FIELDS = {
    "code",
    "trade_date",
    "name",
    "first_time",
    "last_time",
    "fd_amount",
    "turnover_rate",
    "float_mv",
    "open_times",
    "limit_times",
}


def parse_args():
    p = argparse.ArgumentParser(description="Validate historical limit-up-pool sources")
    p.add_argument("--dates", default=DEFAULT_DATES, help="Comma separated YYYYMMDD dates")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--timeout", type=int, default=20)
    return p.parse_args()


def normalize_date(raw: str) -> str:
    return pd.Timestamp(raw).strftime("%Y%m%d")


def norm_tscode(value: Any) -> str:
    raw = str(value or "")
    if "." in raw:
        raw = raw.split(".", 1)[0]
    return raw.zfill(6)


def normalize_akshare(df: pd.DataFrame, date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_FIELDS))
    rename = {
        "代码": "code",
        "名称": "name",
        "首次封板时间": "first_time",
        "最后封板时间": "last_time",
        "封板资金": "fd_amount",
        "换手率": "turnover_rate",
        "流通市值": "float_mv",
        "炸板次数": "open_times",
        "连板数": "limit_times",
    }
    out = df.rename(columns=rename).copy()
    out["trade_date"] = date
    out["code"] = out["code"].map(norm_tscode)
    return out


def normalize_tushare(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_FIELDS))
    rename = {
        "ts_code": "code",
        "turnover_ratio": "turnover_rate",
    }
    out = df.rename(columns=rename).copy()
    out["code"] = out["code"].map(norm_tscode)
    return out


def normalize_zt_pool_cache_shape(df: pd.DataFrame, date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_FIELDS))
    rename = {
        "代码": "code",
        "名称": "name",
        "首次封板时间": "first_time",
        "最后封板时间": "last_time",
        "封板资金": "fd_amount",
        "换手率": "turnover_rate",
        "流通市值": "float_mv",
        "炸板次数": "open_times",
        "连板数": "limit_times",
    }
    out = df.rename(columns=rename).copy()
    out["trade_date"] = date
    out["code"] = out["code"].map(norm_tscode)
    return out


def normalize_numcat(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_FIELDS))
    rename = {
        "symbol": "code",
        "tradedate": "trade_date",
        "turnover_z": "turnover_rate",
    }
    out = df.rename(columns=rename).copy()
    out["code"] = out["code"].map(norm_tscode)
    if "limit_times" not in out.columns:
        out["limit_times"] = out.get("pre_limit_times", pd.NA)
    return out


def normalize_asharehub(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_FIELDS))
    out = df.copy()
    out["code"] = out["ts_code"].map(norm_tscode) if "ts_code" in out.columns else out.get("code", "").map(norm_tscode)
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y%m%d")
    return out


def fetch_akshare(date: str, timeout: int) -> pd.DataFrame:
    loader = ZtPoolLoader(cache_dir=ROOT / "data" / "_tmp_source_validation", require_akshare=True)
    return normalize_akshare(
        loader.fetch_one(date, pool_type=ZT_POOL_TYPE, use_cache=False, save_cache=False),
        date,
    )


def fetch_tushare(date: str, timeout: int) -> pd.DataFrame:
    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TS_TOKEN") or os.environ.get("TUSHARE_PRO_TOKEN")
    if not token:
        raise RuntimeError("missing TUSHARE_TOKEN/TS_TOKEN/TUSHARE_PRO_TOKEN")
    payload = {
        "api_name": "limit_list_d",
        "token": token,
        "params": {"trade_date": date, "limit_type": "U"},
        "fields": ",".join([
            "trade_date", "ts_code", "industry", "name", "close", "pct_chg",
            "amount", "float_mv", "total_mv", "turnover_ratio", "fd_amount",
            "first_time", "last_time", "open_times", "up_stat", "limit_times", "limit",
        ]),
    }
    res = requests.post("http://api.tushare.pro", json=payload, timeout=timeout)
    res.raise_for_status()
    data = res.json()
    if data.get("code") != 0:
        raise RuntimeError(f"tushare error {data.get('code')}: {data.get('msg')}")
    fields = data["data"]["fields"]
    rows = data["data"]["items"]
    return normalize_tushare(pd.DataFrame(rows, columns=fields))


def fetch_tinyshare(date: str, timeout: int) -> pd.DataFrame:
    client = TushareLimitPoolClient(prefer_tinyshare=True)
    raw = client.limit_list_d(date, limit_type="U")
    cache_shape = normalize_limit_list_to_zt_pool(raw)
    return normalize_zt_pool_cache_shape(cache_shape, date)


def fetch_numcat(date: str, timeout: int) -> pd.DataFrame:
    key = os.environ.get("NUMCAT_APIKEY")
    if not key:
        raise RuntimeError("missing NUMCAT_APIKEY")
    payload = {
        "apikey": key,
        "apiname": "limit_pool",
        "fields": ",".join([
            "symbol", "name", "tradedate", "type", "limit_times", "fd_amount",
            "pct_chg", "amount", "turnover_z", "first_time", "last_time",
            "open_times", "limit_type",
        ]),
        "params": {"startdate": date, "enddate": date, "type": "u", "limit": 5000},
    }
    res = requests.post("https://numcat.net/api", json=payload, timeout=timeout)
    res.raise_for_status()
    data = res.json()
    if data.get("code") not in (0, 200, "0", "200", None):
        raise RuntimeError(f"numcat error {data.get('code')}: {data.get('msg') or data.get('message')}")
    rows = data.get("data") or data.get("rows") or data.get("result") or []
    return normalize_numcat(pd.DataFrame(rows))


def fetch_asharehub(date: str, timeout: int) -> pd.DataFrame:
    key = os.environ.get("ASHAREHUB_API_KEY")
    if not key:
        raise RuntimeError("missing ASHAREHUB_API_KEY")
    params = {
        "start_date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
        "end_date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
        "limit_type": "U",
        "limit": 5000,
    }
    res = requests.get(
        "https://asharehub.com/v1/market/limit-list",
        params=params,
        headers={"X-API-Key": key},
        timeout=timeout,
    )
    res.raise_for_status()
    data = res.json()
    rows = data.get("data") if isinstance(data, dict) else data
    return normalize_asharehub(pd.DataFrame(rows or []))


def coverage(df: pd.DataFrame) -> dict[str, Any]:
    cols = set(df.columns)
    present = sorted(REQUIRED_FIELDS & cols)
    missing = sorted(REQUIRED_FIELDS - cols)
    non_null = {
        col: int(df[col].notna().sum()) if col in df.columns else 0
        for col in sorted(REQUIRED_FIELDS)
    }
    return {
        "rows": int(len(df)),
        "present": present,
        "missing": missing,
        "non_null": non_null,
    }


def validate_source(name: str, fetcher: Callable[[str, int], pd.DataFrame], dates: list[str], timeout: int, out_dir: Path) -> dict:
    source_dir = out_dir / name
    source_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for date in dates:
        try:
            df = fetcher(date, timeout)
            df.to_csv(source_dir / f"{date}.csv", index=False, encoding="utf-8-sig")
            item = {"date": date, "status": "ok", **coverage(df)}
        except Exception as exc:  # noqa: BLE001
            item = {"date": date, "status": "error", "error": str(exc), "rows": 0}
        results.append(item)
    return {"source": name, "results": results}


def main() -> int:
    args = parse_args()
    dates = [normalize_date(d.strip()) for d in args.dates.split(",") if d.strip()]
    out_dir = Path(args.out_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only tinyshare is used as the canonical historical zt_pool source.
    # Other providers (akshare/tushare/numcat/asharehub) are kept as commented-out
    # references for future evaluation but are not run by default.
    sources: list[tuple[str, Callable[[str, int], pd.DataFrame]]] = [
        ("tinyshare_limit_list_d", fetch_tinyshare),
        # ("akshare_eastmoney",      fetch_akshare),
        # ("tushare_limit_list_d",   fetch_tushare),
        # ("numcat_limit_pool",      fetch_numcat),
        # ("asharehub_limit_list",   fetch_asharehub),
    ]

    report = [validate_source(name, fn, dates, args.timeout, out_dir) for name, fn in sources]
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for source in report:
        print(f"\n[{source['source']}]")
        for item in source["results"]:
            if item["status"] == "ok":
                missing = ",".join(item.get("missing", [])) or "-"
                print(f"  {item['date']} rows={item['rows']} missing={missing}")
            else:
                print(f"  {item['date']} ERROR {item['error']}")
    print(f"\nsummary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
