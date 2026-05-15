from __future__ import annotations

import csv
import json
import mimetypes
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.tdx_loader import TDXDataLoader


STATIC_DIR = ROOT / "web"
MODELS_DIR = ROOT / "models"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
_STOCK_NAMES: dict[str, str] | None = None


@dataclass
class ModelInfo:
    id: str
    name: str
    version: str
    path: Path
    kind: str
    trained_at: str | None = None
    summary: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None

    def artifact(self, filename: str) -> Path:
        return self.path / filename

    @property
    def has_signals(self) -> bool:
        return self.artifact("signals.csv").exists()

    @property
    def has_trades(self) -> bool:
        return self.artifact("trades.csv").exists()

    def to_dict(self) -> dict[str, Any]:
        metrics = {}
        summary = self.summary or {}
        if isinstance(summary.get("backtest_metrics"), dict):
            metrics = summary["backtest_metrics"]
        elif self.artifact("backtest_metrics.json").exists():
            metrics = read_json(self.artifact("backtest_metrics.json"), {})

        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "trained_at": self.trained_at,
            "path": str(self.path),
            "has_model": self.artifact("model.pkl").exists() or self.artifact("limit_up_lgbm.pkl").exists(),
            "has_signals": self.has_signals,
            "has_trades": self.has_trades,
            "has_equity": self.artifact("equity_curve.csv").exists(),
            "metrics": metrics,
        }


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def load_stock_names() -> dict[str, str]:
    global _STOCK_NAMES
    if _STOCK_NAMES is not None:
        return _STOCK_NAMES
    names: dict[str, str] = {}
    for path in [
        Path(r"C:\new_tdx\T0002\hq_cache\shs.tnf"),
        Path(r"C:\new_tdx\T0002\hq_cache\szs.tnf"),
        Path(r"C:\new_tdx\T0002\hq_cache\bjs.tnf"),
    ]:
        if not path.exists():
            continue
        data = path.read_bytes()
        # TDX .tnf layouts vary slightly by version. A stock record contains
        # a 6-digit code and the GBK name about 30 bytes later, so scan for it.
        for off in range(0, len(data) - 80):
            chunk = data[off:off + 6]
            if not chunk.isdigit() or data[off + 6] != 0:
                continue
            code = chunk.decode("ascii", errors="ignore")
            raw_name = data[off + 31:off + 63].split(b"\x00", 1)[0]
            name = raw_name.decode("gbk", errors="ignore").strip()
            if name and any("\u4e00" <= ch <= "\u9fff" for ch in name):
                names[code] = name
    _STOCK_NAMES = names
    return names


def stock_name(code: str) -> str:
    c = str(code).zfill(6)
    return load_stock_names().get(c, c)


def discover_models() -> list[ModelInfo]:
    models: list[ModelInfo] = []

    registry = read_json(MODELS_DIR / "registry.json", {})
    for item in registry.get("versions", []) if isinstance(registry, dict) else []:
        version = str(item.get("version", "unknown"))
        path = MODELS_DIR / version
        meta = read_json(path / "meta.json", {})
        models.append(
            ModelInfo(
                id=f"registry:{version}",
                name=str(meta.get("name") or "limit_up_main_wave"),
                version=version,
                path=path,
                kind="registry",
                trained_at=str(meta.get("trained_at") or item.get("trained_at") or ""),
                meta=meta,
            )
        )

    for strategy_root in sorted(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else []:
        if not strategy_root.is_dir():
            continue
        if not (strategy_root / "latest").exists() and not any(strategy_root.glob("run_*")):
            continue
        strategy = strategy_root.name
        for path in sorted(strategy_root.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            if path.name != "latest" and not path.name.startswith("run_"):
                continue
            summary = read_json(path / "summary.json", {})
            config = read_json(path / "config.json", {})
            run_id = str(summary.get("run_id") or path.name.replace("run_", ""))
            version = "latest" if path.name == "latest" else run_id
            models.append(
                ModelInfo(
                    id=f"{strategy}:{path.name}",
                    name=str(summary.get("strategy") or config.get("strategy_name") or strategy),
                    version=version,
                    path=path,
                    kind=strategy,
                    trained_at=run_id if run_id != "latest" else None,
                    summary=summary,
                    meta={"config": config},
                )
            )

    models.sort(
        key=lambda m: (
            1 if (m.has_signals and m.has_trades) else 0,
            1 if m.has_signals else 0,
            str(m.trained_at or m.version),
        ),
        reverse=True,
    )
    return models


def get_model(model_id: str | None) -> ModelInfo | None:
    models = discover_models()
    if model_id:
        for model in models:
            if model.id == model_id:
                return model
    for model in models:
        if model.has_signals:
            return model
    return models[0] if models else None


def unique_sorted_dates(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(r.get("date", ""))[:10] for r in rows if r.get("date")})


def signals_for_model(model: ModelInfo, params: dict[str, list[str]]) -> dict[str, Any]:
    rows = read_csv_rows(model.artifact("signals.csv"))
    dates = unique_sorted_dates(rows)
    selected_date = params.get("date", [""])[0] or (dates[-1] if dates else "")
    min_score = as_float(params.get("min_score", ["0"])[0], 0.0)
    top = int(as_float(params.get("top", ["50"])[0], 50))

    filtered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        row_date = str(row.get("date", ""))[:10]
        score = as_float(row.get("score"))
        code = str(row.get("code", "")).zfill(6)
        key = (row_date, code)
        if row_date != selected_date or score < min_score or key in seen:
            continue
        seen.add(key)
        filtered.append(
            {
                "date": row_date,
                "code": code,
                "name": stock_name(code),
                "score": score,
                "stop_loss_price": as_float(row.get("stop_loss_price")),
            }
        )

    filtered.sort(key=lambda r: r["score"], reverse=True)
    filtered = filtered[:top]
    if filtered:
        loader = TDXDataLoader()
        for rank, row in enumerate(filtered, start=1):
            row["rank"] = rank
            row["decision_time"] = "14:30-14:50"
            row["decision"] = "确认买入"
            row["position_pct"] = 0.10
            row["execution_note"] = "尾盘模型确认后人工下单；回测成交不作为当天清单口径"
            try:
                day_df = loader.load_day_file(row["code"])
                day = day_df[day_df["date"] == pd.to_datetime(row["date"])]
                if not day.empty:
                    d = day.iloc[-1]
                    row["open"] = float(d["open"])
                    row["high"] = float(d["high"])
                    row["low"] = float(d["low"])
                    row["close"] = float(d["close"])
                    row["volume"] = float(d["volume"])
                    row["amount"] = float(d.get("turnover", 0.0))
                    row["change_pct"] = 0.0 if d["change_pct"] != d["change_pct"] else float(d["change_pct"])
                    row["turnover_rate"] = None
            except Exception:
                pass
    return {
        "model": model.to_dict(),
        "dates": dates[-260:],
        "selected_date": selected_date,
        "rows": filtered,
        "missing": [] if model.artifact("signals.csv").exists() else ["signals.csv"],
        "selection_window": "14:30-14:50",
        "selection_note": "该列表是尾盘确认买入清单，不等于回测成交明细。成交由人工确认或交易复盘单独记录。",
    }


def trades_for_model(model: ModelInfo, params: dict[str, list[str]]) -> dict[str, Any]:
    code = params.get("code", [""])[0]
    rows = read_csv_rows(model.artifact("trades.csv"))
    if code:
        rows = [r for r in rows if str(r.get("code", "")).zfill(6) == code.zfill(6)]
    for row in rows:
        code_val = str(row.get("code", "")).zfill(6)
        row["code"] = code_val
        row["name"] = stock_name(code_val)
        row["price"] = as_float(row.get("price"))
        row["amount"] = as_float(row.get("amount"))
        row["pnl"] = as_float(row.get("pnl"))
        row["return_pct"] = as_float(row.get("return_pct"))
    return {
        "model": model.to_dict(),
        "rows": rows,
        "missing": [] if model.artifact("trades.csv").exists() else ["trades.csv"],
    }


def kline_for_code(model: ModelInfo, params: dict[str, list[str]]) -> dict[str, Any]:
    code = params.get("code", [""])[0].zfill(6)
    start = params.get("start", [""])[0] or None
    end = params.get("end", [""])[0] or None
    loader = TDXDataLoader()
    df = loader.load_day_file(code)
    marker_rows = read_csv_rows(model.artifact("trades.csv"))
    markers = []
    for row in marker_rows:
        if str(row.get("code", "")).zfill(6) != code:
            continue
        markers.append(
            {
                "date": str(row.get("date", ""))[:10],
                "action": str(row.get("action", "")),
                "price": as_float(row.get("price")),
                "return_pct": as_float(row.get("return_pct")),
            }
        )
    df = df.sort_values("date").reset_index(drop=True)
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    elif markers and not df.empty:
        marker_dates = pd.to_datetime([m["date"] for m in markers if m.get("date")])
        if len(marker_dates):
            dates = list(df["date"])
            lo_date = min(marker_dates)
            hi_date = max(marker_dates)
            lo_idx = next((i for i, d in enumerate(dates) if d >= lo_date), 0)
            hi_idx = next((i for i, d in enumerate(dates) if d >= hi_date), len(dates) - 1)
            left = max(0, lo_idx - 80)
            right = min(len(df), hi_idx + 81)
            df = df.iloc[left:right]
    else:
        df = df.tail(240)

    candles = []
    for _, row in df.iterrows():
        candles.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "amount": float(row["turnover"]),
                "turnover_rate": None,
                "ma5": None if row["ma5"] != row["ma5"] else float(row["ma5"]),
                "ma10": None if row["ma10"] != row["ma10"] else float(row["ma10"]),
                "ma20": None if row["ma20"] != row["ma20"] else float(row["ma20"]),
                "change_pct": 0.0 if row["change_pct"] != row["change_pct"] else float(row["change_pct"]),
            }
        )
    return {
        "code": code,
        "name": stock_name(code),
        "candles": candles,
        "markers": markers,
        "missing": [] if candles else ["tdx day data"],
    }


def equity_for_model(model: ModelInfo) -> dict[str, Any]:
    rows = read_csv_rows(model.artifact("equity_curve.csv"))
    data = [
        {
            "date": str(r.get("date", ""))[:10],
            "total_value": as_float(r.get("total_value")),
            "cash": as_float(r.get("cash")),
            "market_value": as_float(r.get("market_value")),
        }
        for r in rows
    ]
    return {
        "model": model.to_dict(),
        "rows": data,
        "missing": [] if model.artifact("equity_curve.csv").exists() else ["equity_curve.csv"],
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        params = parse_qs(parsed.query)
        try:
            if path.startswith("/api/"):
                self.handle_api(path, params)
            else:
                self.serve_static(path)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def handle_api(self, path: str, params: dict[str, list[str]]) -> None:
        if path == "/api/models":
            self.send_json({"models": [m.to_dict() for m in discover_models()]})
            return

        model = get_model(params.get("model", [None])[0])
        if model is None:
            self.send_json({"error": "no model found"}, status=404)
            return

        if path == "/api/signals":
            self.send_json(signals_for_model(model, params))
        elif path == "/api/trades":
            self.send_json(trades_for_model(model, params))
        elif path == "/api/kline":
            self.send_json(kline_for_code(model, params))
        elif path == "/api/minute":
            self.send_json({"rows": [], "missing": ["本地未发现通达信分钟数据文件，当前先展示日K复盘。"]})
        elif path == "/api/equity":
            self.send_json(equity_for_model(model))
        else:
            self.send_json({"error": f"unknown api: {path}"}, status=404)

    def serve_static(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        file_path = (STATIC_DIR / path.lstrip("/")).resolve()
        if STATIC_DIR.resolve() not in file_path.parents and file_path != STATIC_DIR.resolve():
            self.send_error(403)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return
        data = file_path.read_bytes()
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    httpd = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
