from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# ---- your backtester pieces ----
from backtest import BacktestConfig, backtest_from_json

load_dotenv()  # loads POLYGON_API_KEY, etc.

DATA_JSON_PATH = os.getenv("DATA_JSON_PATH", "data.json")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

app = FastAPI(title="Crypto Strategy Backtest API", version="1.0.0")

# Allow everything in dev; tighten in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


def _read_data_json() -> str:
    p = Path(DATA_JSON_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Missing {DATA_JSON_PATH}")
    return p.read_text(encoding="utf-8")


def _data_fingerprint() -> Tuple[float, int]:
    """
    Returns (mtime, size) of data.json so we can invalidate caches when file changes.
    """
    p = Path(DATA_JSON_PATH)
    st = p.stat()
    return (st.st_mtime, st.st_size)


def _cfg_key(cfg: BacktestConfig) -> Tuple[Any, ...]:
    # Use only fields that influence outputs. (ticker_map is handled by storage.)
    return (
        cfg.adjusted,
        cfg.use_same_day_weights,
        cfg.normalize_weights,
        cfg.infer_cash,
        float(cfg.cash_daily_return),
        float(cfg.trading_cost_bps),
    )


@lru_cache(maxsize=16)
def _compute_backtest_cached(
    data_fp: Tuple[float, int],
    cfg_key: Tuple[Any, ...],
    api_key: str,
) -> Dict[str, Any]:
    """
    Pure function wrapper for caching. NOTE: lru_cache keys are immutable tuples only.
    """
    # Rebuild a BacktestConfig from cfg_key
    cfg = BacktestConfig(
        adjusted=cfg_key[0],
        use_same_day_weights=cfg_key[1],
        normalize_weights=cfg_key[2],
        infer_cash=cfg_key[3],
        cash_daily_return=cfg_key[4],
        trading_cost_bps=cfg_key[5],
    )

    alloc_json_str = _read_data_json()
    result = backtest_from_json(alloc_json_str, cfg=cfg, api_key=api_key)

    # Convert DataFrames to JSON-safe dicts
    results_df = result["results"].copy()
    results_df = results_df.reset_index().rename(columns={"index": "date"})
    results_records = [
        {
            "date": str(row["date"]),
            "gross_return": float(row["gross_return"]),
            "cost": float(row["cost"]),
            "net_return": float(row["net_return"]),
            "equity": float(row["equity"]),
            "drawdown": float(row["drawdown"]),
        }
        for _, row in results_df.iterrows()
    ]

    stats_df = result["stats"]
    stats_obj = {k: float(v) for k, v in stats_df["strategy"].to_dict().items()}

    prices_df = result["prices"].copy().reset_index()
    price_records = [
        {"date": str(row.iloc[0]), **{col: float(row[col]) for col in result["prices"].columns}}
        for _, row in prices_df.iterrows()
    ]

    return {
        "series": results_records,
        "stats": stats_obj,
        "prices": price_records,  # handy if you want them
    }


def _compute_backtest(cfg: BacktestConfig, force: bool = False) -> Dict[str, Any]:
    """
    Wrapper that invalidates cache when data.json changes or force=True.
    """
    fp = _data_fingerprint()
    key = _cfg_key(cfg)

    # lru_cache has no TTL; we "bust" it by changing dummy arg on demand.
    if force:
        _compute_backtest_cached.cache_clear()  # type: ignore[attr-defined]

    try:
        return _compute_backtest_cached(fp, key, POLYGON_API_KEY)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": int(time.time())}


@app.get("/stats")
def get_stats(
    same_day: bool = Query(False, description="Apply weights to same-day close (optimistic)"),
    cash_yield: float = Query(0.0, description="Daily return for cash (e.g., 0.0001 ≈ 1bp/day)"),
    costs_bps: float = Query(2.0, description="One-way trading cost in basis points"),
    force: bool = Query(False, description="Force recompute, ignore cache"),
) -> Dict[str, Any]:
    cfg = BacktestConfig(
        use_same_day_weights=bool(same_day),
        cash_daily_return=float(cash_yield),
        trading_cost_bps=float(costs_bps),
    )
    res = _compute_backtest(cfg, force=force)
    return {"params": {"same_day": same_day, "cash_yield": cash_yield, "costs_bps": costs_bps}, "stats": res["stats"]}


@app.get("/equity")
def get_equity(
    capital: float = Query(100_000.0, description="Starting capital in USD"),
    same_day: bool = Query(False, description="Apply weights to same-day close (optimistic)"),
    cash_yield: float = Query(0.0, description="Daily return for cash"),
    costs_bps: float = Query(2.0, description="One-way trading cost in bps"),
    force: bool = Query(False, description="Force recompute, ignore cache"),
) -> JSONResponse:
    """
    Returns:
      {
        "capital": 100000.0,
        "count": N,
        "series": [
          {"date": "YYYY-MM-DD", "equity": 1.0, "equity_value": 100000.0},
          ...
        ]
      }
    """
    cfg = BacktestConfig(
        use_same_day_weights=bool(same_day),
        cash_daily_return=float(cash_yield),
        trading_cost_bps=float(costs_bps),
    )
    res = _compute_backtest(cfg, force=force)

    items = []
    for row in res["series"]:
        equity = float(row["equity"])
        items.append(
            {
                "date": row["date"],
                "equity": equity,
                "equity_value": capital * equity,
            }
        )

    payload = {"capital": capital, "count": len(items), "series": items}
    return JSONResponse(content=payload)
