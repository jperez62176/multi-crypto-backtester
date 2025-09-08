"""
Cache Polygon.io crypto daily bars in Postgres and return a prices DataFrame.

Usage (standalone):
  export POLYGON_API_KEY=...
  python storage.py --file data.json

Or import and call:
  from storage import fetch_prices_with_cache
  prices = fetch_prices_with_cache(Path("data.json").read_text(), api_key=os.getenv("POLYGON_API_KEY"))
"""

from __future__ import annotations

import os
import time
import json
import argparse
from datetime import date, datetime
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests
import psycopg2
import psycopg2.extras as pgx


# ---------- Config / Defaults ----------
DEFAULT_TICKER_MAP = {
    "btc": "X:BTCUSD",
    "eth": "X:ETHUSD",
    "sol": "X:SOLUSD",
}
POLYGON_BASE_URL = "https://api.polygon.io"
# DB connection defaults (override with env)
PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = int(os.getenv("PGPORT", "5432"))
PGUSER = os.getenv("PGUSER", "crypto")
PGPASSWORD = os.getenv("PGPASSWORD", "crypto")
PGDATABASE = os.getenv("PGDATABASE", "cryptodata")


# ---------- DB helpers ----------
def db_connect():
    return psycopg2.connect(
        host=PGHOST, port=PGPORT, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE
    )


def ensure_schema(conn) -> None:
    """Create the table/indexes if they don't exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS crypto_daily_bars (
        symbol      TEXT NOT NULL,           -- Polygon symbol, e.g., X:BTCUSD
        asset_key   TEXT NOT NULL,           -- your key, e.g., 'btc'
        dt          DATE NOT NULL,
        open        DOUBLE PRECISION,
        high        DOUBLE PRECISION,
        low         DOUBLE PRECISION,
        close       DOUBLE PRECISION,
        volume      DOUBLE PRECISION,
        vwap        DOUBLE PRECISION,
        PRIMARY KEY (symbol, dt)
    );

    CREATE INDEX IF NOT EXISTS idx_crypto_daily_bars_dt ON crypto_daily_bars (dt);
    CREATE INDEX IF NOT EXISTS idx_crypto_daily_bars_asset ON crypto_daily_bars (asset_key);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def existing_dates(conn, symbol: str, start: date, end: date) -> Set[date]:
    sql = """
    SELECT dt
    FROM crypto_daily_bars
    WHERE symbol = %s AND dt BETWEEN %s AND %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, start, end))
        rows = cur.fetchall()
    return {r[0] for r in rows}


def upsert_bars(conn, rows: List[Tuple]) -> None:
    """
    rows: list of tuples matching the VALUES below.
    """
    if not rows:
        return
    sql = """
    INSERT INTO crypto_daily_bars
      (symbol, asset_key, dt, open, high, low, close, volume, vwap)
    VALUES %s
    ON CONFLICT (symbol, dt) DO UPDATE
      SET open = EXCLUDED.open,
          high = EXCLUDED.high,
          low  = EXCLUDED.low,
          close= EXCLUDED.close,
          volume=EXCLUDED.volume,
          vwap = EXCLUDED.vwap
    """
    with conn.cursor() as cur:
        pgx.execute_values(cur, sql, rows, page_size=1000)
    conn.commit()


def read_prices_from_db(conn, asset_keys: List[str], start: date, end: date) -> pd.DataFrame:
    """
    Return a DataFrame with index=YYYY-MM-DD (str) and columns per asset_key, values=close.
    """
    if not asset_keys:
        return pd.DataFrame()

    sql = """
    SELECT dt, asset_key, close
    FROM crypto_daily_bars
    WHERE asset_key = ANY(%s) AND dt BETWEEN %s AND %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (asset_keys, start, end))
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "asset_key", "close"])
    df["date"] = df["date"].astype(str)
    prices = df.pivot_table(index="date", columns="asset_key", values="close").sort_index()
    return prices


# ---------- Polygon helpers ----------
def polygon_get(url: str, params: Dict, api_key: str, retries: int = 4, backoff: float = 1.5) -> dict:
    params = dict(params or {})
    params["apiKey"] = api_key
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        # Handle rate limiting 429 or transient 5xx
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep_s = backoff ** attempt
            time.sleep(sleep_s)
            continue
        resp.raise_for_status()
    # Last try
    resp.raise_for_status()
    return {}  # unreachable


def fetch_polygon_daily(symbol: str, start: date, end: date, api_key: str, adjusted: bool = True) -> pd.DataFrame:
    """
    One call per symbol for the whole date range (free tier friendly).
    """
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    data = polygon_get(
        url,
        {"adjusted": str(adjusted).lower(), "sort": "asc", "limit": 50000},
        api_key=api_key,
    )
    results = data.get("results") or []
    if not results:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "vwap"])

    df = pd.DataFrame(results)
    # Polygon 't' is ms since epoch (UTC)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"})
    return df[["date", "open", "high", "low", "close", "volume", "vwap"]].sort_values("date")


# ---------- Orchestration ----------
def parse_allocations(alloc_json_str: str) -> Tuple[List[str], date, date]:
    """Return (asset_keys, start_date, end_date) from the allocation JSON."""
    alloc = json.loads(alloc_json_str)
    if not isinstance(alloc, list) or not alloc:
        raise ValueError("Allocation JSON must be a non-empty list of rows.")

    # Collect all asset keys (exclude date/cash)
    asset_keys = sorted({
        k.lower()
        for row in alloc
        for k in row.keys()
        if k.lower() not in ("date", "cash")
    })

    # Range
    dates = sorted(pd.to_datetime([row["date"] for row in alloc]).dt.date.tolist())
    return asset_keys, dates[0], dates[-1]


def build_ticker_map(asset_keys: List[str], user_map: Dict[str, str] | None) -> Dict[str, str]:
    m = dict(DEFAULT_TICKER_MAP)
    if user_map:
        m.update({k.lower(): v for k, v in user_map.items()})
    # Fallback: X:{SYMBOL}USD for unknowns
    for a in asset_keys:
        if a not in m:
            m[a] = f"X:{a.upper()}USD"
    return m


def fill_cache_if_needed(
    conn, asset_keys: List[str], ticker_map: Dict[str, str], start: date, end: date, api_key: str, adjusted: bool = True
) -> None:
    """
    For each ticker: find missing dates in [start, end]. If any are missing,
    fetch once from earliest missing -> end and upsert.
    """
    all_days = set(pd.date_range(start, end, freq="D").date)

    for asset in asset_keys:
        symbol = ticker_map[asset]
        have = existing_dates(conn, symbol, start, end)
        missing = sorted(all_days - have)
        if not missing:
            continue

        first_missing = missing[0]
        df = fetch_polygon_daily(symbol, first_missing, end, api_key, adjusted=adjusted)
        if df.empty:
            # No data returned (symbol typo or API issue) — skip gracefully
            continue

        # Only upsert missing dates; keep data lean
        df = df[df["date"].isin(missing)]
        rows = [
            (symbol, asset, d, float(o) if o is not None else None,
             float(h) if h is not None else None,
             float(l) if l is not None else None,
             float(c) if c is not None else None,
             float(v) if v is not None else None,
             float(vw) if vw is not None else None)
            for d, o, h, l, c, v, vw in df[["date", "open", "high", "low", "close", "volume", "vwap"]].itertuples(index=False, name=None)
        ]
        upsert_bars(conn, rows)


def fetch_prices_with_cache(
    alloc_json_str: str,
    api_key: str,
    ticker_map: Dict[str, str] | None = None,
    adjusted: bool = True,
) -> pd.DataFrame:
    """
    Main entry: ensure DB has all needed bars, then return close prices wide-form.
    Index='YYYY-MM-DD' strings, columns=asset_keys, values=close.
    """
    asset_keys, start, end = parse_allocations(alloc_json_str)
    tmap = build_ticker_map(asset_keys, ticker_map)

    with db_connect() as conn:
        ensure_schema(conn)
        fill_cache_if_needed(conn, asset_keys, tmap, start, end, api_key, adjusted=adjusted)
        prices = read_prices_from_db(conn, asset_keys, start, end)

    return prices


# ---------- CLI for quick testing ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default="data.json", help="Path to allocations JSON")
    parser.add_argument("--api-key", default=os.getenv("POLYGON_API_KEY", ""), help="Polygon API key")
    args = parser.parse_args()

    alloc_json_str = open(args.file, "r", encoding="utf-8").read()
    prices = fetch_prices_with_cache(alloc_json_str, api_key=args.api_key)

    print("Prices (head):")
    print(prices.head())
    print("\nColumns:", list(prices.columns))


if __name__ == "__main__":
    main()
