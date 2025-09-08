import json, math, os, requests
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path
import pandas as pd
import numpy as np

from storage import fetch_prices_with_cache


# ------------------------------------
# Config
# ------------------------------------
POLYGON_API_KEY = "wAwogplaBUSpVq4_wemgzpeqP4jp2Hf5"  # set me (or use env + read it)
BASE_URL = "https://api.polygon.io"

# Map simple asset keys in your JSON -> Polygon crypto tickers
DEFAULT_TICKER_MAP = {
    "btc": "X:BTCUSD",
    "eth": "X:ETHUSD",
    "sol": "X:SOLUSD",
    "sui": "X:SUIUSD",
    "link": "X:LINKUSD",
    "avax": "X:AVAXUSD",
    "xpr": "X:XRPUSD",
    "doge": "X:DOGEUSD",
}

@dataclass
class BacktestConfig:
    ticker_map: Dict[str, str] = None
    adjusted: bool = True            # use adjusted data
    use_same_day_weights: bool = False
    # False (default) = allocate at the CLOSE of day t to target weights for day t+1 (no look-ahead)
    # True  = allocate to given weights at the open of the SAME day (optimistic / potential look-ahead)
    normalize_weights: bool = True   # if row weights != 1, normalize to 1.0
    infer_cash: bool = True          # if "cash" missing, fill cash = 1 - sum(crypto weights)
    cash_daily_return: float = 0.0   # daily return for cash (0 by default)
    trading_cost_bps: float = 2.0    # one-way cost in bps applied to traded weight; set 0 to ignore

    def __post_init__(self):
        if self.ticker_map is None:
            self.ticker_map = DEFAULT_TICKER_MAP


# ------------------------------------
# Polygon helpers
# ------------------------------------
def fetch_daily_bars(ticker: str, start_date: str, end_date: str, api_key: str, adjusted: bool = True) -> pd.DataFrame:
    """
    Fetch daily OHLC bars for a crypto ticker between [start_date, end_date], inclusive.
    Dates are 'YYYY-MM-DD'. Returns DataFrame indexed by date (YYYY-MM-DD) with 'close'.
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", []) or []
    if not results:
        return pd.DataFrame(columns=["close"])

    df = pd.DataFrame(results)
    # 't' is ms since epoch (UTC). Convert to date string YYYY-MM-DD
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date.astype(str)
    df = df[["date", "c"]].rename(columns={"c": "close"})
    df = df.set_index("date").sort_index()
    return df


def fetch_price_panel(tickers: Dict[str, str], start_date: str, end_date: str, api_key: str, adjusted: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame of close prices with index=date (YYYY-MM-DD) and columns per asset key (e.g., 'btc','eth').
    """
    frames = []
    for key, polygon_ticker in tickers.items():
        bars = fetch_daily_bars(polygon_ticker, start_date, end_date, api_key, adjusted=adjusted)
        bars = bars.rename(columns={"close": key})
        frames.append(bars)

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1).sort_index()
    return prices


# ------------------------------------
# Backtest core
# ------------------------------------
def prepare_allocations(alloc_json: List[Dict], assets: List[str], cfg: BacktestConfig) -> pd.DataFrame:
    """
    Convert the JSON list into a tidy weights DataFrame indexed by date with columns = assets + ['cash'].
    Handles missing assets (-> 0), optionally infers/normalizes cash, and sorts by ascending date.
    """
    df = pd.DataFrame(alloc_json).copy()

    # Ensure date string format & ascending order
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure columns for all requested assets + cash exist
    for a in assets:
        if a not in df.columns:
            df[a] = 0.0
    if "cash" not in df.columns:
        df["cash"] = 0.0

    # Clean NaNs -> 0
    weight_cols = assets + ["cash"]
    df[weight_cols] = df[weight_cols].fillna(0.0)

    # Optionally infer cash if not given or if totals != 1
    # We'll first compute crypto sum; if infer_cash, set cash = max(0, 1 - sum_crypto)
    if cfg.infer_cash:
        crypto_sum = df[assets].sum(axis=1)
        df["cash"] = np.where(df["cash"] == 0.0, np.maximum(0.0, 1.0 - crypto_sum), df["cash"])

    # Optionally normalize to exactly 1.0
    if cfg.normalize_weights:
        total = df[weight_cols].sum(axis=1)
        # Avoid divide-by-zero: if total==0, make all zero except cash=1
        zero_mask = (total <= 0)
        if zero_mask.any():
            df.loc[zero_mask, "cash"] = 1.0
            total = df[weight_cols].sum(axis=1)

        df[weight_cols] = (df[weight_cols].T / total.values).T

    return df.set_index("date")[weight_cols]


def compute_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: BacktestConfig
) -> pd.DataFrame:
    """
    Compute daily portfolio returns given price closes and daily target weights.
    - If use_same_day_weights=False (default), we apply weights.shift(1) to avoid look-ahead.
    - Cash earns cfg.cash_daily_return.
    - Transaction costs modeled as: cost = sum(abs(delta_w)) * (bps/10,000) on rebalancing days.
    """
    # Daily simple returns for assets
    asset_rets = prices.pct_change().fillna(0.0)

    # Align weights to price index
    weights = weights.reindex(prices.index).ffill().fillna(0.0)

    # Split weights into crypto vs cash
    cash_w = weights["cash"].clip(lower=0.0)
    crypto_cols = [c for c in weights.columns if c != "cash"]
    crypto_w = weights[crypto_cols].clip(lower=0.0)

    # Apply conservative no-lookahead by default:
    if not cfg.use_same_day_weights:
        cash_w = cash_w.shift(1)
        crypto_w = crypto_w.shift(1)

    # Fill pre-start with zeros (no exposure before first weight is defined)
    cash_w = cash_w.fillna(0.0)
    crypto_w = crypto_w.fillna(0.0)

    # Portfolio return (before costs)
    cash_ret_series = pd.Series(cfg.cash_daily_return, index=asset_rets.index)
    crypto_ret_series = (crypto_w * asset_rets[crypto_cols]).sum(axis=1)
    gross_ret = cash_w * cash_ret_series + crypto_ret_series

    # Transaction costs upon rebalancing
    # Define traded weight per day = sum(abs(w_t - w_{t-1})).
    w_full = pd.concat([crypto_w, cash_w.rename("cash")], axis=1)
    dw = w_full.diff().abs().sum(axis=1).fillna(0.0)
    cost = (cfg.trading_cost_bps / 10000.0) * dw
    net_ret = gross_ret - cost

    out = pd.DataFrame({
        "gross_return": gross_ret,
        "cost": cost,
        "net_return": net_ret
    })
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def performance_stats(equity: pd.Series, daily_returns: pd.Series) -> Dict[str, float]:
    """
    Basic performance metrics from a daily equity curve and daily return series.
    """
    if equity.empty:
        return {}

    n_days = len(equity)
    total_return = equity.iloc[-1] - 1.0
    # Annualization factor for daily data (crypto is 365d)
    ann_factor = 365.0
    # Protect against <=0
    cagr = (equity.iloc[-1]) ** (ann_factor / max(1.0, n_days)) - 1.0 if equity.iloc[-1] > 0 else np.nan

    vol = daily_returns.std(ddof=0) * math.sqrt(ann_factor)
    sharpe = (daily_returns.mean() * ann_factor) / vol if vol > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1.0).min()

    return {
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Volatility (ann)": float(vol),
        "Sharpe (rf=0)": float(sharpe),
        "Max Drawdown": float(max_dd),
        "Bars": int(n_days),
    }


# ------------------------------------
# Orchestration
# ------------------------------------
def backtest_from_json(
    alloc_json_str: str,
    cfg: BacktestConfig = BacktestConfig(),
    api_key: str = POLYGON_API_KEY
) -> Dict[str, pd.DataFrame]:
    """
    alloc_json_str: a JSON string representing a list of {date, asset weights..., cash?}
    Returns dict with 'prices', 'weights', 'results', 'stats'
    """
    alloc_list = json.loads(alloc_json_str)

    # Gather all crypto assets mentioned across rows (ignore 'date' & 'cash')
    asset_keys = sorted({
        k.lower()
        for row in alloc_list
        for k in row.keys()
        if k.lower() not in ("date", "cash")
    })

    # Build ticker map for just the assets we need
    tickers = {}
    for a in asset_keys:
        if a in cfg.ticker_map:
            tickers[a] = cfg.ticker_map[a]
        else:
            # Fallback: assume X:{SYM}USD (e.g., 'ada' -> 'X:ADAUSD')
            tickers[a] = f"X:{a.upper()}USD"

    # Determine date range
    dates = sorted({pd.to_datetime(r["date"]).date() for r in alloc_list})
    start_date = dates[0].isoformat()
    end_date = dates[-1].isoformat()

    alloc_json_str = Path("data.json").read_text(encoding="utf-8")
    prices = fetch_prices_with_cache(
        alloc_json_str,
        api_key=os.getenv("POLYGON_API_KEY"),
        # optional: override/extend symbols
        # ticker_map={"sol": "X:SOLUSD", "eth": "X:ETHUSD", "btc": "X:BTCUSD"},
    )
    # Prepare target weights table
    weights = prepare_allocations(alloc_list, list(tickers.keys()), cfg)

    # Compute returns & equity
    results = compute_portfolio_returns(prices, weights, cfg)
    stats = performance_stats(results["equity"], results["net_return"])

    return {
        "prices": prices,
        "weights": weights,
        "results": results,
        "stats": pd.DataFrame(stats, index=["strategy"]).T,
    }


# ------------------------------------
# Example usage
# ------------------------------------
if __name__ == "__main__":
    data_json_str = Path("data.json").read_text(encoding="utf-8")

    # Config: no-lookahead, normalize weights to 1, infer cash if missing, 2 bps trading cost
    cfg = BacktestConfig(
        use_same_day_weights=False,
        normalize_weights=True,
        infer_cash=True,
        cash_daily_return=0.0,
        trading_cost_bps=2.0,
    )

    result = backtest_from_json(data_json_str, cfg, api_key=POLYGON_API_KEY)
    print("\n=== STATS ===")
    print(result["stats"])

    print("\n=== HEAD(results) ===")
    print(result["results"].head())

    # Save CSVs if you like
    # result["results"].to_csv("bt_results.csv")
    # result["weights"].to_csv("bt_weights.csv")
    # result["prices"].to_csv("bt_prices.csv")
