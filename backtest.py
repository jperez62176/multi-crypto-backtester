import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from storage import fetch_prices_with_cache
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


load_dotenv() 
# ------------------------------------
# Config
# ------------------------------------
# Prefer env var to avoid committing secrets
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

@dataclass
class BacktestConfig:
    ticker_map: Dict[str, str] = None     # not used by storage, only for weight columns if you want
    adjusted: bool = True                 # use adjusted data (handled by storage)
    use_same_day_weights: bool = False    # no look-ahead by default
    normalize_weights: bool = True
    infer_cash: bool = True
    cash_daily_return: float = 0.0
    trading_cost_bps: float = 2.0

    def __post_init__(self):
        if self.ticker_map is None:
            self.ticker_map = {}  # storage builds its own map


# ------------------------------------
# Backtest core
# ------------------------------------
def prepare_allocations(alloc_json: List[Dict], assets: List[str], cfg: BacktestConfig) -> pd.DataFrame:
    df = pd.DataFrame(alloc_json).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.sort_values("date").reset_index(drop=True)

    for a in assets:
        if a not in df.columns:
            df[a] = 0.0
    if "cash" not in df.columns:
        df["cash"] = 0.0

    weight_cols = assets + ["cash"]
    df[weight_cols] = df[weight_cols].fillna(0.0)

    if cfg.infer_cash:
        crypto_sum = df[assets].sum(axis=1)
        df["cash"] = np.where(df["cash"] == 0.0, np.maximum(0.0, 1.0 - crypto_sum), df["cash"])

    if cfg.normalize_weights:
        total = df[weight_cols].sum(axis=1)
        zero_mask = (total <= 0)
        if zero_mask.any():
            df.loc[zero_mask, "cash"] = 1.0
            total = df[weight_cols].sum(axis=1)
        df[weight_cols] = (df[weight_cols].T / total.values).T

    return df.set_index("date")[weight_cols]


def compute_portfolio_returns(prices: pd.DataFrame, weights: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    asset_rets = prices.pct_change().fillna(0.0)

    weights = weights.reindex(prices.index).ffill().fillna(0.0)

    cash_w = weights["cash"].clip(lower=0.0)
    crypto_cols = [c for c in weights.columns if c != "cash"]
    crypto_w = weights[crypto_cols].clip(lower=0.0)

    if not cfg.use_same_day_weights:
        cash_w = cash_w.shift(1)
        crypto_w = crypto_w.shift(1)

    cash_w = cash_w.fillna(0.0)
    crypto_w = crypto_w.fillna(0.0)

    cash_ret_series = pd.Series(cfg.cash_daily_return, index=asset_rets.index)
    crypto_ret_series = (crypto_w * asset_rets[crypto_cols]).sum(axis=1)
    gross_ret = cash_w * cash_ret_series + crypto_ret_series

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
    if equity.empty:
        return {}

    n_days = len(equity)
    total_return = equity.iloc[-1] - 1.0
    ann_factor = 365.0
    cagr = (equity.iloc[-1]) ** (ann_factor / max(1.0, n_days)) - 1.0 if equity.iloc[-1] > 0 else float("nan")

    vol = daily_returns.std(ddof=0) * np.sqrt(ann_factor)
    sharpe = (daily_returns.mean() * ann_factor) / vol if vol > 0 else float("nan")
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
def backtest_from_json(alloc_json_str: str, cfg: BacktestConfig = BacktestConfig(), api_key: str = POLYGON_API_KEY):
    # 1) Prices from the cache (storage figures out date range + tickers)
    prices = fetch_prices_with_cache(
        alloc_json_str,
        api_key=api_key,
        # Optional: supply explicit ticker overrides if you have symbols beyond btc/eth/sol:
        # ticker_map={"sui":"X:SUIUSD","link":"X:LINKUSD","avax":"X:AVAXUSD","xrp":"X:XRPUSD","doge":"X:DOGEUSD"}
    )

    # 2) Build weights from the JSON for exactly the assets we have prices for
    alloc_list = json.loads(alloc_json_str)
    assets = list(prices.columns)  # ensures alignment with price data
    weights = prepare_allocations(alloc_list, assets, cfg)

    # 3) Returns/equity + stats
    results = compute_portfolio_returns(prices, weights, cfg)
    stats = performance_stats(results["equity"], results["net_return"])

    return {
        "prices": prices,
        "weights": weights,
        "results": results,
        "stats": pd.DataFrame(stats, index=["strategy"]).T,
    }

def plot_equity(results: pd.DataFrame, starting_capital: float = 100_000.0, out_file: str = "equity_curve.png"):
    """Plot and save the portfolio equity (in USD)."""
    if results.empty or "equity" not in results.columns:
        raise ValueError("results must contain an 'equity' column")

    df = results.copy()
    # Ensure datetime index for nicer x-axis
    df.index = pd.to_datetime(df.index)

    df["equity_value"] = starting_capital * df["equity"]

    plt.figure()
    ax = df["equity_value"].plot()
    ax.set_title("Portfolio Equity")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    try:
        plt.show()
    except Exception:
        # Headless environments: ignore display errors
        pass
    return out_file


# ------------------------------------
# Example usage
# ------------------------------------
if __name__ == "__main__":
    data_json_str = Path("data.json").read_text(encoding="utf-8")

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
    png = plot_equity(result["results"], starting_capital=100_000.0, out_file="equity_curve.png")
    print(f"\nSaved equity chart to {png}")
