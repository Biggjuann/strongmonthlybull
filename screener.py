"""
Stock Screener: Weak Bull Detection on Monthly Timeframe

Translates the Pine Script "ES/NQ Intraday CVD Bias Reader" indicator
to Python, applied on monthly bars. Screens S&P 500 and NASDAQ 100
stocks to find those that entered weak bull within the last 1-3 months.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import concurrent.futures
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Indicator default parameters (matching Pine Script inputs) ──────────────
DEFAULTS = {
    "delta_smooth_len": 5,
    "cvd_ma_len": 21,
    "price_ma_len": 20,
    "atr_len": 14,
    "chop_threshold_pct": 0.35,
    "use_chop_filter": True,
}


# ── Universe helpers ────────────────────────────────────────────────────────
def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list: {e}")
        return []


def get_nasdaq100_tickers() -> list[str]:
    """Fetch current NASDAQ 100 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        # The table with tickers is typically the 4th table
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].str.replace(".", "-", regex=False).tolist()
                logger.info(f"Fetched {len(tickers)} NASDAQ 100 tickers")
                return tickers
            elif "Symbol" in table.columns:
                tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
                logger.info(f"Fetched {len(tickers)} NASDAQ 100 tickers")
                return tickers
        # Fallback: try first table with a column that looks like tickers
        logger.warning("Could not find Ticker/Symbol column, trying fallback")
        return []
    except Exception as e:
        logger.error(f"Failed to fetch NASDAQ 100 list: {e}")
        return []


def get_combined_universe() -> list[str]:
    """Return deduplicated list of S&P 500 + NASDAQ 100 tickers."""
    sp500 = get_sp500_tickers()
    nq100 = get_nasdaq100_tickers()
    combined = sorted(set(sp500 + nq100))
    logger.info(f"Combined universe: {len(combined)} unique tickers")
    return combined


# ── Technical helpers ───────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average matching Pine Script ta.ema."""
    return series.ewm(span=span, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=length, min_periods=1).mean()


# ── Core indicator logic ────────────────────────────────────────────────────
def compute_bias(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Compute the CVD Bias indicator on OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns [Open, High, Low, Close, Volume]
    params : dict of indicator parameters (uses DEFAULTS if None)

    Returns
    -------
    DataFrame with added columns: bias, strong_bull, weak_bull, etc.
    """
    p = {**DEFAULTS, **(params or {})}
    df = df.copy()

    # Need enough bars for calculations
    if len(df) < max(p["cvd_ma_len"], p["price_ma_len"], p["atr_len"]) + 5:
        df["bias"] = "INSUFFICIENT_DATA"
        return df

    # ── Delta proxy and CVD ─────────────────────────────────────────────
    range_ = (df["High"] - df["Low"]).clip(lower=1e-10)
    body = df["Close"] - df["Open"]
    body_pct = body / range_
    delta_raw = df["Volume"] * body_pct
    delta_smooth = ema(delta_raw, p["delta_smooth_len"])
    cvd = delta_smooth.cumsum()
    cvd_ma = ema(cvd, p["cvd_ma_len"])

    # ── Price structure ─────────────────────────────────────────────────
    price_ma = ema(df["Close"], p["price_ma_len"])

    price_above_ma = df["Close"] > price_ma
    price_below_ma = df["Close"] < price_ma

    cvd_up = (cvd > cvd_ma) & (cvd > cvd.shift(1))
    cvd_down = (cvd < cvd_ma) & (cvd < cvd.shift(1))

    # Core conditions (no VWAP on monthly — VWAP is intraday only)
    bull_core = price_above_ma & cvd_up
    bear_core = price_below_ma & cvd_down

    # Weak conditions
    bull_weak = (price_above_ma & (cvd > cvd_ma)) | cvd_up
    bear_weak = (price_below_ma & (cvd < cvd_ma)) | cvd_down

    # ── Chop filter ─────────────────────────────────────────────────────
    atr_value = atr(df["High"], df["Low"], df["Close"], p["atr_len"])
    trend_range = (df["Close"] - price_ma).abs()
    trend_range_pct = np.where(atr_value > 0, trend_range / atr_value, 0.0)
    is_chop = pd.Series(
        p["use_chop_filter"] & (trend_range_pct < p["chop_threshold_pct"]),
        index=df.index,
    )

    # ── Bias states ─────────────────────────────────────────────────────
    strong_bull = bull_core & ~is_chop
    weak_bull = ~strong_bull & bull_weak & ~bear_core & ~is_chop
    strong_bear = bear_core & ~is_chop
    weak_bear = ~strong_bear & bear_weak & ~bull_core & ~is_chop
    neutral = ~strong_bull & ~weak_bull & ~strong_bear & ~weak_bear

    # Store results
    df["strong_bull"] = strong_bull
    df["weak_bull"] = weak_bull
    df["strong_bear"] = strong_bear
    df["weak_bear"] = weak_bear
    df["neutral"] = neutral
    df["is_chop"] = is_chop
    df["price_ma"] = price_ma
    df["cvd"] = cvd
    df["cvd_ma"] = cvd_ma
    df["atr"] = atr_value

    # String bias label
    df["bias"] = "NEUTRAL"
    df.loc[strong_bull, "bias"] = "STRONG BULL"
    df.loc[weak_bull, "bias"] = "WEAK BULL"
    df.loc[strong_bear, "bias"] = "STRONG BEAR"
    df.loc[weak_bear, "bias"] = "WEAK BEAR"

    return df


# ── Single-stock scanner ───────────────────────────────────────────────────
def scan_ticker(ticker: str, params: dict | None = None) -> dict | None:
    """
    Download monthly data for *ticker* and return screening result.

    Returns a dict with screening info if the stock is currently in
    weak bull (entered within last 1-3 months), else None.
    """
    try:
        tk = yf.Ticker(ticker)
        # Get ~3 years of monthly data for enough EMA history
        df = tk.history(period="3y", interval="1mo")

        if df.empty or len(df) < 30:
            return None

        df = df.reset_index()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

        df = compute_bias(df, params)

        if "bias" not in df.columns or df["bias"].iloc[0] == "INSUFFICIENT_DATA":
            return None

        # Check the last 3 months for weak bull entry
        recent = df.tail(4)  # last 4 bars to check transitions
        if len(recent) < 2:
            return None

        current_bias = df["bias"].iloc[-1]
        # Find when weak bull started (looking at last 3 bars)
        last_3 = df.tail(3)
        weak_bull_months = last_3["weak_bull"].sum()

        if weak_bull_months == 0:
            return None

        # Determine how long in weak bull: count consecutive weak_bull from end
        consecutive = 0
        for i in range(len(df) - 1, -1, -1):
            if df["weak_bull"].iloc[i]:
                consecutive += 1
            else:
                break

        # Must be currently weak bull AND entered within 1-3 months
        if not df["weak_bull"].iloc[-1]:
            return None

        if consecutive < 1 or consecutive > 3:
            return None

        # Gather info
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) >= 2 else last_row
        entry_idx = len(df) - consecutive
        entry_row = df.iloc[entry_idx] if entry_idx < len(df) else last_row
        prev_bias = df["bias"].iloc[entry_idx - 1] if entry_idx > 0 else "N/A"

        # Get company name
        try:
            info = tk.info
            name = info.get("shortName", info.get("longName", ticker))
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            market_cap = info.get("marketCap", 0)
        except Exception:
            name = ticker
            sector = "N/A"
            industry = "N/A"
            market_cap = 0

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "current_bias": current_bias,
            "previous_bias": prev_bias,
            "months_in_weak_bull": consecutive,
            "entry_date": str(entry_row.get("Date", "N/A")),
            "current_price": round(float(last_row["Close"]), 2),
            "price_ma": round(float(last_row["price_ma"]), 2),
            "cvd_vs_ma": "ABOVE" if last_row["cvd"] > last_row["cvd_ma"] else "BELOW",
            "price_vs_ma": "ABOVE" if last_row["Close"] > last_row["price_ma"] else "BELOW",
            "is_chop": bool(last_row["is_chop"]),
            "atr": round(float(last_row["atr"]), 2),
            "price_change_pct": round(
                ((last_row["Close"] - prev_row["Close"]) / prev_row["Close"]) * 100, 2
            ) if prev_row["Close"] > 0 else 0,
        }

    except Exception as e:
        logger.debug(f"Error scanning {ticker}: {e}")
        return None


# ── Full universe scan ─────────────────────────────────────────────────────
def run_screener(max_workers: int = 10, params: dict | None = None) -> list[dict]:
    """
    Screen all S&P 500 + NASDAQ 100 stocks for weak bull entry.

    Returns list of dicts sorted by months_in_weak_bull (ascending),
    then by ticker.
    """
    tickers = get_combined_universe()
    if not tickers:
        logger.error("No tickers to scan")
        return []

    results = []
    total = len(tickers)
    done = 0

    logger.info(f"Scanning {total} tickers with {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(scan_ticker, t, params): t for t in tickers
        }
        for future in concurrent.futures.as_completed(future_map):
            done += 1
            if done % 50 == 0:
                logger.info(f"Progress: {done}/{total}")
            result = future.result()
            if result is not None:
                results.append(result)

    results.sort(key=lambda x: (x["months_in_weak_bull"], x["ticker"]))
    logger.info(f"Scan complete. Found {len(results)} stocks in weak bull (1-3 months)")
    return results


# ── Cache helpers ──────────────────────────────────────────────────────────
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener_cache.json")


def save_cache(results: list[dict]) -> None:
    data = {"timestamp": datetime.now().isoformat(), "results": results}
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_cache(max_age_hours: int = 6) -> list[dict] | None:
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        ts = datetime.fromisoformat(data["timestamp"])
        if datetime.now() - ts > timedelta(hours=max_age_hours):
            return None
        return data["results"]
    except Exception:
        return None


if __name__ == "__main__":
    results = run_screener()
    save_cache(results)
    print(f"\n{'='*80}")
    print(f"WEAK BULL SCREENER — Monthly Timeframe — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"Found {len(results)} stocks in weak bull (1-3 months)\n")
    for r in results:
        print(f"  {r['ticker']:6s} | {r['name'][:30]:30s} | "
              f"Months: {r['months_in_weak_bull']} | "
              f"Price: ${r['current_price']:>8.2f} | "
              f"From: {r['previous_bias']}")
