"""
Stock Screener: Weak Bull Detection on Monthly Timeframe

Translates the Pine Script "ES/NQ Intraday CVD Bias Reader" indicator
to Python, applied on monthly bars. Screens S&P 500 and NASDAQ 100
stocks to find those that JUST entered weak bull within the last 1-3 months.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import concurrent.futures
import logging
import json
import os

from tickers import get_combined_universe

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


# ── Technical helpers ───────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average matching Pine Script ta.ema."""
    return series.ewm(span=span, adjust=False).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder's smoothed moving average (RMA) matching Pine Script ta.rma.
    Equivalent to EMA with alpha = 1/length.
    """
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range using Wilder's RMA — matches Pine Script ta.atr."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, length)


# ── Core indicator logic ────────────────────────────────────────────────────
def compute_bias(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Compute the CVD Bias indicator on OHLCV data.
    Matches Pine Script logic exactly, including VWAP.

    Parameters
    ----------
    df : DataFrame with columns [Open, High, Low, Close, Volume]
    params : dict of indicator parameters (uses DEFAULTS if None)

    Returns
    -------
    DataFrame with added bias columns.
    """
    p = {**DEFAULTS, **(params or {})}
    df = df.copy()

    # Need enough bars for calculations
    if len(df) < max(p["cvd_ma_len"], p["price_ma_len"], p["atr_len"]) + 5:
        df["bias"] = "INSUFFICIENT_DATA"
        return df

    # ── Delta proxy and CVD ─────────────────────────────────────────────
    # Pine: range_ = math.max(high - low, syminfo.mintick)
    range_ = (df["High"] - df["Low"]).clip(lower=1e-10)
    # Pine: body_ = close - open
    body = df["Close"] - df["Open"]
    # Pine: bodyPct = body_ / range_
    body_pct = body / range_
    # Pine: deltaRaw = volume * bodyPct
    delta_raw = df["Volume"] * body_pct
    # Pine: deltaSmooth = ta.ema(deltaRaw, deltaSmoothLen)
    delta_smooth = ema(delta_raw, p["delta_smooth_len"])
    # Pine: cvd = ta.cum(deltaSmooth)
    cvd = delta_smooth.cumsum()
    # Pine: cvdMA = ta.ema(cvd, cvdMaLen)
    cvd_ma = ema(cvd, p["cvd_ma_len"])

    # ── Price structure ─────────────────────────────────────────────────
    # Pine: priceMA = ta.ema(close, priceMaLen)
    price_ma = ema(df["Close"], p["price_ma_len"])

    # Pine: vwapValue = ta.vwap(close)
    # ta.vwap(source) = cumsum(source * volume) / cumsum(volume)
    cum_vol = df["Volume"].cumsum()
    cum_pv = (df["Close"] * df["Volume"]).cumsum()
    vwap_value = cum_pv / cum_vol.clip(lower=1)

    # Pine: priceAboveMA = close > priceMA
    price_above_ma = df["Close"] > price_ma
    price_below_ma = df["Close"] < price_ma

    # Pine: priceAboveVWAP = close > vwapValue
    price_above_vwap = df["Close"] > vwap_value
    price_below_vwap = df["Close"] < vwap_value

    # Pine: cvdUp = cvd > cvdMA and cvd > cvd[1]
    cvd_up = (cvd > cvd_ma) & (cvd > cvd.shift(1))
    # Pine: cvdDown = cvd < cvdMA and cvd < cvd[1]
    cvd_down = (cvd < cvd_ma) & (cvd < cvd.shift(1))

    # Pine: bullCore = priceAboveMA and cvdUp and (not useVWAP or priceAboveVWAP)
    # useVWAP=true → requires priceAboveVWAP
    bull_core = price_above_ma & cvd_up & price_above_vwap
    # Pine: bearCore = priceBelowMA and cvdDown and (not useVWAP or priceBelowVWAP)
    bear_core = price_below_ma & cvd_down & price_below_vwap

    # Pine: bullWeak = (priceAboveMA and cvd > cvdMA) or (useVWAP and priceAboveVWAP and cvdUp)
    bull_weak = (price_above_ma & (cvd > cvd_ma)) | (price_above_vwap & cvd_up)
    # Pine: bearWeak = (priceBelowMA and cvd < cvdMA) or (useVWAP and priceBelowVWAP and cvdDown)
    bear_weak = (price_below_ma & (cvd < cvd_ma)) | (price_below_vwap & cvd_down)

    # ── Chop filter ─────────────────────────────────────────────────────
    # Pine: atrValue = ta.atr(atrLen)  — uses Wilder's RMA
    atr_value = atr(df["High"], df["Low"], df["Close"], p["atr_len"])
    # Pine: trendRange = math.abs(close - priceMA)
    trend_range = (df["Close"] - price_ma).abs()
    # Pine: trendRangePct = atrValue > 0 ? trendRange / atrValue : 0.0
    trend_range_pct = np.where(atr_value > 0, trend_range / atr_value, 0.0)
    # Pine: isChop = useChopFilter ? trendRangePct < chopThresholdPct : false
    is_chop = pd.Series(
        p["use_chop_filter"] & (trend_range_pct < p["chop_threshold_pct"]),
        index=df.index,
    )

    # ── Bias states (exact Pine Script logic) ───────────────────────────
    # Pine: strongBull = bullCore and not isChop
    strong_bull = bull_core & ~is_chop
    # Pine: weakBull = not strongBull and bullWeak and not bearCore and not isChop
    weak_bull = ~strong_bull & bull_weak & ~bear_core & ~is_chop
    # Pine: strongBear = bearCore and not isChop
    strong_bear = bear_core & ~is_chop
    # Pine: weakBear = not strongBear and bearWeak and not bullCore and not isChop
    weak_bear = ~strong_bear & bear_weak & ~bull_core & ~is_chop
    # Pine: neutral = not strongBull and not weakBull and not strongBear and not weakBear
    neutral = ~strong_bull & ~weak_bull & ~strong_bear & ~weak_bear

    # Store results
    df["strong_bull"] = strong_bull
    df["weak_bull"] = weak_bull
    df["strong_bear"] = strong_bear
    df["weak_bear"] = weak_bear
    df["neutral"] = neutral
    df["is_chop"] = is_chop
    df["price_ma"] = price_ma
    df["vwap"] = vwap_value
    df["cvd"] = cvd
    df["cvd_ma"] = cvd_ma
    df["atr"] = atr_value
    df["trend_range_pct"] = trend_range_pct

    # String bias label
    df["bias"] = "NEUTRAL"
    df.loc[strong_bull, "bias"] = "STRONG BULL"
    df.loc[weak_bull, "bias"] = "WEAK BULL"
    df.loc[strong_bear, "bias"] = "STRONG BEAR"
    df.loc[weak_bear, "bias"] = "WEAK BEAR"

    return df


# ── Debug: full bias history for a single ticker ──────────────────────────
def debug_ticker(ticker: str, params: dict | None = None) -> list[dict]:
    """Return the full monthly bias history for a ticker (for debugging)."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", interval="1mo")
        if df is None or df.empty:
            return [{"error": f"No data for {ticker}"}]

        df = df.reset_index()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

        df = compute_bias(df, params)

        rows = []
        for i in range(len(df)):
            r = df.iloc[i]
            rows.append({
                "date": str(r.get("Date", "")),
                "open": round(float(r["Open"]), 2),
                "high": round(float(r["High"]), 2),
                "low": round(float(r["Low"]), 2),
                "close": round(float(r["Close"]), 2),
                "volume": int(r["Volume"]),
                "bias": str(r.get("bias", "N/A")),
                "is_chop": bool(r.get("is_chop", False)),
                "price_ma": round(float(r.get("price_ma", 0)), 2),
                "vwap": round(float(r.get("vwap", 0)), 2),
                "price_vs_vwap": "ABOVE" if r["Close"] > r.get("vwap", 0) else "BELOW",
                "cvd": round(float(r.get("cvd", 0)), 2),
                "cvd_ma": round(float(r.get("cvd_ma", 0)), 2),
                "atr": round(float(r.get("atr", 0)), 2),
                "trend_range_pct": round(float(r.get("trend_range_pct", 0)), 4),
            })
        return rows
    except Exception as e:
        return [{"error": str(e)}]


# ── Single-stock scanner ───────────────────────────────────────────────────
def scan_ticker(ticker: str, params: dict | None = None) -> dict | None:
    """
    Download monthly data for *ticker* and return screening result.

    Returns a dict with screening info if the stock JUST entered
    weak bull (1-3 months), filtering out stocks that have been
    bullish (weak or strong) for longer.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", interval="1mo")

        if df is None or df.empty:
            return None

        df = df.reset_index()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

        # Keep the current month's bar (even though incomplete) to match
        # TradingView real-time behavior. TV evaluates the current bar live.

        # Need enough bars for EMA calculations
        p = params or DEFAULTS
        min_bars = max(
            p.get("cvd_ma_len", DEFAULTS["cvd_ma_len"]),
            p.get("price_ma_len", DEFAULTS["price_ma_len"]),
            p.get("atr_len", DEFAULTS["atr_len"]),
        ) + 5
        if len(df) < min_bars:
            return None

        df = compute_bias(df, params)

        if "bias" not in df.columns or df["bias"].iloc[0] == "INSUFFICIENT_DATA":
            return None

        # Must be currently weak bull on the last completed month
        if not df["weak_bull"].iloc[-1]:
            return None

        # Count consecutive months of ANY bullish state (weak bull OR strong bull)
        # from the end. This prevents stocks that have been bullish for a long
        # time (flipping between strong/weak bull) from appearing as new entries.
        bullish_streak = 0
        for i in range(len(df) - 1, -1, -1):
            if df["weak_bull"].iloc[i] or df["strong_bull"].iloc[i]:
                bullish_streak += 1
            else:
                break

        # Only want stocks that JUST turned bullish within 1-3 months
        if bullish_streak > 3:
            return None

        # Also count consecutive weak_bull specifically
        weak_bull_streak = 0
        for i in range(len(df) - 1, -1, -1):
            if df["weak_bull"].iloc[i]:
                weak_bull_streak += 1
            else:
                break

        # Gather info
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) >= 2 else last_row
        entry_idx = len(df) - bullish_streak
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
            "current_bias": str(last_row["bias"]),
            "previous_bias": prev_bias,
            "months_in_weak_bull": weak_bull_streak,
            "bullish_months_total": bullish_streak,
            "entry_date": str(entry_row.get("Date", "N/A")),
            "current_price": round(float(last_row["Close"]), 2),
            "price_ma": round(float(last_row["price_ma"]), 2),
            "cvd_vs_ma": "ABOVE" if last_row["cvd"] > last_row["cvd_ma"] else "BELOW",
            "price_vs_ma": "ABOVE" if last_row["Close"] > last_row["price_ma"] else "BELOW",
            "vwap": round(float(last_row["vwap"]), 2),
            "price_vs_vwap": "ABOVE" if last_row["Close"] > last_row["vwap"] else "BELOW",
            "is_chop": bool(last_row["is_chop"]),
            "atr": round(float(last_row["atr"]), 2),
            "price_change_pct": round(
                ((last_row["Close"] - prev_row["Close"]) / prev_row["Close"]) * 100, 2
            ) if prev_row["Close"] > 0 else 0,
        }

    except Exception as e:
        logger.warning(f"Error scanning {ticker}: {e}")
        return None


# ── Full universe scan ─────────────────────────────────────────────────────
def run_screener(max_workers: int = 10, params: dict | None = None) -> list[dict]:
    """
    Screen all S&P 500 + NASDAQ 100 stocks for weak bull entry.

    Returns list of dicts sorted by months_in_weak_bull (ascending),
    then by ticker.
    """
    tickers = get_combined_universe()
    logger.info(f"Scanning {len(tickers)} unique tickers")

    results = []
    errors = 0
    skipped = 0
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
                logger.info(f"Progress: {done}/{total} | matches: {len(results)}")
            try:
                result = future.result()
            except Exception as e:
                errors += 1
                logger.warning(f"Future error: {e}")
                continue
            if result is not None:
                results.append(result)
            else:
                skipped += 1

    results.sort(key=lambda x: (x["months_in_weak_bull"], x["ticker"]))
    logger.info(
        f"Scan complete. {len(results)} weak bull | "
        f"{skipped} filtered out | {errors} errors | {total} total"
    )
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
              f"WB: {r['months_in_weak_bull']}mo | Bull: {r['bullish_months_total']}mo | "
              f"Price: ${r['current_price']:>8.2f} | "
              f"From: {r['previous_bias']}")
