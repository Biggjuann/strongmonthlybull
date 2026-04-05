# CVD Bias Bullish Entry Screener

Stock screener that identifies S&P 500 and NASDAQ 100 stocks entering a bullish state (green background) on the monthly timeframe, based on the Pine Script "ES/NQ Intraday CVD Bias Reader" indicator.

Screens for stocks that have **just entered** weak bull or strong bull within the last 1-3 months.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source venv/bin/activate
python3 app.py
```

Open `http://localhost:5000` in your browser, then click **Force Fresh Scan**.

The scan screens 515 tickers and takes a few minutes to complete. Results are cached for 6 hours — use **Force Fresh Scan** to bypass the cache.

## Debug a Single Ticker

While the app is running, check any ticker's monthly bias history:

```
http://localhost:5000/api/debug/AAPL
```

Or from the terminal:

```bash
python3 -c "
from screener import debug_ticker
import json
r = debug_ticker('AAPL')
for m in r.get('months', [])[-6:]:
    print(json.dumps(m, indent=2))
"
```

## How It Works

The indicator computes a bias state for each monthly bar:

- **Strong Bull** (bright green) — price above EMA, CVD rising above its MA, above VWAP, not choppy
- **Weak Bull** (darker green) — partial bullish alignment (above EMA with CVD above MA, or above VWAP with CVD rising)
- **Neutral** — no clear direction or chop filter active
- **Weak Bear** (orange) — partial bearish alignment
- **Strong Bear** (red) — full bearish alignment

Key components:
- **CVD (Cumulative Volume Delta)** — proxy using `volume * (close - open) / (high - low)`, smoothed with 5-period EMA, then cumulative sum
- **Price EMA** — 20-period exponential moving average
- **VWAP** — volume-weighted average price, anchored yearly
- **Chop Filter** — filters out sideways moves where `|close - EMA| / ATR < 0.35`
- **ATR** — 14-period Average True Range using Wilder's RMA

## Files

| File | Purpose |
|------|---------|
| `screener.py` | Indicator logic, single-ticker scanner, full universe scanner |
| `app.py` | Flask web server with scan API and debug endpoint |
| `tickers.py` | Hardcoded S&P 500 + NASDAQ 100 ticker lists |
| `sample_data.py` | Demo data fallback when Yahoo Finance is unavailable |
| `templates/index.html` | Dashboard UI |
