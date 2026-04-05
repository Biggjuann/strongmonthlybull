"""
Flask dashboard for the Weak Bull Monthly Screener.

Supports live scanning via Yahoo Finance or demo mode with sample data
when the API is unavailable.
"""

from flask import Flask, render_template, jsonify, request
from screener import run_screener, save_cache, load_cache, debug_ticker
from sample_data import SAMPLE_RESULTS
from datetime import datetime
import threading
import logging
import os

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set DEMO_MODE=1 env var to skip live API calls
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

# Global state for background scan
scan_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "results": [],
    "last_updated": None,
    "error": None,
    "demo": DEMO_MODE,
}
scan_lock = threading.Lock()


def background_scan(force: bool = False):
    """Run the screener in background."""
    with scan_lock:
        if scan_state["running"]:
            return
        scan_state["running"] = True
        scan_state["error"] = None

    try:
        # Demo mode: use sample data
        if scan_state["demo"]:
            with scan_lock:
                scan_state["results"] = SAMPLE_RESULTS
                scan_state["last_updated"] = datetime.now().isoformat()
                scan_state["running"] = False
            return

        if not force:
            cached = load_cache(max_age_hours=6)
            if cached is not None:
                with scan_lock:
                    scan_state["results"] = cached
                    scan_state["last_updated"] = datetime.now().isoformat()
                    scan_state["running"] = False
                return

        results = run_screener(max_workers=8)
        save_cache(results)
        with scan_lock:
            scan_state["results"] = results
            scan_state["last_updated"] = datetime.now().isoformat()
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        # Fallback to sample data on error
        with scan_lock:
            scan_state["error"] = f"{e} — showing demo data"
            scan_state["results"] = SAMPLE_RESULTS
            scan_state["last_updated"] = datetime.now().isoformat()
            scan_state["demo"] = True
    finally:
        with scan_lock:
            scan_state["running"] = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def start_scan():
    force = request.json.get("force", False) if request.is_json else False
    t = threading.Thread(target=background_scan, args=(force,), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def status():
    with scan_lock:
        return jsonify({
            "running": scan_state["running"],
            "count": len(scan_state["results"]),
            "last_updated": scan_state["last_updated"],
            "error": scan_state["error"],
            "demo": scan_state["demo"],
        })


@app.route("/api/results")
def results():
    with scan_lock:
        data = list(scan_state["results"])

    # Filtering
    sector = request.args.get("sector", "")
    months = request.args.get("months", "")
    sort_by = request.args.get("sort", "months_bullish")
    sort_dir = request.args.get("dir", "asc")

    filtered = data
    if sector:
        filtered = [r for r in filtered if r.get("sector", "") == sector]
    if months:
        try:
            m = int(months)
            filtered = [r for r in filtered if r.get("months_bullish") == m]
        except ValueError:
            pass

    reverse = sort_dir == "desc"
    try:
        filtered.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)
    except Exception:
        pass

    # Gather unique sectors for filter dropdown
    sectors = sorted(set(r.get("sector", "N/A") for r in data))

    return jsonify({
        "results": filtered,
        "total": len(filtered),
        "sectors": sectors,
        "last_updated": scan_state.get("last_updated"),
        "demo": scan_state.get("demo", False),
    })


@app.route("/api/debug/<ticker>")
def debug_single(ticker):
    """Show full monthly bias history for a ticker. Usage: /api/debug/AAPL"""
    result = debug_ticker(ticker.upper())
    result["ticker"] = ticker.upper()
    return jsonify(result)


# Boot: load cache if available (works under both gunicorn and direct run)
cached = load_cache(max_age_hours=6)
if cached is not None:
    scan_state["results"] = cached
    scan_state["last_updated"] = datetime.now().isoformat()
    logger.info(f"Loaded {len(cached)} cached results on boot")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
