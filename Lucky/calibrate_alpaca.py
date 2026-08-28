"""Measure what the IEX feed actually costs before trusting research built on it.

The free Alpaca plan serves IEX prints only. This strategy's signal is a volume
ratio and its gate is VWAP, so a partial tape is not a cosmetic difference - it
can move the signal itself. Rather than argue about that in the abstract, this
script diffs Alpaca against the cached yfinance consolidated bars on the SAME
tickers and dates, and then re-runs the validated backtest on both.

Three questions, in increasing order of what matters:

  1. Bar coverage    - what fraction of consolidated 1-min bars exist on IEX?
  2. Volume coverage - on the bars that do exist, what fraction of the volume?
  3. Trade agreement - does the strategy fire the same trades and return the
                       same number? This is the only one that decides anything.

A high (1) and (2) with a broken (3) means the feed is unusable regardless of
how good the coverage headline looks.

    python -m Lucky.calibrate_alpaca [--feed iex|sip] [--tickers N]
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import gap_vwap_strategy as S

from .alpaca import Alpaca
from .config import CACHE_DIR, DATA_DIR

log = logging.getLogger(__name__)

BARS_PKL = DATA_DIR / "bars_1m_all.pkl"
EVENTS_CSV = DATA_DIR / "events_1_3.csv"


def load_yf_bars(limit: int | None = None) -> dict[str, pd.DataFrame]:
    with BARS_PKL.open("rb") as fh:
        bars = pickle.load(fh)
    bars = {t: d for t, d in bars.items() if d is not None and not d.empty}
    if limit:
        bars = dict(sorted(bars.items())[:limit])
    return bars


def window(bars: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo = min(d.index.min() for d in bars.values())
    hi = max(d.index.max() for d in bars.values())
    return lo, hi


def compare_bars(yf: pd.DataFrame, al: pd.DataFrame) -> dict:
    """Coverage of one ticker's Alpaca bars against its yfinance bars."""
    if al.empty:
        return {
            "yf_bars": len(yf), "al_bars": 0, "bar_coverage": 0.0,
            "vol_coverage": 0.0, "matched_bars": 0, "close_mae_bps": np.nan,
        }
    # Align on the minute. Both frames are tz-aware America/New_York.
    common = yf.index.intersection(al.index)
    yv = yf["Volume"].astype(float)
    av = al["Volume"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_cov = float(av.reindex(common).sum() / yv.reindex(common).sum()) if len(common) else 0.0
        close_mae = (
            float(((al["Close"].reindex(common) / yf["Close"].reindex(common) - 1).abs() * 1e4).mean())
            if len(common) else np.nan
        )
    return {
        "yf_bars": len(yf),
        "al_bars": len(al),
        "bar_coverage": len(common) / len(yf) if len(yf) else 0.0,
        "vol_coverage": vol_cov,
        "matched_bars": len(common),
        "close_mae_bps": close_mae,
    }


def run(feed: str = "iex", ticker_limit: int | None = None) -> pd.DataFrame:
    yf_bars = load_yf_bars(ticker_limit)
    lo, hi = window(yf_bars)
    log.info("yfinance cache: %d tickers, %s -> %s", len(yf_bars), lo.date(), hi.date())

    client = Alpaca(CACHE_DIR, feed=feed)
    log.info("fetching Alpaca 1Min bars (feed=%s) for %d tickers...", feed, len(yf_bars))
    al_bars = client.bars(
        list(yf_bars),
        lo.tz_convert("UTC"),
        (hi + pd.Timedelta(minutes=1)).tz_convert("UTC"),
        "1Min",
    )
    log.info("Alpaca returned data for %d/%d tickers in %d calls",
             len(al_bars), len(yf_bars), client.calls)

    rows = []
    for t, d in yf_bars.items():
        rec = compare_bars(d, al_bars.get(t, pd.DataFrame()))
        rec["ticker"] = t
        rows.append(rec)
    cov = pd.DataFrame(rows).set_index("ticker").sort_values("vol_coverage")

    # ---- the question that decides it: same trades, same number? ----
    events = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    results = {}
    for name, bars in (("yfinance", yf_bars), (f"alpaca_{feed}", al_bars)):
        if not bars:
            continue
        base = S.candle_baselines(bars)
        trades = S.backtest(events, bars, base)
        results[name] = (trades, S.summarise(trades))

    _report(cov, results, feed)

    out = DATA_DIR / f"alpaca_{feed}_coverage.csv"
    cov.to_csv(out)
    for name, (trades, _) in results.items():
        trades.to_csv(DATA_DIR / f"trades_calib_{name}.csv", index=False)
    log.info("wrote %s", out)
    return cov


def _report(cov: pd.DataFrame, results: dict, feed: str) -> None:
    print(f"\n{'='*66}\n  ALPACA '{feed}' vs YFINANCE CONSOLIDATED\n{'='*66}")
    print(f"\nTickers compared          {len(cov)}")
    print(f"  with zero Alpaca data   {(cov.al_bars == 0).sum()}")
    print("\nPer-ticker coverage (median / mean / 10th pct):")
    for col, label in (("bar_coverage", "bars present"), ("vol_coverage", "volume captured")):
        s = cov[col]
        print(f"  {label:<18} {s.median():6.1%} / {s.mean():6.1%} / {s.quantile(0.10):6.1%}")
    mae = cov.close_mae_bps.dropna()
    if len(mae):
        print(f"  close price MAE     {mae.median():6.1f} bps (median)")

    print("\nWorst 10 by volume coverage:")
    print(cov.head(10)[["yf_bars", "al_bars", "bar_coverage", "vol_coverage"]].to_string())

    print(f"\n{'-'*66}\n  STRATEGY OUTPUT ON EACH FEED\n{'-'*66}")
    for name, (trades, summary) in results.items():
        print(f"\n{name}:")
        print(f"  {summary}")

    if len(results) == 2:
        (a, _), (b, _) = results["yfinance"], results[f"alpaca_{feed}"]
        ka = set(zip(a.ticker, a.date)) if len(a) else set()
        kb = set(zip(b.ticker, b.date)) if len(b) else set()
        both = ka & kb
        print(f"\nTrade agreement:")
        print(f"  yfinance only   {len(ka - kb):3d}")
        print(f"  alpaca only     {len(kb - ka):3d}")
        print(f"  both feeds      {len(both):3d}")
        denom = len(ka | kb)
        print(f"  Jaccard overlap {len(both)/denom:.1%}" if denom else "  no trades")
        print(
            "\nVERDICT: if the trade sets barely overlap, the feed is not a noisier\n"
            "view of the same strategy - it is a different strategy. Re-validate\n"
            "from scratch on this feed before trusting the +2.05% headline."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    ap.add_argument("--tickers", type=int, default=None, help="limit ticker count (smoke test)")
    args = ap.parse_args()
    run(args.feed, args.tickers)


if __name__ == "__main__":
    main()
