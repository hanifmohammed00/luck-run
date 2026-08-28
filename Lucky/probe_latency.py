"""Measure whether yfinance is fresh enough to drive live entries.

The backtest reads completed sessions from disk, so bar timing was free. A live
runner does not have that luxury: it must decide, at 09:47:05, whether the bar
labelled 09:46 is real. Three things can go wrong, in increasing order of how
much they cost:

  1. STALENESS  - the newest bar is minutes behind. Entries fire late.
  2. PARTIAL BARS - Yahoo returns the still-forming minute as though it were
     finished. This is the dangerous one. The signal is `volume >= 5x baseline`,
     so an under-filled bar understates volume and the spike is missed outright.
  3. GAPS / FAILURES - thin names skip minutes and some polls just fail. Those
     must register as "no data", never as "no signal".

The deliverable is a number: how long after a minute closes does its volume
stop changing. That becomes `--bar-lag` in the runner, which then deliberately
trades one bar behind. A correct signal a minute late beats a wrong signal now.

Sampling uses real $1-3 event tickers, not megacaps - staleness on AAPL says
nothing about a stock that trades 20 minutes out of 390.

Run it at the open (08:25 Chicago / 09:25 ET) and let it cover the entry window:

    .venv/bin/python -m Lucky.probe_latency --minutes 120
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from .config import DATA_DIR
from .yf_utils import split_batch_frame, to_exchange_time

log = logging.getLogger(__name__)

ET = "America/New_York"


def watchlist(n: int = 12) -> list[str]:
    """Thin $1-3 names of the kind the strategy actually trades.

    Falls back to a hardcoded sample if the events file is unavailable, so the
    probe still runs on a machine with a fresh checkout.
    """
    path = DATA_DIR / "events_1_3.csv"
    if path.exists():
        e = pd.read_csv(path, parse_dates=["date"])
        q = e[e.open_price.between(1, 3) & e.gap_pct.between(4, 20, inclusive="left")]
        if len(q):
            # most recent events first: those tickers are still active
            return list(q.sort_values("date", ascending=False).ticker.unique()[:n])
    return ["AMC", "WRAP", "LICN", "GMEX", "LVWR", "OPK", "LUNG", "SLNH"][:n]


def poll(tickers: list[str]) -> tuple[dict[str, pd.DataFrame], float, bool]:
    """One batched 1-minute pull. Returns (frames, elapsed_seconds, ok)."""
    import yfinance as yf

    t0 = time.time()
    try:
        raw = yf.download(tickers, period="1d", interval="1m", group_by="ticker",
                          threads=True, progress=False, prepost=False, auto_adjust=False)
    except Exception as exc:
        log.warning("poll failed: %s", exc)
        return {}, time.time() - t0, False
    frames = split_batch_frame(raw, tickers)
    frames = {t: to_exchange_time(d) for t, d in frames.items() if not d.empty}
    return frames, time.time() - t0, bool(frames)


def run(minutes: int, interval: int, tickers: list[str]) -> pd.DataFrame:
    """Poll repeatedly, recording every observation of every bar.

    The key record is (ticker, bar_minute, observed_at, volume): re-seeing the
    same bar with a different volume is how a partial bar reveals itself.
    """
    deadline = time.time() + minutes * 60
    observations: list[dict] = []
    polls: list[dict] = []

    while time.time() < deadline:
        started = pd.Timestamp.now(tz=ET)
        frames, elapsed, ok = poll(tickers)
        polls.append({"at": started, "elapsed_s": elapsed, "ok": ok,
                      "tickers_returned": len(frames)})

        for ticker, df in frames.items():
            for ts, row in df.tail(6).iterrows():
                observations.append({
                    "ticker": ticker,
                    "bar": ts,
                    "observed_at": started,
                    # a bar labelled 09:46 is only complete at 09:47:00
                    "age_s": (started - (ts + pd.Timedelta(minutes=1))).total_seconds(),
                    "volume": float(row["Volume"]),
                    "close": float(row["Close"]),
                })

        newest = {t: d.index.max() for t, d in frames.items()}
        if newest:
            lag = [(started - (b + pd.Timedelta(minutes=1))).total_seconds()
                   for b in newest.values()]
            log.info("%s  %d/%d tickers  fetch %.1fs  newest-bar lag: med %.0fs max %.0fs",
                     started.strftime("%H:%M:%S"), len(frames), len(tickers),
                     elapsed, pd.Series(lag).median(), max(lag))
        else:
            log.warning("%s  no data returned", started.strftime("%H:%M:%S"))

        # pace on the interval, minus whatever the fetch already consumed
        time.sleep(max(1.0, interval - elapsed))

    obs = pd.DataFrame(observations)
    pol = pd.DataFrame(polls)

    # save BEFORE reporting: a report bug must not discard hours of collection,
    # and the report can always be re-run offline from these files
    stamp = dt.datetime.now().strftime("%Y%m%d")
    obs.to_csv(DATA_DIR / f"probe_observations_{stamp}.csv", index=False)
    pol.to_csv(DATA_DIR / f"probe_polls_{stamp}.csv", index=False)
    log.info("wrote data/probe_observations_%s.csv (%d rows)", stamp, len(obs))

    try:
        _report(obs, pol)
    except Exception:
        log.exception("report failed - data is saved, rerun with --report-only %s", stamp)
    return obs


def _report(obs: pd.DataFrame, pol: pd.DataFrame) -> None:
    print(f"\n{'='*68}\n  YFINANCE LIVE LATENCY PROBE\n{'='*68}")
    if obs.empty:
        print("\nNo observations collected - market closed, or every poll failed.")
        if not pol.empty:
            print(f"polls: {len(pol)}, ok: {int(pol.ok.sum())}")
        return

    print(f"\nPolls           {len(pol)}  ({int(pol.ok.sum())} ok, "
          f"{int((~pol.ok).sum())} failed = {(~pol.ok).mean():.1%})")
    print(f"Fetch time      median {pol.elapsed_s.median():.2f}s  p95 {pol.elapsed_s.quantile(.95):.2f}s")
    print(f"Bars observed   {len(obs)} observations of "
          f"{obs.groupby(['ticker','bar']).ngroups} distinct bars, {obs.ticker.nunique()} tickers")

    # --- 1. staleness of the newest bar at each poll ---
    newest = obs.groupby(["ticker", "observed_at"]).age_s.min()
    print(f"\n1. STALENESS (age of newest bar beyond its close)")
    print(f"   median {newest.median():5.0f}s   p95 {newest.quantile(.95):5.0f}s   "
          f"max {newest.max():5.0f}s")

    # --- 2. did a bar's volume change after we first saw it? ---
    g = obs.sort_values("observed_at").groupby(["ticker", "bar"])
    first_v, last_v = g.volume.first(), g.volume.last()
    seen_twice = g.size() > 1
    revised = (first_v != last_v) & seen_twice
    print(f"\n2. PARTIAL BARS (volume changed after first sighting)")
    if seen_twice.sum() == 0:
        print("   too few repeat observations - raise --minutes or lower --interval")
    else:
        share = revised[seen_twice].mean()
        print(f"   revised: {int(revised.sum())}/{int(seen_twice.sum())} bars ({share:.1%})")
        if revised.any():
            understated = (last_v[revised] - first_v[revised]) / last_v[revised].replace(0, pd.NA)
            print(f"   when revised, first read captured only "
                  f"{(1 - understated).median():.0%} of final volume (median)")
            # how old was the bar when it finally settled?
            settled = obs[obs.set_index(["ticker", "bar"]).index.isin(revised[revised].index)]
            final = settled.merge(last_v.rename("final_v"), on=["ticker", "bar"])
            at_final = final[final.volume == final.final_v].groupby(["ticker", "bar"]).age_s.min()
            if len(at_final):
                print(f"   settles by  median {at_final.median():.0f}s  "
                      f"p95 {at_final.quantile(.95):.0f}s after bar close")
                print(f"\n   >>> SUGGESTED runner --bar-lag: {int(at_final.quantile(.95)) + 5}s")

    # --- 3. coverage ---
    print(f"\n3. GAPS")
    per = obs.groupby("ticker").bar.nunique().sort_values()
    span = (obs.bar.max() - obs.bar.min()).total_seconds() / 60 + 1
    print(f"   window spans {span:.0f} minutes; bars per ticker: "
          f"median {per.median():.0f}, min {per.min()}, max {per.max()}")
    print(f"   thinnest names: {', '.join(f'{t}={n}' for t, n in per.head(4).items())}")

    print(f"\n{'-'*68}")
    print("READ IT LIKE THIS:")
    print("  staleness < 30s and revisions ~0%  -> runner can act on the newest bar")
    print("  revisions common but settle fast   -> use --bar-lag, trade one bar behind")
    print("  staleness minutes, or >30% failed  -> yfinance cannot drive live entries;")
    print("                                        the signal needs a real-time feed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=int, default=120, help="how long to probe")
    ap.add_argument("--interval", type=int, default=20, help="seconds between polls")
    ap.add_argument("--tickers", default=None, help="comma-separated override")
    ap.add_argument("--report-only", default=None, metavar="YYYYMMDD",
                    help="re-print the report from a saved probe, no polling")
    args = ap.parse_args()

    if args.report_only:
        obs = pd.read_csv(DATA_DIR / f"probe_observations_{args.report_only}.csv",
                          parse_dates=["bar", "observed_at"])
        pol = pd.read_csv(DATA_DIR / f"probe_polls_{args.report_only}.csv", parse_dates=["at"])
        _report(obs, pol)
        return

    tk = args.tickers.split(",") if args.tickers else watchlist()
    now = pd.Timestamp.now(tz=ET)
    log.info("probing %d tickers for %d min: %s", len(tk), args.minutes, ", ".join(tk))
    if not (dt.time(9, 30) <= now.time() <= dt.time(16, 0)) or now.weekday() >= 5:
        log.warning("market appears CLOSED (%s ET) - bars will not update and the "
                    "probe will report nothing useful", now.strftime("%a %H:%M"))
    run(args.minutes, args.interval, tk)


if __name__ == "__main__":
    main()
