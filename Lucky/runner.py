"""Live dry-run: the gap-VWAP strategy driven by yfinance, one minute at a time.

Does the probe's job too. Two pollers would double the request rate against an
unofficial endpoint, so this records the same bar observations probe_latency
does - on a better watchlist (today's actual gappers) - and the same report
comes out at the close.

SAFETY MODEL: polling and recording is the floor, strategy is best-effort on
top. Signal logic runs inside a try/except and a crash there cannot stop the
observation loop, so running this is never worse than running the probe.

Submits nothing. There is no Alpaca code here at all - a dry run needs none,
and leaving it out means nothing can accidentally reach a broker. Order
submission, position sizing and crash-resume land after the bar-lag number is known.

DIVERGENCE FROM backtest(), deliberate:
  - backtest() skips sessions with <30 bars as a data-quality filter. Live that
    would block every entry before 10:00, and real entries fire at 09:31. Not
    applied here; the end-of-day replay applies it, so the two can differ on a
    barely-trading name.

    .venv/bin/python -m Lucky.runner                  # today, dry run
    .venv/bin/python -m Lucky.runner --bar-lag 300    # tighter fallback cap, if you trust it
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import resource
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

import gap_vwap_strategy as S

from .config import DATA_DIR
from .probe_latency import _report, poll
from .yf_utils import download_batch, to_exchange_time

log = logging.getLogger(__name__)

ET = "America/New_York"
P = S.DEFAULT

# A bar's volume keeps revising after it first appears (94.6% of bars do,
# probe_latency measured it) - median settle time 39s, p95 544s. Waiting the
# full --bar-lag (the p95, worst case) on every bar is correct but slow.
# Trusting a bar once its volume stops moving across STABLE_POLLS consecutive
# polls gets most bars in ~2 * `interval` seconds (close to the 39s median)
# while --bar-lag remains the fallback cap for stragglers that never settle.
STABLE_POLLS = 2


def _settled_bars(df: pd.DataFrame, prev_bar: dict, stable_count: dict,
                  started: pd.Timestamp, max_lag_s: int) -> pd.DataFrame:
    """Bars usable now: Volume AND Open AND Close unchanged for STABLE_POLLS
    polls in a row, or older than max_lag_s regardless. Mutates prev_bar/
    stable_count in place (per-ticker dicts the caller keeps across polls).

    ADDED 2026-08-07: was volume-only. Confirmed via probe_observations on
    ZYBT that day - the entry fill (3.2999952316284) matches a mid-formation
    observation of the entry bar's close from 7s BEFORE that bar's official
    minute even ended, not the bar's eventual settled value (3.2819, fully
    stable by the next poll). Volume looked stable while price was still
    moving underneath it - the volume-only check missed exactly the bar it
    most needed to catch, on exactly the most violent minute of the day.
    Checking Open too (not just Close) matters because Open is what decides
    the fill: `fill = max(trigger, o[entry_i])`.
    """
    ok = []
    for ts in df.index:
        key = (float(df.at[ts, "Volume"]), float(df.at[ts, "Open"]), float(df.at[ts, "Close"]))
        if prev_bar.get(ts) == key:
            stable_count[ts] = stable_count.get(ts, 0) + 1
        else:
            stable_count[ts] = 0
        age_s = (started - (ts + pd.Timedelta(minutes=1))).total_seconds()
        if stable_count[ts] >= STABLE_POLLS or age_s >= max_lag_s:
            ok.append(ts)
    prev_bar.clear()
    prev_bar.update({ts: (float(r.Volume), float(r.Open), float(r.Close)) for ts, r in df.iterrows()})
    return df.loc[df.index.isin(ok)]


def _today() -> dt.date:
    """Session date in EXCHANGE time. The local clock is wrong here: the
    runner can now sleep across midnight, and a non-ET machine disagrees with
    the bar index for part of every evening."""
    return pd.Timestamp.now(tz=ET).date()


def candidates(max_names: int = 40) -> pd.DataFrame:
    """Today's gap-ups, from daily bars, at the open.

    A prior close outside ~$0.80-$3.00 cannot open in $1-3 on an 8-20% gap, so
    the universe is cut on yesterday's close before any request is made.
    """
    uni = pd.read_csv(DATA_DIR / "universe.csv")
    sym = "ticker" if "ticker" in uni else "symbol"
    pre = uni[pd.to_numeric(uni["last_close"], errors="coerce").between(0.75, 3.10)]
    tickers = sorted(pre[sym].unique())
    log.info("universe %d -> %d plausible on prior close", len(uni), len(tickers))

    rows = []
    for i in range(0, len(tickers), 200):
        chunk = tickers[i : i + 200]
        frames = download_batch(chunk, period="5d", interval="1d")
        for t, d in frames.items():
            d = d.dropna(subset=["Open", "Close"])
            if len(d) < 2:
                continue
            today, prior = d.iloc[-1], d.iloc[-2]
            if prior.Close <= 0:
                continue
            rows.append({"ticker": t, "open_price": float(today.Open),
                         "prior_close": float(prior.Close),
                         # the event scan that built the backtest used a 25k
                         # liquidity floor; without it the runner would trade
                         # names the strategy was never validated on
                         "avg_volume": float(d.Volume.iloc[:-1].mean()),
                         "gap_pct": (today.Open - prior.Close) / prior.Close * 100})
        time.sleep(1.0)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    q = df[df.open_price.between(P.price_min, P.price_max)
           & df.gap_pct.between(P.gap_min, P.gap_max, inclusive="left")
           & (df.avg_volume >= 25_000)]
    # biggest gaps first: that is the order the fill cap consumes
    return q.sort_values("gap_pct", ascending=False).head(max_names).reset_index(drop=True)


LIVE_STATE = DATA_DIR / "live_state.json"
INTENTS = DATA_DIR / "intents.json"


def read_intents() -> dict:
    """Manual close/ride decisions taken from the dashboard. Missing file is
    the normal case - it only exists once a button has been pressed."""
    try:
        return json.loads(INTENTS.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def write_live_state(signals: dict, cands: pd.DataFrame, all_signals: dict,
                     session_bars: dict, started: pd.Timestamp, polls: int,
                     exits: dict | None = None, phase: str = "live") -> None:
    """Snapshot the in-flight session for the dashboard, every poll.

    The runner otherwise writes nothing until the close, so a dashboard has
    no way to see a position while it is open. This is a one-way publish -
    nothing reads it back, so a failure here must never touch the session.

    A ticker in `exits` has already resolved (TP/SL/EODT fired mid-session,
    live-exit tracking added 2026-08-07) - it moves from
    `positions` (open, still marked-to-market) to `closed` (frozen at its
    exit) so the dashboard visibly shows the close instead of leaving a
    blown-through stop sitting in "Open positions" until the nightly replay.
    """
    exits = exits or {}
    intents = read_intents()
    open_rows, closed_rows = [], []
    for t, s in signals.items():
        fill = float(s["fill"])
        if t in exits:
            ex = exits[t]
            closed_rows.append({
                "ticker": t, "entry_time": s["entry_time"], "entry": fill,
                "signal_time": s["signal_time"], "signal_multiple": s.get("signal_multiple"),
                "gap_pct": s.get("gap_pct"), "trigger": s.get("trigger"),
                "outcome": ex["outcome"], "exit_time": ex["exit_time"],
                "exit_price": float(ex["exit_price"]), "return_pct": float(ex["return_pct"]),
                "minutes_held": ex.get("minutes_held"), "closed_at": ex.get("closed_at"),
            })
            continue
        df = session_bars.get(t)
        last = float(df.Close.iloc[-1]) if df is not None and not df.empty else None
        open_rows.append({
            "ticker": t, "entry_time": s["entry_time"], "entry": fill, "last": last,
            "signal_time": s["signal_time"], "signal_multiple": s.get("signal_multiple"),
            "gap_pct": s.get("gap_pct"), "trigger": s.get("trigger"),
            # gross of costs: this is a mark, not a realised trade
            "unrealised_pct": None if last is None else (last - fill) / fill * 100,
            "target": fill * (1 + P.take_profit / 100),
            "stop": fill * (1 - P.stop_loss / 100),
            "intent": intents.get(t, {}).get("action"),
        })

    cand_rows = []
    if cands is not None and not cands.empty:
        for r in cands.itertuples():
            sig = all_signals.get(r.ticker)
            cand_rows.append({"ticker": r.ticker, "gap_pct": float(r.gap_pct),
                              "signalled": sig is not None, "filled": r.ticker in signals,
                              "signal_multiple": sig.get("signal_multiple") if sig else None})

    payload = {"session": str(_today()), "updated": started.strftime("%H:%M:%S"),
               "polls": polls, "positions": open_rows, "closed": closed_rows,
               "candidates": cand_rows, "phase": phase,
               "cutoff": P.no_entry_after, "max_fills": P.max_trades_per_day}
    try:
        LIVE_STATE.write_text(json.dumps(payload, default=str))
    except OSError:
        log.exception("live_state write failed - session continues")

    _append_equity_snapshot(started, open_rows, closed_rows)


def _append_equity_snapshot(started: pd.Timestamp, open_rows: list[dict],
                            closed_rows: list[dict]) -> None:
    """ADDED 2026-08-07: one row per poll of today's running P&L, in percent
    (not dollars - runner.py doesn't know POSITION_SIZE, that's a dashboard
    display assumption; dashboard.py multiplies at render time, same as it
    already does for every other $ figure). Nothing previously recorded
    equity at anything finer than one point per DAY (dashboard.py's
    equity_curve()) - there was no intraday curve to show, only the
    end-of-day number. Append-only, like every other *_log.csv here; a
    write failure must not touch the polling loop."""
    path = DATA_DIR / f"equity_intraday_{_today().strftime('%Y%m%d')}.csv"
    row = pd.DataFrame([{
        "time": started.strftime("%H:%M:%S"),
        "realised_return_pct": sum(r["return_pct"] for r in closed_rows),
        "unrealised_return_pct": sum(r["unrealised_pct"] for r in open_rows
                                     if r.get("unrealised_pct") is not None),
        "open_count": len(open_rows), "closed_count": len(closed_rows),
    }])
    try:
        row.to_csv(path, mode="a", header=not path.exists(), index=False)
    except OSError:
        log.exception("equity snapshot write failed - session continues")


TRADE_LOG_COLS = ["ticker", "date", "signal_time", "entry_time", "fill",
                  "outcome", "exit_time", "exit_price", "return_pct", "minutes_held"]


def _append_trade_log(rows_df: pd.DataFrame, today: dt.date) -> None:
    """Append resolved trades to the running trade_log.csv (source=live is
    implicit - dashboard.load_trades() labels every row from this file that
    way). Shared by the live exit check (mid-session) and `_finish()` (the
    end-of-day fallback for anything that never resolved live), so both
    write the exact same shape and neither can drift from the other."""
    if rows_df.empty:
        return
    log_path = DATA_DIR / "trade_log.csv"
    entry = rows_df.assign(date=today)[TRADE_LOG_COLS].rename(columns={"fill": "entry_price"})
    entry.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def verify_gaps(cands: pd.DataFrame, bars: dict, today: dt.date) -> pd.DataFrame:
    """Re-derive the gap from the first REAL 1-minute bar; drop what fails.

    candidates() reads today's Open off the still-forming daily bar at 09:31,
    and Yahoo revises that field afterwards. Two live examples: DXST
    (2026-08-05) scanned as +10.3% on a daily Open of 2.99 that settled at
    2.46 against a 2.71 prior close, and EVTL (2026-08-06) scanned as +8.5%
    on 1.41 that settled at 1.28 against a 1.30 prior close. Both were gaps
    DOWN. Both traded. The first 1m bar is what actually printed, so the gap
    is measured off that instead.

    A name with no tape yet cannot be verified, so it is dropped - a gap that
    can't be confirmed is not one worth trading.

    Only the SIGN is vetoed here, not the size: the band filter keeps running
    on the scanner's gap_pct. Recomputing the whole gap off the 1m tape was
    tried and rejected - the first 1m bar is not always the official opening
    print (they agree exactly at the median, but the tail is wide), and on
    events_1_3.csv that version cut two genuine gap-ups (GIPR +8.9%, LUNG
    +12.0%, both +9% winners) for landing just under gap_min once re-derived.
    The sign is robust to that; the size is not.

    ponytail: this leaves the "scanner said +10%, reality was +1%" case
    unhandled - same bad field, no sign flip to catch it. Unmeasured, and the
    only fix tried cost more than it saved. Revisit with more saved sessions.
    """
    if cands.empty:
        return cands
    real = {}
    for t in cands.ticker:
        df = bars.get(t)
        if df is None or df.empty:
            continue
        session = df[[ts.date() == today for ts in df.index]]
        if not session.empty:
            real[t] = float(session.Open.iloc[0])

    out = cands[cands.ticker.isin(real)].copy()
    if out.empty:
        return out
    out["real_open"] = out.ticker.map(real)
    out = out[out.real_open >= out.prior_close]      # opened at or below yesterday: never trade it
    return out.drop(columns=["real_open"]).reset_index(drop=True)


def baselines_for(tickers: list[str]) -> dict:
    """Mean 1-minute candle volume over the prior sessions, per ticker.

    Reuses candle_baselines() rather than reimplementing: the 18x error that
    a daily-volume/390 proxy caused is exactly what that function exists to
    avoid, and a second implementation would be free to drift back into it.
    """
    end = _today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=20)
    bars: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 40):
        chunk = tickers[i : i + 40]
        # yfinance serves at most 8 days of 1m per request, so walk the window
        # in 7-day steps - same reason fetch_1m.py does
        cur, parts = start, {}
        while cur < end:
            stop = min(cur + dt.timedelta(days=7), end)
            for t, d in download_batch(chunk, start=cur, end=stop, interval="1m").items():
                if not d.empty:
                    parts.setdefault(t, []).append(d)
            cur = stop
            time.sleep(1.0)
        for t, frames in parts.items():
            d = pd.concat(frames)
            d = d[~d.index.duplicated()]
            bars[t] = to_exchange_time(d).between_time("09:30", "16:00", inclusive="left")
    base = S.candle_baselines(bars, P)
    today = _today()
    # keep only the entry for today; prior days are what built it
    return {t: v for (t, d), v in base.items() if d == today}, bars


def serve_dashboard(port: int = 8787, browser: str | None = "safari") -> None:
    """Dashboard in a daemon thread, then open it in the browser.

    Nothing here may be fatal: a busy port or a missing browser must never
    cost a trading session. Daemon thread so it dies with the runner rather
    than holding the process open after the close.
    """
    import threading
    import webbrowser
    from http.server import ThreadingHTTPServer

    from .dashboard import Handler

    url = f"http://localhost:{port}"
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        # already serving (a dashboard left running from earlier is fine)
        log.info("dashboard port %d already in use - reusing %s", port, url)
    else:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log.info("dashboard on %s", url)
    try:
        (webbrowser.get(browser) if browser else webbrowser).open(url)
    except Exception:
        try:
            webbrowser.open(url)               # whatever the default browser is
        except Exception:
            log.info("could not open a browser - visit %s yourself", url)


def _next_open(now: pd.Timestamp, until: dt.time) -> pd.Timestamp:
    """When to start: `now` if a session is already under way, else the next 09:31 ET.

    Before the open, today's daily bar does not exist yet and iloc[-1] is
    YESTERDAY - the scan would silently build a watchlist from yesterday's
    gappers. Same trap after the close, which is why this walks to the NEXT
    open rather than only waiting out the current morning.

    ponytail: weekends only, no holiday calendar. A real one needs a new dep or
    Alpaca creds, and this module deliberately has neither. On a holiday it
    wakes, finds no gap-ups, and degrades to the latency probe - what it
    already does on a quiet day. Swap in alpaca.calendar() if that idling
    ever costs something.
    """
    o = now.normalize() + pd.Timedelta(hours=9, minutes=31)
    end = now.normalize() + pd.Timedelta(hours=until.hour, minutes=until.minute)
    if now.weekday() < 5 and o <= now < end:
        return now                              # already mid-session, go
    while o <= now or o.weekday() >= 5:
        o = (o + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=9, minutes=31)
    return o


def _idle_done(signals: dict, now_hhmm: str, exits: dict | None = None) -> bool:
    """True when the session can be abandoned: past the entry cutoff, and
    nothing that could still be open is - either nothing signalled all
    morning, or everything that did has since resolved via a live exit
    (live exit tracking, done 2026-08-07). No new entries are possible past
    the cutoff either way, so
    a fully-resolved day is safe to stop watching."""
    if not P.no_entry_after or now_hhmm < P.no_entry_after:
        return False
    exits = exits or {}
    return not signals or all(t in exits for t in signals)


# How often to print the countdown, by how much of the wait is left. Started
# the night before, a flat 5-minute cadence buries the terminal in ~100
# identical lines nobody reads, and still tells you nothing more than "yes,
# still waiting". Tiered, the log is quiet while the answer is obvious and
# tightens as the open gets close enough to care about: hourly overnight,
# half-hourly inside 2h, every 10 minutes inside 30, then minute-by-minute for
# the last 10 - the only stretch where a missing line means something.
# (seconds-remaining floor, seconds between log lines), first match wins.
WAIT_LOG_LADDER = ((2 * 3600, 3600), (30 * 60, 1800), (10 * 60, 600), (0, 60))


def _wait_log_every(left: float) -> int:
    """Seconds between countdown lines with `left` seconds still to wait."""
    return next(every for floor, every in WAIT_LOG_LADDER if left > floor)


def _wait_until(target: pd.Timestamp) -> None:
    """Sleep to `target`, heart-beating the dashboard while we do.

    Started at 08:00, the runner is idle for 90 minutes. Without the
    heartbeat the dashboard's staleness check reads that as "not running",
    which is exactly the wrong thing to tell someone who just launched it.

    So the WAKE interval stays at 60s regardless of how far off `target` is -
    it is coupled to dashboard.STALE_AFTER_S (150s), not to anything about
    the wait, and stretching it to match the log ladder would make the
    dashboard call an overnight runner dead. Writing one small JSON file a
    minute is not the cost worth optimising here; only the LOG cadence tiers.
    """
    last_log = 0.0
    while (left := (target - pd.Timestamp.now(tz=ET)).total_seconds()) > 0:
        if time.time() - last_log >= _wait_log_every(left):
            log.info("sleeping until the open at %s (%s to go)",
                     target.strftime("%a %Y-%m-%d %H:%M %Z"), pd.Timedelta(seconds=int(left)))
            last_log = time.time()
        write_live_state({}, None, {}, {}, pd.Timestamp.now(tz=ET), 0,
                         phase=f"waiting for the open at {target.strftime('%H:%M')}")
        time.sleep(min(60, left))


def _today_already_traded() -> list[str]:
    """Tickers already logged for today's date in trade_log.csv - a strong
    signal this session already ran and finished, or another process beat
    this one to it.

    ADDED 2026-08-07 after two separate same-day incidents: an unexplained
    restart of `-m Lucky.runner` (origin never confirmed either time)
    fetched the WHOLE day's already-elapsed bars in
    one request - `run()` has no notion of "resume from where I left off",
    it only knows how to scan a session from scratch - and `find_entry()`
    retroactively "discovered" hours-old signals, immediately resolved them
    against bars that had already fully played out, and logged all of it as
    fresh live fills under whatever Params happened to be shipped at
    restart time. The first incident wrote 8 fabricated rows before being
    caught; the second was caught before it wrote anything, only because
    someone happened to be watching. This check exists so the second time
    doesn't depend on that luck."""
    path = DATA_DIR / "trade_log.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, usecols=["ticker", "date"])
    except (pd.errors.EmptyDataError, ValueError):
        return []
    today = str(_today())
    return sorted(df.loc[df.date.astype(str) == today, "ticker"].unique())


def run(bar_lag: int, interval: int, until: dt.time, max_names: int, force: bool = False) -> None:
    _wait_until(_next_open(pd.Timestamp.now(tz=ET), until))

    already = _today_already_traded()
    if already and not force:
        log.error(
            "REFUSING TO START: trade_log.csv already has real fills logged "
            "for today (%s). A same-day restart re-scans the WHOLE day's "
            "bars from scratch and can fabricate duplicate/phantom trades on "
            "top of what's already there (this has happened twice). If this "
            "is a deliberate resume after a genuine "
            "crash with zero fills so far today, pass --force. Exiting.",
            ", ".join(already))
        return
    if already:
        log.warning("--force set: proceeding despite %d ticker(s) already logged today (%s)",
                    len(already), ", ".join(already))

    cands = candidates(max_names)
    if cands.empty:
        log.warning("no qualifying gap-ups today - running as a pure latency probe")
        watch = []
    else:
        log.info("%d candidates: %s", len(cands),
                 ", ".join(f"{r.ticker}({r.gap_pct:.0f}%)" for r in cands.itertuples()))
        watch = list(cands.ticker)

    if not watch:
        from .probe_latency import watchlist
        watch = watchlist()

    base, prior_bars = baselines_for(watch)
    log.info("baselines computed for %d/%d candidates", len(base), len(watch))

    # The 09:31 daily Open is a snapshot that revises; re-derive every gap
    # from the first real 1m bar and make anything that fails untradeable.
    # Pruning `base` (not `watch`) is what disables it: the strategy step
    # gates on `ticker not in base`, so the latency probe keeps polling and
    # the candidate log keeps recording what the scanner originally saw.
    if not cands.empty:
        keep = set(verify_gaps(cands, prior_bars, _today()).ticker)
        dropped = sorted(t for t in base if t not in keep)
        base = {t: v for t, v in base.items() if t in keep}
        if dropped:
            log.info("real-open check: %d candidate(s) not tradeable today (%s)",
                     len(dropped), ", ".join(dropped))

    gaps = dict(zip(cands.ticker, cands.gap_pct)) if not cands.empty else {}
    opens = dict(zip(cands.ticker, cands.open_price)) if not cands.empty else {}

    observations, polls, signals, all_signals, exits = [], [], {}, {}, {}
    session_bars: dict[str, pd.DataFrame] = {}
    prev_bar_state: dict[str, dict] = {t: {} for t in watch}
    stable_count: dict[str, dict] = {t: {} for t in watch}
    filled = 0
    today = _today()
    stamp = _today().strftime("%Y%m%d")
    last_checkpoint = time.time()

    while pd.Timestamp.now(tz=ET).time() < until:
        started = pd.Timestamp.now(tz=ET)
        frames, elapsed, ok = poll(watch)
        polls.append({"at": started, "elapsed_s": elapsed, "ok": ok,
                      "tickers_returned": len(frames)})

        # ---- floor: record observations (this is the probe) ----
        for ticker, df in frames.items():
            df = df[[t.date() == today for t in df.index]]
            if df.empty:
                continue
            session_bars[ticker] = df
            for ts, row in df.tail(6).iterrows():
                observations.append({
                    "ticker": ticker, "bar": ts, "observed_at": started,
                    "age_s": (started - (ts + pd.Timedelta(minutes=1))).total_seconds(),
                    "volume": float(row["Volume"]), "open": float(row["Open"]),
                    "close": float(row["Close"]),
                })

        # ---- best-effort: strategy ----
        # Every candidate is checked every poll regardless of the fill cap --
        # that's what all_signals is for (the candidate log needs to know what
        # WOULD have signalled, not just what got a slot). Only the first
        # max_trades_per_day of them, in the order they actually fire, become
        # real fills (signals) and count toward `filled`.
        try:
            for ticker in watch:
                if ticker in all_signals or ticker not in base:
                    continue
                df = session_bars.get(ticker)
                if df is None or len(df) < 2:
                    continue
                # trust a bar once its volume stops changing across
                # STABLE_POLLS polls, or once it's older than bar_lag anyway
                sess = _settled_bars(df, prev_bar_state.setdefault(ticker, {}),
                                     stable_count.setdefault(ticker, {}), started, bar_lag)
                if len(sess) < 2:
                    continue
                sig = S.find_entry(sess, base[ticker], P)
                if sig:
                    all_signals[ticker] = {**sig, "detected_at": started.strftime("%H:%M:%S"),
                                           "gap_pct": gaps.get(ticker),
                                           "open_price": opens.get(ticker)}
                    if filled < P.max_trades_per_day:
                        signals[ticker] = all_signals[ticker]
                        filled += 1
                        log.info("SIGNAL %-6s spike %s x%.1f -> entry %s @ %.4f  "
                                 "[DRY RUN, no order sent]  (%d/%d fills)",
                                 ticker, sig["signal_time"], sig["signal_multiple"],
                                 sig["entry_time"], sig["fill"], filled, P.max_trades_per_day)
                        if filled >= P.max_trades_per_day:
                            log.info("daily fill cap reached - still logging candidates, no more fills")
                    else:
                        log.info("candidate %-6s would have signalled (over cap, logged only)", ticker)
        except Exception:
            log.exception("strategy step failed - polling continues")

        # ---- best-effort: live exit management ----
        # A filled position isn't watched anywhere else - `_finish()` only
        # simulates TP/SL/EODT retroactively, once, at the close. Checking
        # here means a blown-through stop shows as closed
        # immediately instead of sitting in "Open positions" all afternoon.
        # simulate_exit() on the bars-so-far returns "EOD" whenever nothing
        # has fired *yet* - that's indistinguishable from "still open" until
        # the session actually ends, so only a non-EOD outcome is a real,
        # no-look-ahead exit: it means some bar we've already seen crossed
        # the target or the stop. Once resolved it stays resolved - the
        # settled bars a later poll sees only ever gain trailing rows, so
        # the same first-crossing bar simulate_exit finds now is exactly
        # what `_finish()` would find later; recording it now just means
        # not waiting until then to know it.
        try:
            intents = read_intents()
            for t, s in signals.items():
                if t in exits:
                    continue
                df = session_bars.get(t)
                if df is None:
                    continue
                sess = _settled_bars(df, prev_bar_state.setdefault(t, {}),
                                     stable_count.setdefault(t, {}), started, bar_lag)
                if len(sess) <= s["entry_i"]:
                    continue
                ex = S.simulate_exit(sess, s["entry_i"], s["fill"], P)
                if ex["outcome"] == "EOD":
                    # a "close" pressed in the dashboard ends the trade now, at
                    # the latest settled price - same MANUAL truncation
                    # _finish() applies retroactively at day's end, just acted
                    # on live instead of waiting. TP/SL above still wins if it
                    # already fired before the button was pressed.
                    if intents.get(t, {}).get("action") == "close":
                        ex = {**ex, "outcome": "MANUAL"}
                    else:
                        continue                # nothing fired yet as of the bars we have
                ex["return_pct"] -= P.cost_pct
                exits[t] = {**ex, "closed_at": started.strftime("%H:%M:%S")}
                _append_trade_log(pd.DataFrame([{**s, **ex, "ticker": t}]), today)
                log.info("EXIT %-6s %s @ %.4f  %+.2f%%  held %dm  "
                         "[DRY RUN, no order sent]",
                         t, ex["outcome"], ex["exit_price"], ex["return_pct"],
                         int(ex["minutes_held"]))
        except Exception:
            log.exception("exit check failed - polling continues")

        write_live_state(signals, cands, all_signals, session_bars, started, len(polls), exits)

        newest = [d.index.max() for d in frames.values() if not d.empty]
        if newest:
            lag = [(started - (b + pd.Timedelta(minutes=1))).total_seconds() for b in newest]
            log.info("%s  %d/%d tickers  fetch %.1fs  lag med %.0fs  signals %d  open %d",
                     started.strftime("%H:%M:%S"), len(frames), len(watch), elapsed,
                     pd.Series(lag).median(), len(signals), len(signals) - len(exits))

        # nothing left to do: past the entry cutoff, and nothing that could
        # still be open is (either nothing ever signalled, or every fill has
        # since resolved via the live exit check above - done 2026-08-07).
        if _idle_done(signals, started.strftime("%H:%M"), exits):
            log.info("no open positions left past %s ET - stopping early",
                     P.no_entry_after)
            break

        # checkpoint every 5 min: a crash (e.g. fd exhaustion late in the
        # session) must not cost the whole day - this is what August 4 lost
        if time.time() - last_checkpoint > 300:
            try:
                pd.DataFrame(observations).to_csv(
                    DATA_DIR / f"probe_observations_{stamp}.csv", index=False)
                pd.DataFrame(polls).to_csv(
                    DATA_DIR / f"probe_polls_{stamp}.csv", index=False)
                last_checkpoint = time.time()
            except Exception:
                log.exception("checkpoint save failed - continuing")

        time.sleep(max(1.0, interval - elapsed))

    _finish(observations, polls, signals, session_bars, base, gaps, opens, cands, all_signals, exits)


def _finish(observations, polls, signals, session_bars, base, gaps, opens,
            cands=None, all_signals=None, exits=None) -> None:
    stamp = _today().strftime("%Y%m%d")
    obs, pol = pd.DataFrame(observations), pd.DataFrame(polls)

    # save first, always: a reporting bug must not cost a session of collection
    obs.to_csv(DATA_DIR / f"probe_observations_{stamp}.csv", index=False)
    pol.to_csv(DATA_DIR / f"probe_polls_{stamp}.csv", index=False)
    pd.to_pickle(session_bars, DATA_DIR / f"runner_bars_{stamp}.pkl")
    log.info("saved observations (%d rows) and %d ticker sessions", len(obs), len(session_bars))

    # candidate-level log: every ticker the runner watched, whether it ever
    # signalled, and whether it actually got a fill slot. `all_signals` (not
    # `signals`) drives `signalled` -- it's checked every poll with no cap, so
    # a candidate that would have triggered after the fill cap was hit still
    # shows up here instead of silently reading as signalled=False.
    if cands is not None and not cands.empty:
        all_signals = all_signals or {}
        rows = []
        for r in cands.itertuples():
            sig = all_signals.get(r.ticker)
            rows.append({
                "date": _today(), "ticker": r.ticker, "gap_pct": r.gap_pct,
                "open_price": r.open_price, "avg_volume": r.avg_volume,
                "gap_rank": r.Index + 1, "signalled": sig is not None,
                "signal_time": sig["signal_time"] if sig else None,
                "signal_multiple": sig["signal_multiple"] if sig else None,
                "entry_time": sig["entry_time"] if sig else None,
                "filled": r.ticker in signals,
            })
        pd.DataFrame(rows).to_csv(DATA_DIR / f"candidates_{stamp}.csv", index=False)
        log.info("saved %d candidates (%d signalled)", len(rows), sum(r["signalled"] for r in rows))

    live = []
    intents = read_intents()
    exits = exits or {}
    for t, s in signals.items():
        # already resolved and logged live by the in-session exit check -
        # reuse it rather than recomputing, so this can never disagree with
        # what was already written to trade_log.csv.
        if t in exits:
            ex = {k: v for k, v in exits[t].items() if k != "closed_at"}
            live.append({**s, **ex, "ticker": t})
            continue
        df = session_bars.get(t)
        if df is None:
            continue
        # a "close" pressed in the dashboard ends the trade at that minute:
        # truncate the session there and let simulate_exit run out of bars.
        # TP/SL still win if they fired BEFORE the button was pressed.
        closed_at = intents.get(t, {}).get("action") == "close" and intents[t].get("at")
        sess = df[[ts.strftime("%H:%M") <= closed_at for ts in df.index]] if closed_at else df
        if closed_at and len(sess) <= s["entry_i"] + 1:
            sess = df                      # pressed before the fill printed: ignore it
            closed_at = None
        ex = S.simulate_exit(sess, s["entry_i"], s["fill"], P)
        if closed_at and ex["outcome"] == "EOD":
            ex["outcome"] = "MANUAL"
        ex["return_pct"] -= P.cost_pct
        live.append({**s, **ex, "ticker": t})
    live_df = pd.DataFrame(live)
    live_df.to_csv(DATA_DIR / f"runner_live_{stamp}.csv", index=False)
    # only what DIDN'T already resolve live gets appended here - anything in
    # `exits` was written to trade_log.csv the moment it closed, mid-session.
    fresh = live_df[~live_df.ticker.isin(exits)] if len(live_df) else live_df
    _append_trade_log(fresh, _today())

    # replay: the backtest over the same bars. Divergence = a live/offline bug,
    # which is the whole reason to dry-run before wiring a broker in.
    ev = pd.DataFrame([{"ticker": t, "date": pd.Timestamp(_today()),
                        "gap_pct": gaps.get(t), "open_price": opens.get(t)}
                       for t in session_bars if gaps.get(t) is not None])
    replay = pd.DataFrame()
    if not ev.empty:
        bl = {(t, _today()): v for t, v in base.items()}
        replay = S.backtest(ev, session_bars, bl, P)
        replay.to_csv(DATA_DIR / f"runner_replay_{stamp}.csv", index=False)

    try:
        _report(obs, pol)
    except Exception:
        log.exception("latency report failed - data is saved")

    print(f"\n{'='*68}\n  DRY RUN vs BACKTEST REPLAY (same bars)\n{'='*68}")
    lt = set(live_df.ticker) if len(live_df) else set()
    rt = set(replay.ticker) if len(replay) else set()
    print(f"  live signals    {len(lt)}  {sorted(lt)}")
    print(f"  replay trades   {len(rt)}  {sorted(rt)}")
    print(f"  agree           {len(lt & rt)}   live-only {sorted(lt - rt)}  "
          f"replay-only {sorted(rt - lt)}")
    if len(live_df):
        print(f"\n  live P&L  {live_df.return_pct.mean():+.2f}%/trade "
              f"over {len(live_df)} trades (paper, nothing submitted)")
    if len(replay):
        print(f"  replay    {replay.return_pct.mean():+.2f}%/trade over {len(replay)} trades")
    print("\n  live-only or replay-only names are the thing to investigate:\n"
          "  same rules on the same bars should give the same trades.")


def _check_next_open() -> None:
    """Date branching that only misfires at 4am on a Sunday is worth an assert.
    Cheap enough to run on every start, so it always actually runs."""
    def at(s):
        return pd.Timestamp(s, tz=ET)
    u = dt.time(15, 55)
    # 2026-08-05 is a Wednesday, 08-08 a Saturday, 08-09 a Sunday
    assert _next_open(at("2026-08-05 07:00"), u) == at("2026-08-05 09:31"), "early: same morning"
    assert _next_open(at("2026-08-05 10:00"), u) == at("2026-08-05 10:00"), "mid-session: start now"
    assert _next_open(at("2026-08-05 20:00"), u) == at("2026-08-06 09:31"), "evening: next day"
    assert _next_open(at("2026-08-07 20:00"), u) == at("2026-08-10 09:31"), "Fri night: skip weekend"
    assert _next_open(at("2026-08-08 12:00"), u) == at("2026-08-10 09:31"), "Saturday: skip to Monday"
    assert _next_open(at("2026-08-09 08:00"), u) == at("2026-08-10 09:31"), "Sunday am: not today"
    assert _next_open(at("2026-08-05 16:30"), u) == at("2026-08-06 09:31"), "after `until`: next day"

    # derived from P.no_entry_after, not hardcoded - this drifted silently
    # once already when the cutoff moved 10:30->12:00 (2026-08-06)
    hh, mm = (int(x) for x in P.no_entry_after.split(":"))
    before = f"{hh:02d}:{mm-1:02d}" if mm else f"{hh-1:02d}:59"
    assert not _idle_done({}, before), "before the cutoff: keep going"
    assert _idle_done({}, P.no_entry_after), "at the cutoff with no signal: stop"
    assert _idle_done({}, "23:59"), "well past the cutoff, still nothing: stop"
    assert not _idle_done({"LDI": {}}, "23:59"), "a position still open: never stop early"
    assert _idle_done({"LDI": {}}, "23:59", {"LDI": {}}), \
        "a position fired AND has since closed live: stop early (added 2026-08-07)"
    assert not _idle_done({"LDI": {}, "ABCD": {}}, "23:59", {"LDI": {}}), \
        "one of two still open: keep going"


def _check_verify_gaps() -> None:
    """The real DXST/EVTL failure: a daily Open that revises DOWN through the
    prior close, after the scanner already called it a gap up."""
    day = dt.date(2026, 8, 5)

    def bars(first_open):
        idx = pd.date_range(f"{day} 09:30", periods=3, freq="1min", tz=ET)
        return pd.DataFrame({"Open": [first_open, first_open, first_open],
                             "High": [first_open] * 3, "Low": [first_open] * 3,
                             "Close": [first_open] * 3, "Volume": [1000] * 3}, index=idx)

    cands = pd.DataFrame([
        {"ticker": "DXST", "open_price": 2.99, "prior_close": 2.71,
         "avg_volume": 1e6, "gap_pct": 10.33},          # scanner saw +10.3%
        {"ticker": "GOOD", "open_price": 2.40, "prior_close": 2.00,
         "avg_volume": 1e6, "gap_pct": 20.0},           # a genuine gap up
    ])
    # DXST really opened 2.46, i.e. BELOW its 2.71 prior close; GOOD is honest
    got = verify_gaps(cands, {"DXST": bars(2.46), "GOOD": bars(2.40)}, day)
    assert list(got.ticker) == ["GOOD"], f"gap-down must be dropped, got {list(got.ticker)}"

    # size is NOT re-derived: a real open under the scanner's, but still above
    # the prior close, must survive (this is the GIPR/LUNG case)
    survives = verify_gaps(cands, {"DXST": bars(2.75), "GOOD": bars(2.05)}, day)
    assert set(survives.ticker) == {"DXST", "GOOD"}, f"sign-only veto, got {list(survives.ticker)}"

    # no tape yet -> unverifiable -> not traded
    assert verify_gaps(cands, {}, day).empty, "no bars means no verification means no trade"

    # _settled_bars: volume unchanged for STABLE_POLLS polls -> usable early;
    # still-moving volume -> held back until the bar_lag fallback cap
    idx = pd.date_range("2026-08-05 09:30", periods=3, freq="1min", tz=ET)
    df = pd.DataFrame({"Open": [1.0, 1.0, 1.0], "Close": [1.0, 1.0, 1.0],
                       "Volume": [100.0, 200.0, 300.0]}, index=idx)
    now = idx[-1] + pd.Timedelta(minutes=1, seconds=5)   # bar 0 is 95s old, bar 2 is 5s old
    pv, sc = {}, {}
    out1 = _settled_bars(df, pv, sc, now, max_lag_s=548)
    assert len(out1) == 0, "first sighting of every bar: nothing stable yet, none old enough"
    out2 = _settled_bars(df, pv, sc, now, max_lag_s=548)          # volume unchanged this poll
    assert len(out2) == 0, "one repeat isn't STABLE_POLLS=2 yet"
    out3 = _settled_bars(df, pv, sc, now, max_lag_s=548)          # unchanged again -> stable
    assert list(out3.index) == list(idx), "two unchanged polls in a row: all bars trusted"
    df.loc[idx[2], "Volume"] = 350.0                              # bar 2 revises upward
    out4 = _settled_bars(df, pv, sc, now, max_lag_s=548)
    assert list(out4.index) == list(idx[:2]), "a revision resets that bar's stability count"
    old_now = idx[2] + pd.Timedelta(seconds=610)                  # bar 2 now older than max_lag_s
    out5 = _settled_bars(df, pv, sc, old_now, max_lag_s=548)
    assert idx[2] in out5.index, "past the fallback cap, used regardless of stability"

    # ADDED 2026-08-07: volume can look stable while price is still moving
    # underneath it - the real ZYBT failure (fill matched a mid-formation
    # observation, not the bar's eventual settled value). Volume-only
    # settlement must NOT be fooled by this.
    df2 = pd.DataFrame({"Open": [2.00, 2.00, 2.00], "Close": [2.00, 2.00, 2.00],
                        "Volume": [500.0, 500.0, 500.0]}, index=idx)
    pv2, sc2 = {}, {}
    _settled_bars(df2, pv2, sc2, now, max_lag_s=548)              # sighting 1
    _settled_bars(df2, pv2, sc2, now, max_lag_s=548)              # repeat 1 (stable_count -> 1)
    o1 = _settled_bars(df2, pv2, sc2, now, max_lag_s=548)         # repeat 2 (stable_count -> 2)
    assert list(o1.index) == list(idx), "volume+price both stable for 2 repeats: trusted"
    df2.loc[idx[1], "Open"] = 2.15                                 # bar 1's OPEN revises, volume doesn't
    o2 = _settled_bars(df2, pv2, sc2, now, max_lag_s=548)
    assert idx[1] not in o2.index, \
        "volume unchanged but Open revised: must NOT be trusted (this is the ZYBT bug)"
    assert idx[0] in o2.index and idx[2] in o2.index, "bars that didn't revise stay trusted"


def _raise_fd_limit() -> None:
    """macOS defaults background processes to a 256 fd soft limit. A session
    this long makes thousands of yfinance/requests calls; on Aug 4 that
    exhausted the default and crashed the process at the finish line, on the
    very save call meant to protect against crashes. Raise it up front."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 8192) if hard != resource.RLIM_INFINITY else 8192
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            log.info("raised fd soft limit %d -> %d", soft, target)
    except Exception:
        log.warning("could not raise fd limit; run `ulimit -n 4096` before "
                    "starting if you see 'Too many open files' again")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    _check_next_open()
    _check_verify_gaps()
    _raise_fd_limit()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bar-lag", type=int, default=548,
                    help="fallback cap (seconds): a bar this old is used even if its volume "
                         "hasn't visibly stabilized yet. Measured p95 settle time; the "
                         "STABLE_POLLS convergence check (see _settled_bars) is what actually "
                         "clears most bars faster than this, in the typical case")
    ap.add_argument("--interval", type=int, default=20, help="seconds between polls")
    ap.add_argument("--until", default="15:55", help="stop time, ET")
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--force", action="store_true",
                    help="proceed even if trade_log.csv already has fills logged for "
                         "today (normally refused - see _today_already_traded())")
    ap.add_argument("--skip-eod-short", action="store_true",
                    help="skip the post-close short-the-day's-biggest-gainer step")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="do not serve or open the dashboard")
    ap.add_argument("--dashboard-port", type=int, default=8787)
    ap.add_argument("--browser", default="safari",
                    help="browser to open the dashboard in; '-' for the system default")
    args = ap.parse_args()

    if not args.no_dashboard:
        serve_dashboard(args.dashboard_port, None if args.browser == "-" else args.browser)

    hh, mm = (int(x) for x in args.until.split(":"))
    now = pd.Timestamp.now(tz=ET)
    start = _next_open(now, dt.time(hh, mm))
    log.info("now %s ET; %s", now.strftime("%a %Y-%m-%d %H:%M"),
             "session live, starting immediately" if start == now else
             f"sleeping until {start.strftime('%a %Y-%m-%d %H:%M %Z')} "
             f"({pd.Timedelta(seconds=int((start - now).total_seconds()))} away)")
    log.info("dry run: bar_lag=%ds, poll=%ds, until %s ET. NOTHING WILL BE SUBMITTED.",
             args.bar_lag, args.interval, args.until)
    run(args.bar_lag, args.interval, dt.time(hh, mm), args.max_candidates, args.force)

    if not args.skip_eod_short:
        from .eod_short import run_eod_short
        _wait_until(pd.Timestamp.combine(_today(), dt.time(16, 5)).tz_localize(ET))
        uni = pd.read_csv(DATA_DIR / "universe.csv")
        sym = "ticker" if "ticker" in uni else "symbol"
        # NASDAQ only (2026-08-07): the automated pick found VSEE, +560% on
        # $0.033 with no post-market data - it had moved to OTC since
        # universe.csv was last built and this filter never existed to catch
        # that. universe.csv's own "exchange" column is itself stale by
        # definition (a snapshot from whenever it was last rebuilt), so this
        # is a floor, not a guarantee - a ticker delisted SINCE the snapshot
        # can still slip through. universe.csv ships as a fixed snapshot now
        # (the builder was in the removed 5m-era CLI, see git history).
        if "exchange" in uni.columns:
            dropped = sorted(uni.loc[~uni["exchange"].eq("NASDAQ"), sym].unique())
            uni = uni[uni["exchange"].eq("NASDAQ")]
            if dropped:
                log.info("eod-short universe: dropped %d non-NASDAQ ticker(s) "
                         "(e.g. %s)", len(dropped), ", ".join(dropped[:5]))
        else:
            log.warning("universe.csv has no 'exchange' column - cannot "
                        "restrict eod-short to NASDAQ, running unfiltered")
        run_eod_short(sorted(uni[sym].unique()))


if __name__ == "__main__":
    main()
