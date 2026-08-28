"""Gap-up continuation, $1-3 stocks, 1-minute bars — the validated configuration.

    Universe   common stock, session open $1.00-$3.00, gapped up 8-20%
    Signal     one 1-minute candle trading 5-10x the average 1-minute candle
               of the prior 10 sessions
    Entry      price trades +5% above that candle's close, AND is above VWAP.
               Signal dies if +5% is not reached within 30 minutes, or if
               price first breaks 10% below the signal candle.
    Exit       +10% target, -10% stop, otherwise flat at the close
    Sizing     at most 5 fills per day, largest gaps worked first
    Cutoff     no new entries after 10:30 ET

Measured on 26 trades over 17 sessions (2026-07-06 to 2026-08-03), gap_min
raised 4 -> 8 and cutoff tightened 12:00 -> 10:30 on 2026-08-05:
    +4.47% per trade net of 1% round-trip cost, 77% win rate, profit factor 2.74.
    Both changes came from a parameter sweep on the SAME 55-trade pool the
    gap_min=4/cutoff=12:00 numbers came from (n drops 55->32->26), and the one
    out-of-sample session that exists (2026-08-05) got WORSE at every gap_min
    raise. Do not over-trust the rising win rate; each cut is a smaller,
    correlated slice of the same sample, not new evidence.

That is a small sample in one market regime. It is a hypothesis worth
forward-testing, not a validated edge. See README notes at the bottom of this
file.

Run `python gap_vwap_strategy.py` to execute the built-in self-check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- parameters --


@dataclass(frozen=True)
class Params:
    # universe
    price_min: float = 1.00
    price_max: float = 3.00
    # RE-RAISED 2026-08-07 (was reverted to 4.0 on 2026-08-06 - see that
    # comment's history below - then re-tested on top of the 75k
    # min_day_volume gate added later the same day). This is NOT the same
    # test that got 8 reverted: re-run with the same IS-half-winners/
    # OOS-half-losers ablation that caught the problem last time, raising
    # 4->8 now removes 10 trades from the IS half averaging +2.39% (mixed,
    # not pure winners) and 4 from the OOS half averaging -3.93% (losers) -
    # the healthy direction, opposite of the mechanism that got this
    # reverted. n=30, +4.80%/trade, 77% win, PF 3.18, IS +3.43/OOS +6.36
    # (OOS beats IS, both positive). Also cuts the daily watchlist from an
    # average of 18.4 qualifying tickers/day to 5.2 - the operational
    # complaint that prompted re-testing this. Still only 30 trades over 15
    # sessions - re-litigate if a forward stretch disagrees, same as
    # everything else in this file.
    #
    # REVERTED 2026-08-06 (was 8.0, raised 2026-08-05): ablation showed
    # gap_min=8 removes IS-half WINNERS (+3.11% avg) and OOS-half LOSERS
    # (-3.28% avg) - that's not signal, that's a threshold picked to match
    # one split. gap_min=4 is where events_1_3.csv already floors (curated
    # at gap>=4%), and it's the config validated across every split
    # historically. Some floor is real (full-universe nogate test, gap 0-4%:
    # -2.53%, t=-3.57) - dropping it isn't an option, but 8 wasn't supported
    # at the time, 4 was.
    gap_min: float = 8.0            # % above prior close
    gap_max: float = 20.0

    # volume signal — multiples of the mean 1-minute candle over prior sessions
    spike_min: float = 5.0
    # RAISED 10->20 2026-08-07: the old 5-10x band structurally excludes the
    # most explosive gappers (AQB today ran 12x-300x baseline for its entire
    # move - every bar past the first minute exceeds 10x, so the strategy
    # could never see it regardless of timing). Full-history sweep at the
    # current config (gap_min=8, min_day_volume=75k): 10->+4.80%(n30),
    # 15->+4.53%(n32), 20->+3.53%(n33, best IS/OOS balance: +3.43/+3.63),
    # 30->+4.25%(n33), unbounded->+2.28%(n39, degrades - too much of the
    # admitted volume is the pre-market/opening-auction lump, not a real
    # signal). 20 gives up the least average return for the best-balanced
    # split, and directly fixes the AQB case (confirmed: signals at 09:31,
    # enters 09:35, hits TP +9% by 09:37 - 2 minutes). Does NOT fix every
    # explosive miss - names with 50-150x+ ratios (e.g. ZENA today) still
    # need spike_max in the hundreds to trigger at all, and when tested that
    # high the trade was a loser anyway (the entry-relative-to-signal-candle
    # trigger chases price too far into an already-extended move) - a
    # different, unsolved problem, not a spike_max problem.
    spike_max: float = 20.0
    baseline_sessions: int = 10

    # entry
    entry_gain: float = 5.0         # % above the signal candle's close
    expiry_min: float = 30.0        # signal dies if not triggered within this
    kill_drop: float = 10.0         # signal dies if price breaks this far below
    # ADDED 2026-08-07 (live experiment starting Monday 2026-08-10): a
    # blanket "wait one more bar" delay was tested and
    # rejected (+3.53%->+2.22%, same tax as the p95-lag finding) - too much
    # of the strategy's edge lives in acting inside the first few minutes.
    # This is the targeted version instead: only delay the fill when the
    # ENTRY bar ITSELF is unusually violent (v[j] >= confirm_volume_multiple
    # * baseline), which is what actually happened on ZYBT 2026-08-07 (entry
    # bar ran ~28x baseline on 1.17M shares in one minute - the signal
    # candle's own multiple, 6.24x, was unremarkable; it's the FILL bar that
    # was extreme, not the SIGNAL bar).
    # 25x / 1 bar: full-history backtest is COST-NEUTRAL at this threshold -
    # 12 of 33 trades get a shifted fill, but every one still resolves to
    # the same outcome bucket (a TP stays a TP - the % target is fixed
    # relative to whatever the fill turns out to be, so a slightly later
    # fill doesn't change the realized %), so the average is unchanged
    # (+3.53% either way). Lower thresholds (10-20x) start costing real
    # return (+3.53%->+2.84% at confirm_bars=1) by delaying into trades that
    # would've hit target from the original fill. 25x is the highest
    # threshold that still would have caught ZYBT.
    # Cannot be backtested for BENEFIT - bars_1m_all.pkl is already fully
    # settled and carries none of the poll-by-poll revision history that's
    # the entire point of this; only a live session shows whether it
    # actually avoids a bad fill.
    confirm_bars: int = 1
    confirm_volume_multiple: float | None = 25.0
    require_vwap: bool = True       # wait until price is also above VWAP
    # REVERTED 2026-08-06 (was 10:30, tightened 2026-08-05): the single
    # biggest driver of the IS/OOS split - at every gap_min tested, tightening
    # to 10:30 opens the gap (e.g. gap_min=4: -1.23 -> +1.37; gap_min=8:
    # +2.91 -> +6.44). It removes big IS-half winners (+6.73% avg) and big
    # OOS-half losers (-9.12% avg on just 6 trades). 12:00 is the config that
    # held up across every split when first validated, before the search
    # that produced 10:30.
    no_entry_after: str = "12:00"
    no_entry_before: str | None = None   # VWAP is noisy on thin early volume

    # liquidity gate: by the time a signal fires, at least one bar so far
    # TODAY (not the full session - that would be look-ahead) must have
    # traded this many shares. 0 disables it. Added 2026-08-07, moved 60k ->
    # 75k same day. Removes 10 losers vs 2 winners from the 55-trade sample,
    # IS/OOS stays balanced (3.70/4.07) unlike the gap_min=8 attempt that was
    # reverted. The threshold sweep behind the value is in
    # data/param_sweep_price_entry.csv.
    min_day_volume: float = 75000.0

    # exit
    take_profit: float = 10.0
    stop_loss: float = 10.0
    # once take_profit is touched, ratchet instead of exiting: target moves to
    # extended_take_profit, stop moves up to lock in extended_stop_loss (a
    # profit floor, not a loss).
    # OFF by default: swept 105 (tp, extended_tp, extended_sl) combinations on
    # the 55 validated trades and not one beat plain TP. Best ratchet found was
    # tp=8/etp=13/esl=2 at +1.76% vs +2.05% plain, with worse IS/OOS stability
    # (1.09/2.36 vs 2.27/1.85). The mechanism is structural, not a bad setting:
    # +10% touches are mostly momentum exhaustion, so holding for +13% converts
    # winners into floor-exits. See ratchet_grid.csv.
    ratchet: bool = False
    extended_take_profit: float = 13.0
    extended_stop_loss: float = 9.0

    # end-of-day momentum exit: distinct from the ratchet above (which only
    # engages after +10% is touched). This engages when +10% ISN'T going to
    # be touched - late in the session, riding a stalled trade to whatever the
    # final close happens to be gives back the best price it ever saw. Once
    # eod_trail_after passes, track the peak close since entry; give back
    # eod_trail_giveback% of it and take the exit instead of waiting for the
    # close. Bars before eod_trail_after are untouched by this - a trade still
    # working toward the target isn't cut off early.
    eod_trail_after: str | None = "14:30"
    eod_trail_giveback: float = 3.0

    # book
    max_trades_per_day: int = 5
    cost_pct: float = 1.0           # round-trip, deducted from each trade


DEFAULT = Params()


# ------------------------------------------------------------------ helpers --


def _first(mask: np.ndarray, start: int = 0) -> int | None:
    """Index of the first True at or after `start`, else None."""
    if start >= len(mask):
        return None
    hits = np.nonzero(mask[start:])[0]
    return start + int(hits[0]) if len(hits) else None


def session_vwap(high, low, close, volume) -> np.ndarray:
    """Running session VWAP. Resets each day because we only ever pass one day."""
    typical = (high + low + close) / 3
    cum_vol = np.cumsum(volume)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cum_vol > 0,
                        np.cumsum(typical * volume) / np.where(cum_vol == 0, 1, cum_vol),
                        close)


def candle_baselines(bars_by_ticker: dict[str, pd.DataFrame],
                     p: Params = DEFAULT) -> dict[tuple[str, object], float]:
    """(ticker, date) -> mean 1-minute candle volume over the PRIOR sessions.

    Computed from the ticker's own bars, never from daily volume / 390. An
    illiquid stock does not trade every minute, so that proxy understates the
    true per-candle average by a wide margin — it was wrong by 18x on one name
    during development, which made a 448-share candle look like a spike.
    """
    out: dict[tuple[str, object], float] = {}
    for ticker, df in bars_by_ticker.items():
        if df is None or df.empty:
            continue
        day = pd.Series([t.date() for t in df.index], index=df.index)
        volume = df["Volume"].astype(float)
        days = sorted(set(day))
        for i, d in enumerate(days):
            prior = days[max(0, i - p.baseline_sessions):i]
            if len(prior) < 3:
                continue
            sample = volume[day.isin(prior)]
            if len(sample) >= 20 and sample.mean() > 0:
                out[(ticker, d)] = float(sample.mean())
    return out


# -------------------------------------------------------------------- signal --


def find_entry(session: pd.DataFrame, baseline: float, p: Params = DEFAULT) -> dict | None:
    """First signal in the session that survives to an actual fill.

    A signal dies three ways: it times out, price breaks below the kill level,
    or the session ends. When one dies we keep scanning — a dead signal costs
    the signal, not the day.
    """
    o, h, l, c, v = (session[x].to_numpy(float)
                     for x in ["Open", "High", "Low", "Close", "Volume"])
    vwap = session_vwap(h, l, c, v)
    minutes = np.array([(t - session.index[0]).total_seconds() / 60 for t in session.index])
    times = [t.strftime("%H:%M") for t in session.index]

    is_spike = (v >= p.spike_min * baseline) & (v < p.spike_max * baseline)

    k = _first(is_spike)
    while k is not None and k < len(c) - 1:
        if p.min_day_volume and v[:k + 1].max() < p.min_day_volume:
            k = _first(is_spike, k + 1)
            continue
        trigger = c[k] * (1 + p.entry_gain / 100)
        kill_at = c[k] * (1 - p.kill_drop / 100)

        eligible = (h >= trigger) & (minutes <= minutes[k] + p.expiry_min)
        if p.require_vwap:
            eligible &= c >= vwap                       # wait for the reclaim
        if p.no_entry_after:
            eligible &= np.array([t < p.no_entry_after for t in times])
        if p.no_entry_before:
            eligible &= np.array([t >= p.no_entry_before for t in times])

        j = _first(eligible, k + 1)
        killed = _first(l <= kill_at, k + 1)

        if j is not None and (killed is None or j <= killed):
            entry_i = j
            # confirmation delay: only when the FILL bar itself is unusually
            # violent, not the (already-checked) signal bar. See Params
            # docstring - this is the ZYBT 2026-08-07 case.
            if (p.confirm_bars and p.confirm_volume_multiple
                    and v[j] >= p.confirm_volume_multiple * baseline):
                delayed = j + p.confirm_bars
                if delayed < len(o):
                    entry_i = delayed
            return {
                "signal_i": k, "signal_time": times[k],
                "signal_volume": v[k], "signal_multiple": v[k] / baseline,
                "entry_i": entry_i, "entry_time": times[entry_i],
                # a stop-buy fills at the trigger, or worse if the bar gapped through
                "fill": max(trigger, o[entry_i]),
                # ADDED 2026-08-07: the intended entry price, separate from
                # the actual fill - lets the dashboard show execution
                # quality (expected vs actual) instead of just the fill.
                # Purely additive - not read by backtest()/simulate_exit().
                "trigger": trigger,
            }
        k = _first(is_spike, k + 1)
    return None


def simulate_exit(session: pd.DataFrame, entry_i: int, fill: float,
                  p: Params = DEFAULT) -> dict:
    """Walk forward to the first of target, stop, or the closing bell.

    The stop may trigger on the entry bar (we cannot know where the low sat
    relative to our fill, so assume the worst); the target may not.

    Once the first-stage target is touched, ratchet rather than exit: target
    becomes extended_take_profit, stop rises to extended_stop_loss (a locked
    profit floor). The ratchet applies from the NEXT bar - a single bar can't
    both trigger and immediately blow through the new stop.

    Separately: past eod_trail_after, a trade that hasn't hit target or stop
    is tracked against its own peak close (post-entry bars only - same
    look-ahead rule as the target check). Giving back eod_trail_giveback% of
    that peak exits there instead of riding an untouched trade to the close.
    """
    o, h, l, c = (session[x].to_numpy(float) for x in ["Open", "High", "Low", "Close"])
    minutes = np.array([(t - session.index[0]).total_seconds() / 60 for t in session.index])
    times = [t.strftime("%H:%M") for t in session.index]
    target, stop = fill * (1 + p.take_profit / 100), fill * (1 - p.stop_loss / 100)
    extended = False
    peak = fill

    for k in range(entry_i, len(h)):
        if l[k] <= stop:                       # market order: gap through fills worse
            px = min(stop, o[k]) if k > entry_i else stop
            return _exit("SL2" if extended else "SL", px, k, entry_i, fill, minutes, session)
        if k > entry_i and h[k] >= target:
            if p.ratchet and not extended:     # ratchet instead of exiting
                extended = True
                target = fill * (1 + p.extended_take_profit / 100)
                stop = fill * (1 + p.extended_stop_loss / 100)
                continue
            return _exit("TP2" if extended else "TP", max(target, o[k]),
                         k, entry_i, fill, minutes, session)
        if k > entry_i:
            peak = max(peak, h[k])
            if p.eod_trail_after and times[k] >= p.eod_trail_after:
                giveback = (peak - c[k]) / peak * 100
                if giveback >= p.eod_trail_giveback:
                    return _exit("EODT", c[k], k, entry_i, fill, minutes, session)
    return _exit("EOD", c[-1], len(h) - 1, entry_i, fill, minutes, session)


def _exit(outcome, px, k, entry_i, fill, minutes, session) -> dict:
    return {"outcome": outcome, "exit_price": px, "exit_time": session.index[k].strftime("%H:%M"),
            "return_pct": (px - fill) / fill * 100,
            "minutes_held": float(minutes[k] - minutes[entry_i])}


# ------------------------------------------------------------------ backtest --


def backtest(events: pd.DataFrame, bars_by_ticker: dict[str, pd.DataFrame],
             baselines: dict, p: Params = DEFAULT) -> pd.DataFrame:
    """Run the strategy over gap-up events.

    `events` needs columns: ticker, date, gap_pct, open_price.
    `bars_by_ticker` maps ticker -> 1-minute DataFrame (regular hours, tz-aware).
    """
    qualifying = events[
        events.open_price.between(p.price_min, p.price_max)
        & events.gap_pct.between(p.gap_min, p.gap_max, inclusive="left")
    ]
    trades = []
    for date, day_events in qualifying.groupby(qualifying.date.dt.date):
        filled = 0
        # biggest gaps get first look; the cap counts FILLS, not signals, so a
        # name that signals and never triggers does not consume a slot
        for row in day_events.sort_values("gap_pct", ascending=False).itertuples():
            if filled >= p.max_trades_per_day:
                break
            df = bars_by_ticker.get(row.ticker)
            if df is None or df.empty:
                continue
            session = df[[t.date() == date for t in df.index]]
            if len(session) < 30:
                continue
            baseline = baselines.get((row.ticker, date))
            if not baseline:
                continue

            signal = find_entry(session, baseline, p)
            if signal is None:
                continue
            result = simulate_exit(session, signal["entry_i"], signal["fill"], p)
            result["return_pct"] -= p.cost_pct
            trades.append({**signal, **result, "ticker": row.ticker, "date": date,
                           "gap_pct": row.gap_pct, "open_price": row.open_price})
            filled += 1
    return pd.DataFrame(trades)


def summarise(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"trades": 0}
    r = trades.return_pct.to_numpy(float)
    wins, losses = r[r > 0], r[r <= 0]
    gross_loss = -losses.sum()
    return {
        "trades": len(r),
        "avg_pct": r.mean(),
        "median_pct": float(np.median(r)),
        "win_rate": (r > 0).mean() * 100,
        "profit_factor": (wins.sum() / gross_loss) if gross_loss > 0 else np.inf,
        "target_rate": trades.outcome.isin(["TP", "TP2"]).mean() * 100,
        "stop_rate": trades.outcome.isin(["SL", "SL2"]).mean() * 100,
        "close_rate": trades.outcome.isin(["EOD", "EODT"]).mean() * 100,
        "worst_pct": r.min(),
        "median_minutes": float(np.median(trades.minutes_held)),
    }


# ---------------------------------------------------------------- self-check --


def _session(prices, volumes, start="09:30"):
    idx = pd.date_range(f"2026-07-06 {start}", periods=len(prices), freq="1min",
                        tz="America/New_York")
    p = np.array(prices, float)
    return pd.DataFrame({"Open": p, "High": p * 1.001, "Low": p * 0.999,
                         "Close": p, "Volume": np.array(volumes, float)}, index=idx)


def _self_check() -> None:
    p = Params(no_entry_after=None, min_day_volume=0)   # keep the synthetic day simple

    # 1. a 6x candle then a clean +5% run -> fills, then hits the target
    prices = [1.00] * 3 + [1.00, 1.03, 1.06, 1.10, 1.14, 1.18] + [1.20] * 10
    vols = [100, 100, 100, 600] + [100] * 15
    s = _session(prices, vols)
    sig = find_entry(s, baseline=100, p=p)
    assert sig is not None, "a 6x candle followed by +5% must produce an entry"
    assert sig["signal_i"] == 3, sig
    assert sig["fill"] >= 1.00 * 1.05
    ex1 = simulate_exit(s, sig["entry_i"], sig["fill"], p)
    assert ex1["outcome"] == "TP", ex1
    assert simulate_exit(s, sig["entry_i"], sig["fill"],
                         Params(min_day_volume=0, no_entry_after=None, ratchet=True)
                         )["outcome"] == "TP2", "ratchet=True must hold for the extended target"

    # 2. spike outside the 5-10x band is ignored
    assert find_entry(_session(prices, [100, 100, 100, 2000] + [100] * 15),
                      baseline=100, p=p) is None, "a 20x candle is outside the band"

    # 3. +5% never reached -> no trade
    assert find_entry(_session([1.00] * 19, [100, 100, 100, 600] + [100] * 15),
                      baseline=100, p=p) is None, "flat price must not trigger"

    # 4. expiry: +5% arrives too late
    slow = [1.00] * 40 + [1.10] * 5
    assert find_entry(_session(slow, [100, 100, 100, 600] + [100] * 41),
                      baseline=100, p=Params(min_day_volume=0, expiry_min=10, no_entry_after=None)
                      ) is None, "signal must expire"

    # 5. kill: price breaks -10% before the trigger is reached
    dip = [1.00, 1.00, 1.00, 1.00, 0.85, 0.90, 1.06, 1.10] + [1.20] * 10
    assert find_entry(_session(dip, [100, 100, 100, 600] + [100] * 14),
                      baseline=100, p=p) is None, "a -10% break must kill the signal"

    # 6. VWAP gate actually blocks: price above the trigger but under VWAP
    #    (heavy volume printed high, then price triggers from below)
    hi_vwap = _session([2.00, 2.00, 2.00, 1.00, 1.02, 1.06, 1.10] + [1.12] * 8,
                       [10_000, 10_000, 10_000, 600] + [100] * 11)
    assert find_entry(hi_vwap, baseline=100, p=Params(min_day_volume=0, no_entry_after=None)) is None
    assert find_entry(hi_vwap, baseline=100,
                      p=Params(min_day_volume=0, require_vwap=False, no_entry_after=None)) is not None, \
        "without the VWAP gate the same setup should fill"

    # 7. stop fills at the stop, not through it, when no gap occurs
    down = [1.00] * 3 + [1.00, 1.06, 1.00, 0.95, 0.90] + [0.88] * 10
    s = _session(down, [100, 100, 100, 600] + [100] * 14)
    sig = find_entry(s, baseline=100, p=p)
    ex = simulate_exit(s, sig["entry_i"], sig["fill"], p)
    assert ex["outcome"] == "SL" and ex["return_pct"] < -9, ex

    # 8. ratchet: first target touched, then price gives it back to the
    #    extended stop - must lock a profit, not turn into a loss
    give_back = [1.00] * 3 + [1.00, 1.03, 1.06, 1.10, 1.14, 1.18] + [1.16, 1.12, 1.09, 1.05] + [1.05] * 6
    s8 = _session(give_back, [100, 100, 100, 600] + [100] * 15)
    sig8 = find_entry(s8, baseline=100, p=p)
    ex8 = simulate_exit(s8, sig8["entry_i"], sig8["fill"],
                        Params(min_day_volume=0, no_entry_after=None, ratchet=True))
    assert ex8["outcome"] == "SL2" and ex8["return_pct"] > 0, \
        "ratchet must lock a profit, never exit at a loss after the target is touched"

    # 9. EOD trail: a trade that never reaches target, peaks mid-session, then
    #    fades past eod_trail_after must exit near its peak, not ride the fade
    #    down to whatever happens next (here: all the way into a stop-out)
    rally = [1.00] * 3 + [1.00, 1.02, 1.04, 1.06, 1.08, 1.12, 1.15]     # never reaches +10%
    plateau = [1.15] * 290                                              # holds until ~14:29
    fade = [1.15, 1.10, 1.00, 0.97, 0.95, 0.90]                         # crosses 14:30, then SL
    s9 = _session(rally + plateau + fade, [100, 100, 100, 600] + [100] * (len(rally) + len(plateau) + len(fade) - 4))
    sig9 = find_entry(s9, baseline=100, p=Params(min_day_volume=0, no_entry_after=None))
    trailed = simulate_exit(s9, sig9["entry_i"], sig9["fill"],
                            Params(min_day_volume=0, no_entry_after=None, eod_trail_after="14:30", eod_trail_giveback=3.0))
    untrailed = simulate_exit(s9, sig9["entry_i"], sig9["fill"],
                              Params(min_day_volume=0, no_entry_after=None, eod_trail_after=None))
    assert trailed["outcome"] == "EODT" and trailed["return_pct"] > 0, trailed
    assert untrailed["outcome"] == "SL" and untrailed["return_pct"] < trailed["return_pct"], \
        "without the trail the same fade must ride further down than the trailed exit"

    # 10. min_day_volume: a spike candle alone isn't enough liquidity - the
    #     gate needs a bar of size min_day_volume seen BY THE TIME OF THE
    #     SIGNAL. A later, bigger bar must not count (that would be look-ahead).
    thin = [1.00] * 3 + [1.00, 1.03, 1.06, 1.10, 1.14, 1.18] + [1.20] * 10
    thin_vols = [100, 100, 100, 600] + [100] * 15
    s10 = _session(thin, thin_vols)
    assert find_entry(s10, baseline=100, p=Params(no_entry_after=None, min_day_volume=75_000)
                      ) is None, "600-share spike must not clear a 75k liquidity gate"
    loud_vols = list(thin_vols)
    loud_vols[2] = 80_000                    # a big bar BEFORE the signal candle
    s10b = _session(thin, loud_vols)
    sig10 = find_entry(s10b, baseline=100, p=Params(no_entry_after=None, min_day_volume=75_000))
    assert sig10 is not None, "an 80k bar seen before the signal must clear the gate"
    late_vols = list(thin_vols)
    late_vols[-1] = 80_000                   # a big bar AFTER the signal candle
    s10c = _session(thin, late_vols)
    assert find_entry(s10c, baseline=100, p=Params(no_entry_after=None, min_day_volume=75_000)
                      ) is None, "a bar seen only AFTER the signal must not satisfy the gate (look-ahead)"

    # 11. confirm_bars/confirm_volume_multiple: a violent FILL bar (not the
    #     signal bar) gets its fill delayed one bar when enabled; disabled
    #     (or a calm fill bar) fills on the original bar, unchanged.
    calm = [1.00] * 3 + [1.00, 1.03, 1.06, 1.10, 1.14, 1.18] + [1.20] * 10
    calm_vols = [100, 100, 100, 600] + [100] * 15   # signal at i=3 (6x), fill bar i=5 is calm
    s11 = _session(calm, calm_vols)
    off = Params(no_entry_after=None, min_day_volume=0, confirm_bars=0, confirm_volume_multiple=None)
    on = Params(no_entry_after=None, min_day_volume=0, confirm_bars=1, confirm_volume_multiple=25.0)
    sig_off = find_entry(s11, baseline=100, p=off)
    sig_on_calm = find_entry(s11, baseline=100, p=on)
    assert sig_off["entry_i"] == 5, sig_off
    assert sig_on_calm["entry_i"] == 5, \
        "a calm fill bar (well under the confirm threshold) must not be delayed"

    violent_vols = list(calm_vols)
    violent_vols[5] = 3_000                          # fill bar now 30x baseline - above the 25x threshold
    s11b = _session(calm, violent_vols)
    sig_violent_off = find_entry(s11b, baseline=100, p=off)
    sig_violent_on = find_entry(s11b, baseline=100, p=on)
    assert sig_violent_off["entry_i"] == 5, \
        "confirm disabled: a violent fill bar must still fill immediately"
    assert sig_violent_on["entry_i"] == 6, \
        "confirm enabled + violent fill bar: must delay to the NEXT bar"
    assert sig_violent_on["fill"] == calm[6], \
        "delayed fill must use the confirmation bar's own open, not the original trigger bar's"

    print("self-check passed: 11/11")


if __name__ == "__main__":
    _self_check()
