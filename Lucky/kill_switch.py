"""Pre-committed stop rule for the long strategy.

The problem it solves: after a losing stretch, re-tuning entry/exit params
to fix it is how gap_min=8/cutoff=10:30 happened in the first place (fitted
to a favourable stretch, reverted 2026-08-06). The fix isn't a better rule
for WHEN to trade, it's a rule for WHEN TO STOP that's decided before the
data arrives, so a bad week can't rewrite it.

Bounds come from the historical std (8.84%, from the 55-trade validated
sample) via a one-sided z-test against zero:
    kill_line    = -z * std/sqrt(n)   (reject "true edge >= 0")
    confirm_line = +z * std/sqrt(n)   (reject "true edge <= 0")
n_min=20 and confidence=95% are standard statistical defaults, NOT fit to
any outcome - see _check_calibration for why that matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import DATA_DIR

# one-sided z-scores for common confidence levels
_Z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


@dataclass(frozen=True)
class KillParams:
    n_min: int = 20                 # don't call a verdict on fewer trades than this
    confidence: float = 0.95
    std_estimate: float = 8.84      # %, from the 55-trade historical sample


def bounds(n: int, p: KillParams = KillParams()) -> tuple[float, float]:
    """(kill_line, confirm_line) in %/trade, at sample size n."""
    z = _Z[p.confidence]
    se = p.std_estimate / math.sqrt(n)
    return -z * se, z * se


def evaluate(returns: list[float], p: KillParams = KillParams()) -> dict:
    """Verdict on a sequence of trade returns (%), oldest first.

    TOO_EARLY  - fewer than n_min trades, no verdict possible
    KILL       - running avg is significantly below zero
    CONFIRM    - running avg is significantly above zero
    CONTINUE   - inconclusive, keep going
    """
    n = len(returns)
    if n < p.n_min:
        return {"verdict": "TOO_EARLY", "n": n, "n_min": p.n_min, "avg": None,
                "kill_line": None, "confirm_line": None}
    avg = sum(returns) / n
    lo, hi = bounds(n, p)
    verdict = "KILL" if avg <= lo else "CONFIRM" if avg >= hi else "CONTINUE"
    return {"verdict": verdict, "n": n, "n_min": p.n_min, "avg": avg,
            "kill_line": lo, "confirm_line": hi}


def first_verdict(returns: list[float], p: KillParams = KillParams()) -> dict | None:
    """Walk the sequence chronologically; return the FIRST non-CONTINUE
    verdict, or None if it never resolves. This is what a live deployment
    actually experiences - it stops at the first trigger, not at the end."""
    for i in range(p.n_min, len(returns) + 1):
        v = evaluate(returns[:i], p)
        if v["verdict"] != "CONTINUE":
            return v
    return None


def _check_calibration() -> dict:
    """Does the rule false-kill a sample known to have a positive edge?

    Walks the ORIGINAL 55-trade validated sequence (chronological, the data
    the std_estimate itself comes from) and checks whether the rule would
    have killed it before the end. That sample's true average is +2.00% -
    a good rule should rarely kill it. This is a check on the RULE's
    calibration, not a search for a rule that scores well; n_min=20 and
    confidence=95% were picked before this was run, not after.
    """
    import pickle
    import gap_vwap_strategy as S

    bars = pickle.load(open(DATA_DIR / "bars_1m_all.pkl", "rb"))
    base = S.candle_baselines(bars)
    e = pd.read_csv(DATA_DIR / "events_1_3.csv", parse_dates=["date"])
    tr = S.backtest(e, bars, base, S.DEFAULT).sort_values("date")
    seq = list(tr.return_pct)

    results = {}
    for n_min in (15, 20, 30):
        for conf in (0.90, 0.95):
            v = first_verdict(seq, KillParams(n_min=n_min, confidence=conf))
            results[(n_min, conf)] = v["verdict"] if v else "never triggers"
    return results


def _self_check() -> None:
    p = KillParams(n_min=5, std_estimate=10.0)
    lo5, hi5 = bounds(5, p)     # +-9.19% at n=5, std=10

    # a run of clean losers well past the line must kill
    assert evaluate([-15.0] * 5, p)["verdict"] == "KILL"
    # a run of clean winners well past the line must confirm
    assert evaluate([15.0] * 5, p)["verdict"] == "CONFIRM"
    # mixed, fewer than n_min: no verdict yet
    assert evaluate([1.0, -1.0, 2.0, -2.0], p)["verdict"] == "TOO_EARLY"
    # mixed, at n_min, inside the bounds: keep going
    assert evaluate([1.0, -1.0, 0.5, -0.5, 0.2], p)["verdict"] == "CONTINUE"

    # bounds must tighten as n grows (more evidence -> smaller goalposts)
    lo10, hi10 = bounds(10, p)
    lo40, hi40 = bounds(40, p)
    assert abs(lo40) < abs(lo10) and hi40 < hi10

    # first_verdict stops at the first trigger, not the end of a longer run
    seq = [-15.0] * 5 + [20.0] * 5     # kills early on the losers, ignores the later recovery
    early = first_verdict(seq, p)
    assert early is not None and early["verdict"] == "KILL" and early["n"] == 5

    print("kill_switch self-check passed: 6/6")


if __name__ == "__main__":
    _self_check()
    print("\ncalibration check: does the rule false-kill the known-positive 55-trade sample?")
    for (n_min, conf), verdict in _check_calibration().items():
        print(f"  n_min={n_min:2d} conf={conf:.0%}  -> {verdict}")
