# Lucky: does buying cheap stocks after they jump actually work?

Mostly no. This repo is how I found that out.

## What this is

lucky-run tests one trading idea from start to finish: when a cheap stock
gaps up at the open, does it keep running? Inside is a backtester, the one
strategy that held up under testing, a live paper-trading runner with a
dashboard, and five weeks of forward results that a pre-set stop rule cut
short. Every trade is in the repo.

## The thesis

Cheap stocks ($1 to $3) that gap up at the open on news tend to keep
drifting up during the day before they fade, and you should be able to trade
that drift with a target and a stop. I built a backtester, went looking for
a version of the rule that held up, found one narrow setup that looked good
on paper, and paper-traded it live for five weeks. Then a stop rule I'd
committed to in advance shut it down.

Most of the dead ends and the reasoning behind each parameter are in the
code comments. This file is the summary.

The Python package is `Lucky/`.

## Where the idea came from

Gap-up penny stocks are the loudest thing on the screen most mornings. Up
20%, up 50%, sometimes a few hundred percent, on many times their usual
volume. It feels like momentum you could ride: buyers crowd in, the stock
grinds higher through the morning, a target and a stop skim a piece of the
move.

A few trades I put on by hand pointed the other way, too: short the single
most stretched name right at the close and bet it gives back the last leg
overnight. Both are in here.

## Phase 1: the broad search (5-minute data)

The first pass threw every reasonable version of "buy the jump" at 5-minute
bars: 2,512 US stocks priced $15 or under, about 15,600 gap-up trading days.
June and July 2026, which is as far back as Yahoo's intraday history goes.

Seven variants. Buy at the open and hold to the close. Buy after a 3%, 5%,
or 10% confirmation. Wait for a volume-spike bar, then buy. Bet against the
move instead. Split everything by company float. On top of that, a 320-cell
grid over jump size, volume multiple, target, and stop. Every one of them
lost money once I ran it on an unfiltered set of days and scored it against
days it hadn't been tuned on.

Here's the part worth keeping. The only thing that reliably separated
winners from losers was how much the stock traded over the whole day, and
you can't know that when you'd be buying. The early results looked good
because the first test set had already thrown out 86% of gap days for not
being busy enough. Put those days back and the edge was gone. Quiet days
lost 2.5% a trade, the busiest days made 6%, and nothing tells you which
kind of day you're in until it's over.

The rest was backwards from what I expected. Bigger jumps did worse than
smaller ones. Waiting longer to buy helped. A wide stop always beat a tight
one, because the tight stop mostly fired on ordinary noise in stocks that
bounced straight back. Reward-to-risk ratio: didn't matter at all.

## Phase 2: narrowing down (1-minute data)

If no broad rule works, maybe one very specific rule does. Phase 2 switched
to 1-minute bars for about 2,470 tickers, roughly a 30-day window, and
looked for a single precise setup instead of a family of them.

The one that held up (`gap_vwap_strategy.py`):

```
Universe   common stock, opens $1.00 to $3.00, gapped up 4-20%
Signal     one 1-minute candle trading 5-20x that ticker's own average
           1-minute candle volume over the prior 10 sessions, and at least
           one bar so far that day has traded 75,000+ shares
Entry      price trades 5% above the signal candle's close and is above
           the session VWAP
             - signal expires if it doesn't trigger within 30 minutes
             - signal dies if price first breaks 10% below the signal candle
             - no new entries after 12:00 ET
Exit       10% target, 10% stop, otherwise flat at the close
             - after 14:30 ET, exit if price gives back 3% from its
               post-entry high
Sizing     5 fills a day at most, biggest gaps first
Costs      flat 1% round trip off every trade
```

On a curated set of 433 gap events (262 tickers, $1 to $3, gap of 4% or
more, all inside the 1-minute window):

| | |
|---|---|
| Trades | 33 |
| Average | +3.53% per trade, after the 1% cost |
| Win rate | 70% |
| Profit factor | 2.30 |
| Median hold | 16 minutes |
| In-sample vs out-of-sample | about even, +3.4% and +3.6% on a chronological split |

There's a secondary idea too: after the close, short the day's single
biggest gainer (close-to-close, price $1 to $10), 40% target, 20% stop,
cover by 18:00 ET.

I didn't believe any of it. 33 trades, out of maybe 13 distinct sessions,
in a single market. The headline t-stat
ignores a search that ran through roughly 40 formulations and a few thousand
parameter combinations on the same shrinking sample. A hypothesis to
forward-test. Not an edge.

## Phase 3: the forward test

A backtest you tuned tells you what you want to hear. The only test that
means anything is new data, run as if it were live, with the rules locked
first.

- `Lucky/runner.py` is the live dry run. It wakes at 09:31 ET, scans for
  gap-ups, polls every 20 seconds, runs the entry and exit logic on bars as
  they settle, and logs every fill. It sends no orders. There's no broker
  code in it. After the close it runs the end-of-day short check.
- `Lucky/dashboard.py` is a local web page: open positions, running P&L,
  kill-switch status.
- `Lucky/kill_switch.py` is the stop rule, and it was set in advance. The
  bounds come from the historical trade-to-trade standard deviation via a
  one-sided z-test; the sample size and confidence level were fixed before
  any forward data existed. So a losing run can't argue you out of quitting.

Five weeks of paper trading, August 2026, about 40 fills:

| | Trades | Result |
|---|---|---|
| Long strategy (the main idea) | 31 | -2.68% per trade, 32% win |
| End-of-day short (secondary) | 9 | +5.23% per trade, 78% win, tiny sample |
| Combined | 40 | -0.9% per trade, about -7% of the paper account |

At 31 trades the long strategy's running average crossed under the kill
line: -2.68% against a -2.61% threshold. The kill switch fired. The last run
of longs went 6 and 12.

## What I concluded

The main strategy didn't hold up on data it wasn't built on. The forward
number, -2.68% a trade, landed almost exactly where Phase 1 said it would: a
coin flip, once you stop hand-picking favorable days. The small-sample
warnings I kept writing into the code were right.

What worked was the process. Deciding the stop rule ahead of time meant the
call to quit was already made before the bad run instead of argued about
during it. The end-of-day short is up 5.23% a trade, which is nice, but nine
trades is noise.

If there's real money in gap-up penny stocks, it's probably not in the
price chart. It's in stuff this project never used: whether an actual
regulatory filing is behind the jump, and short-interest and borrow data.
That's where I'd go next.

## What's in the repo

| File | What it does |
|---|---|
| `gap_vwap_strategy.py` | the strategy: `Params`, `find_entry()`, `simulate_exit()`, `backtest()`, self-check |
| `Lucky/runner.py` | the live dry-run loop (scans, polls, logs, sends nothing) |
| `Lucky/dashboard.py`, `dashboard.html` | the local dashboard the runner serves |
| `Lucky/eod_short.py` | the after-the-close short check |
| `Lucky/kill_switch.py` | the pre-set stop rule |
| `Lucky/replay.py` | rebuilds a session the runner missed, from settled bars |
| `Lucky/probe_latency.py` | measures how stale and how revised Yahoo's 1-minute bars are |
| `Lucky/robustness.py` | bootstrap and Monte-Carlo checks on the strategy |
| `Lucky/alpaca.py`, `calibrate_alpaca.py` | Alpaca client, broker link only, never wired to the runner |
| `Lucky/yf_utils.py`, `cache.py`, `config.py` | Yahoo download helpers, a file cache, paths |
| `fetch_1m.py` | chunked 1-minute bar fetcher |

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Live dry run. Serves the dashboard, sleeps until the next 09:31 ET,
paper-trades the session:

```bash
caffeinate -i .venv/bin/python -m Lucky.runner
```

Strategy self-check:

```bash
.venv/bin/python gap_vwap_strategy.py
```

Reproduce the headline backtest:

```python
import pandas as pd, pickle, gap_vwap_strategy as S
bars = pickle.load(open("data/bars_1m_all.pkl", "rb"))
base = S.candle_baselines(bars)
e = pd.read_csv("data/events_1_3.csv", parse_dates=["date"])
print(S.summarise(S.backtest(e, bars, base, S.DEFAULT)))
# 33 trades, +3.53%/trade, 70% win, PF 2.30
```

Rebuild a session the runner missed:

```bash
.venv/bin/python -m Lucky.replay 2026-08-17 2026-08-27   # inclusive range
```

## The data

`data/` is committed, so you can dig through the results without re-fetching
anything. The two big 1-minute bar files (`bars_1m_all.pkl`, ~410 MB, and
`event_bars.pkl`, ~60 MB) are in Git LFS, so install `git-lfs` before you
clone, or rebuild them from Yahoo with `fetch_1m.py`. `cache/` isn't
committed; it's big and you can regenerate it.

The files that matter: `data/events_1_3.csv` (the 433-event validation set),
`data/bars_1m_all.pkl` (the 1-minute bars), `data/trade_log.csv` and
`data/eod_short_log.csv` (the full forward-test record),
`data/candidates_YYYYMMDD.csv` (every ticker the runner watched, by day).

### What the results can't tell you

- Small sample, one market. About 30 trading days of 1-minute data, and the
  strategy only fires on 13 or so of them.
- Survivorship bias. The universe is the current stock listing, so anything
  delisted since (often right after a crash) isn't in it.
- Costs are a flat 1% round trip. Real spreads on $1 to $3 stocks move
  around a lot, and they're worst in exactly the conditions this strategy
  goes after.
- No halts. Bars skip forward; the simulator has no idea an order couldn't
  actually have filled there.
- The data is Yahoo Finance: free, unofficial, about a 30-day window for
  1-minute bars, and bars that get revised after they first appear.

---

Not investment advice. Every number here is from testing against past data,
mostly paper trading, none of it from an account with real money in it.
Don't trade this.
