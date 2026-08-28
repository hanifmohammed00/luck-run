"""Alpaca market-data client - the replacement data source for yfinance.

Why this exists: yfinance caps intraday history at ~30 days for 1-minute bars
(the dominant limitation - every sample is small because of it).
Alpaca serves 1-minute bars back to 2016 with no per-request day cap beyond
pagination, which is the only route to a sample this strategy can actually be
tested on.

FEED WARNING - read before trusting any number that comes out of here.
The free plan serves the `iex` feed: trades printed on IEX only, which is a
single-digit percentage of the consolidated tape. This strategy's entire signal
is a VOLUME RATIO (one candle at 5-10x the mean candle), and its gate is VWAP.
Both are computed from volume, so a partial tape does not just add noise - it
can move the signal systematically. `sip` (Algo Trader Plus) is the full tape.

Run `python -m Lucky.calibrate_alpaca` before using IEX data for research: it
diffs Alpaca bars against the cached yfinance consolidated bars on the same
tickers and dates and reports the actual coverage, rather than assuming it.

Bars are normalised to the yfinance column convention (Open/High/Low/Close/
Volume, tz-aware America/New_York index) so they drop into the existing code.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from .cache import Cache

log = logging.getLogger(__name__)

DATA_BASE = "https://data.alpaca.markets"
PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"

# Alpaca returns bars as {t,o,h,l,c,v,n,vw}; map to the yfinance names the rest
# of the codebase already speaks. `n` (trade count) and `vw` (bar VWAP) are
# extras yfinance never gave us - kept, since `n` is a real liquidity signal on
# $1-3 names where a bar's volume can be one print.
BAR_COLUMNS = {
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "v": "Volume",
    "n": "TradeCount",
    "vw": "BarVWAP",
}

MAX_LIMIT = 10_000  # bars per page, Alpaca's ceiling
SYMBOLS_PER_REQUEST = 100


def credentials() -> tuple[str, str]:
    """Read the key/secret from the environment, falling back to .env.

    Mirrors Lucky.fmp.api_key(). Accepts either Alpaca's own env var names
    (APCA_API_KEY_ID / APCA_API_SECRET_KEY) or the ALPACA_* spelling.
    """
    names = (
        ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
        ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
        ("ALPACA_KEY_ID", "ALPACA_SECRET_KEY"),
    )
    env_file: dict[str, str] = {}
    path = Path(__file__).resolve().parent.parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_file[k.strip()] = v.strip()

    for key_name, secret_name in names:
        key = os.environ.get(key_name) or env_file.get(key_name, "")
        secret = os.environ.get(secret_name) or env_file.get(secret_name, "")
        if key and secret:
            return key, secret

    raise SystemExit(
        "Alpaca credentials not set. Add to .env (which is gitignored):\n"
        "  APCA_API_KEY_ID=...\n"
        "  APCA_API_SECRET_KEY=...\n"
        "Generate them at https://app.alpaca.markets/paper/dashboard/overview"
    )


class Alpaca:
    """Market-data client. One instance per run; holds the cache and call count."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        feed: str = "iex",
        adjustment: str = "raw",
        pause: float = 0.0,
    ) -> None:
        self.key, self.secret = credentials()
        self.cache = Cache(cache_dir, "alpaca")
        self.feed = feed
        # 'raw' keeps traded prices: a gap must be measured on what actually
        # printed, not on a back-adjusted series (the split caveat).
        self.adjustment = adjustment
        self.pause = pause
        self.calls = 0
        self.blocked = False

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "accept": "application/json",
        }

    def _get(self, url: str, params: dict | None = None) -> dict | None:
        for attempt in range(4):
            try:
                r = requests.get(url, headers=self._headers, params=params, timeout=30)
            except Exception as exc:
                log.warning("alpaca %s failed: %s", url, exc)
                return None
            self.calls += 1
            if self.pause:
                time.sleep(self.pause)

            if r.status_code == 429:
                # 200 req/min on the free plan; back off rather than give up.
                wait = 2 ** attempt
                log.warning("alpaca rate limited, sleeping %ds (call %d)", wait, self.calls)
                time.sleep(wait)
                continue
            if r.status_code in (401, 403):
                log.error("alpaca auth/subscription rejected (%d): %s", r.status_code, r.text[:200])
                self.blocked = True
                return None
            if r.status_code != 200:
                log.warning("alpaca %s -> HTTP %d: %s", url, r.status_code, r.text[:200])
                return None
            try:
                return r.json()
            except Exception:
                return None
        log.warning("alpaca gave up after repeated rate limits")
        return None

    # ---------- bars ----------

    def bars(
        self,
        symbols: str | list[str],
        start: dt.date | dt.datetime | str,
        end: dt.date | dt.datetime | str,
        timeframe: str = "1Min",
        *,
        session_only: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Historical bars for one or many symbols.

        Returns {symbol: DataFrame} with yfinance-style columns and a
        tz-aware America/New_York index. Handles pagination transparently.
        `session_only` trims to the 09:30-16:00 regular session, matching what
        the cached yfinance bars contain.
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        symbols = sorted(set(symbols))
        out: dict[str, pd.DataFrame] = {}

        for i in range(0, len(symbols), SYMBOLS_PER_REQUEST):
            chunk = symbols[i : i + SYMBOLS_PER_REQUEST]
            raw = self._bars_page_loop(chunk, start, end, timeframe)
            for sym, rows in raw.items():
                frame = _to_frame(rows, session_only=session_only)
                if not frame.empty:
                    out[sym] = frame
            if self.blocked:
                break
        return out

    def _bars_page_loop(self, symbols, start, end, timeframe) -> dict[str, list]:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": _iso(start),
            "end": _iso(end),
            "limit": MAX_LIMIT,
            "adjustment": self.adjustment,
            "feed": self.feed,
            "sort": "asc",
        }
        collected: dict[str, list] = {}
        token = None
        while True:
            if token:
                params["page_token"] = token
            data = self._get(f"{DATA_BASE}/v2/stocks/bars", params)
            if data is None:
                break
            for sym, rows in (data.get("bars") or {}).items():
                collected.setdefault(sym, []).extend(rows)
            token = data.get("next_page_token")
            if not token:
                break
        return collected

    def cached_bars(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        timeframe: str = "1Min",
    ) -> pd.DataFrame:
        """Per-symbol cached fetch, keyed on the exact window and feed.

        Keyed by feed so IEX and SIP pulls can never be silently mixed in one
        cache directory - that would be undebuggable.
        """
        key = f"{symbol}_{timeframe}_{self.feed}_{start:%Y%m%d}_{end:%Y%m%d}"
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        frame = self.bars(symbol, start, end, timeframe).get(symbol, pd.DataFrame())
        self.cache.put(key, frame)
        return frame

    # ---------- reference data ----------

    def assets(self, *, tradable_only: bool = True) -> pd.DataFrame:
        """Active US equities, with Alpaca's own tradability/shortability flags.

        This is strictly better than the NASDAQ Trader directory for the live
        side: `tradable` and `easy_to_borrow` tell you what you can actually
        get a fill on, which a listing file cannot.
        """
        data = self._get(
            f"{PAPER_BASE}/v2/assets",
            {"status": "active", "asset_class": "us_equity"},
        )
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if tradable_only and "tradable" in df:
            df = df[df["tradable"]]
        return df.reset_index(drop=True)

    def news(self, symbols, start, end, limit: int = 50) -> pd.DataFrame:
        """Benzinga-sourced headlines. Free on all plans.

        Kept for a possible news filter: separating gaps backed by real news
        from promotion is the untested variable, and this reaches it without
        spending FMP's ~250/day quota.
        """
        params = {
            "symbols": ",".join(symbols) if not isinstance(symbols, str) else symbols,
            "start": _iso(start),
            "end": _iso(end),
            "limit": limit,
            "sort": "asc",
        }
        rows, token = [], None
        while True:
            if token:
                params["page_token"] = token
            data = self._get(f"{DATA_BASE}/v1beta1/news", params)
            if not data:
                break
            rows.extend(data.get("news") or [])
            token = data.get("next_page_token")
            if not token:
                break
        return pd.DataFrame(rows)

    def calendar(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Real trading calendar incl. early closes - the simulator currently
        assumes a 09:30-16:00 session for every day."""
        data = self._get(
            f"{PAPER_BASE}/v2/calendar",
            {"start": _iso(start)[:10], "end": _iso(end)[:10]},
        )
        return pd.DataFrame(data or [])


def _iso(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dt.datetime):
        return value.isoformat()
    return dt.datetime.combine(value, dt.time.min).isoformat() + "Z"


def _to_frame(rows: list[dict], *, session_only: bool) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "t" not in df:
        return pd.DataFrame()
    idx = pd.to_datetime(df["t"], format="ISO8601", utc=True)
    df = df.drop(columns=["t"]).rename(columns=BAR_COLUMNS)
    df.index = idx.dt.tz_convert("America/New_York")
    df.index.name = "Datetime"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if session_only:
        df = df.between_time("09:30", "16:00", inclusive="left")
    return df
