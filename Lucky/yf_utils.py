"""Thin wrappers around yfinance that normalise its column layouts."""

from __future__ import annotations

import logging
import warnings

import pandas as pd

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _import_yf():
    import yfinance as yf  # imported lazily so --help works without the dep

    return yf


def split_batch_frame(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Turn a yf.download result into {ticker: OHLCV frame}.

    Handles both the single-ticker (flat columns) and multi-ticker
    (MultiIndex) layouts, in either column order.
    """
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out

    if not isinstance(df.columns, pd.MultiIndex):
        sub = df.copy()
        if len(tickers) == 1:
            out[tickers[0]] = _clean(sub)
        return out

    lvl0 = set(df.columns.get_level_values(0))
    ticker_level = 0 if not lvl0 & set(OHLCV) else 1
    available = list(dict.fromkeys(df.columns.get_level_values(ticker_level)))

    for tkr in available:
        try:
            sub = df.xs(tkr, axis=1, level=ticker_level)
        except KeyError:
            continue
        sub = _clean(sub)
        if not sub.empty:
            out[tkr] = sub
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c in OHLCV + ["Adj Close", "Stock Splits", "Dividends"]]
    sub = df[keep].copy()
    sub = sub.dropna(how="all")
    if "Close" in sub:
        sub = sub[sub["Close"].notna()]
    return sub


def download_batch(
    tickers: list[str],
    *,
    period: str | None = None,
    start=None,
    end=None,
    interval: str = "1d",
    actions: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download one batch of tickers and return per-ticker frames."""
    yf = _import_yf()
    kwargs = dict(
        interval=interval,
        auto_adjust=False,  # keep raw OHLC: gaps must be measured on traded prices
        actions=actions,
        group_by="ticker",
        threads=True,
        progress=False,
        prepost=False,
    )
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = yf.download(tickers, **kwargs)
        except Exception as exc:
            log.warning("yfinance batch failed (%d tickers): %s", len(tickers), exc)
            return {}
    return split_batch_frame(df, tickers)


def to_exchange_time(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Ensure an intraday frame is indexed in US market local time."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df = df.copy()
    df.index = idx.tz_convert(tz)
    return df
