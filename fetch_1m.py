"""Fetch 1-minute bars for the qualifying tickers over the full yfinance window.

The whole window is needed, not just event days: the spike baseline is the mean
1-minute candle volume over the 10 sessions BEFORE each event.
"""
import warnings, datetime as dt, time, pickle, logging, os
warnings.filterwarnings("ignore")
import pandas as pd, yfinance as yf
from Lucky.yf_utils import split_batch_frame, to_exchange_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
import sys
src = sys.argv[1] if len(sys.argv)>1 else "data/need_1m_tickers.csv"
need = pd.read_csv(src)
tickers = sorted(need.ticker.unique())
end = dt.date.today() + dt.timedelta(days=1)
start = dt.date.today() - dt.timedelta(days=29)
os.makedirs("cache/intraday_1m", exist_ok=True)
out = {}
for i, t in enumerate(tickers, 1):
    cp = f"cache/intraday_1m/{t}.pkl"
    if os.path.exists(cp):
        try:
            out[t] = pickle.load(open(cp, "rb")); continue
        except Exception:
            pass
    frames, cur = [], start
    while cur < end:                       # 1m is capped at 8 days per request
        stop = min(cur + dt.timedelta(days=7), end)
        for _ in range(3):
            d = yf.download([t], start=cur, end=stop, interval="1m", auto_adjust=False,
                            group_by="ticker", threads=False, progress=False, prepost=False)
            f = split_batch_frame(d, [t]).get(t, pd.DataFrame())
            time.sleep(0.75)
            if not f.empty:
                frames.append(f); break
        cur = stop
    if frames:
        raw = pd.concat(frames); raw = raw[~raw.index.duplicated()]
        raw = to_exchange_time(raw).between_time("09:30", "16:00", inclusive="left")
        out[t] = raw
        pickle.dump(raw, open(cp, "wb"))
    if i % 20 == 0:
        logging.info("  %d/%d tickers, %d with data", i, len(tickers), len(out))

logging.info("done: %d/%d tickers", len(out), len(tickers))
