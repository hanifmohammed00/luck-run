"""Bootstrap and Monte Carlo robustness checks on the validated strategy.

Bootstrap: resample the trade sample with replacement to get a confidence
interval on the average return. Answers "how much of the headline number
could be luck, given only this sample."

Monte Carlo: resample the trade sequence (with replacement, at a chosen
position size) to build many possible equity curves and drawdown outcomes.
Answers "how bad could this realistically get," not "is the edge real" --
same underlying sample, so it can't manufacture more signal than the
backtest already has.

    .venv/bin/python -m Lucky.robustness
    .venv/bin/python -m Lucky.robustness --size 3000 --horizon 100 --breach 3000
"""
import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gap_vwap_strategy as S


def load_trades(events_path: str = "data/events_1_3.csv",
                 bars_path: str = "data/bars_1m_all.pkl",
                 params: S.Params = S.DEFAULT) -> pd.DataFrame:
    bars = pickle.load(open(bars_path, "rb"))
    baselines = S.candle_baselines(bars)
    events = pd.read_csv(events_path, parse_dates=["date"])
    return S.backtest(events, bars, baselines, params)


def bootstrap(returns: np.ndarray, n_sims: int = 20_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(returns)
    means = rng.choice(returns, size=(n_sims, n), replace=True).mean(axis=1)
    return {
        "n_trades": n,
        "n_sims": n_sims,
        "mean_pct": float(returns.mean()),
        "ci_low_pct": float(np.percentile(means, 2.5)),
        "ci_high_pct": float(np.percentile(means, 97.5)),
        "p_mean_nonpositive": float((means <= 0).mean()),
        "means": means,
    }


def monte_carlo(returns: np.ndarray, size_dollars: float, horizon: int,
                 breach_dollars: float, n_sims: int = 20_000, seed: int = 1) -> dict:
    """Resample `horizon` trades per path, `n_sims` paths, at `size_dollars`/trade."""
    rng = np.random.default_rng(seed)
    sample_pct = rng.choice(returns, size=(n_sims, horizon), replace=True)
    pnl = sample_pct * size_dollars / 100.0
    equity = np.cumsum(pnl, axis=1)
    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdown = running_peak - equity
    max_dd = drawdown.max(axis=1)
    final = equity[:, -1]
    return {
        "size_dollars": size_dollars,
        "horizon": horizon,
        "n_sims": n_sims,
        "median_final": float(np.median(final)),
        "p5_final": float(np.percentile(final, 5)),
        "p95_final": float(np.percentile(final, 95)),
        "median_max_dd": float(np.median(max_dd)),
        "p95_max_dd": float(np.percentile(max_dd, 95)),
        "p_breach": float((max_dd >= breach_dollars).mean()),
        "equity": equity,
        "max_dd": max_dd,
        "final": final,
    }


def plot(bs: dict, mc: dict, breach_dollars: float, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    n_show = min(300, mc["equity"].shape[0])
    idx = np.random.default_rng(2).choice(mc["equity"].shape[0], n_show, replace=False)
    x = np.arange(1, mc["horizon"] + 1)
    for i in idx:
        ax.plot(x, mc["equity"][i], color="steelblue", alpha=0.05, linewidth=0.8)
    p5 = np.percentile(mc["equity"], 5, axis=0)
    p50 = np.percentile(mc["equity"], 50, axis=0)
    p95 = np.percentile(mc["equity"], 95, axis=0)
    ax.plot(x, p50, color="black", linewidth=1.5, label="median")
    ax.plot(x, p5, color="firebrick", linewidth=1, linestyle="--", label="5th / 95th pct")
    ax.plot(x, p95, color="firebrick", linewidth=1, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(f"{mc['n_sims']:,} simulated equity paths\n(${mc['size_dollars']:,.0f}/trade, {mc['horizon']} trades)")
    ax.set_xlabel("trade #")
    ax.set_ylabel("cumulative P&L ($)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.hist(mc["max_dd"], bins=60, color="steelblue", alpha=0.8)
    ax.axvline(breach_dollars, color="firebrick", linewidth=1.5, linestyle="--",
               label=f"breach line ${breach_dollars:,.0f}")
    ax.set_title(f"max drawdown per path\nP(breach) = {mc['p_breach']*100:.1f}%")
    ax.set_xlabel("max drawdown ($)")
    ax.set_ylabel("simulations")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(bs["means"], bins=60, color="seagreen", alpha=0.8)
    ax.axvline(bs["mean_pct"], color="black", linewidth=1.5, label="sample mean")
    ax.axvline(0, color="firebrick", linewidth=1, linestyle="--", label="breakeven")
    ax.set_title(f"bootstrap: mean return/trade\n95% CI [{bs['ci_low_pct']:.2f}%, {bs['ci_high_pct']:.2f}%], n={bs['n_trades']} trades")
    ax.set_xlabel("mean return per trade (%)")
    ax.set_ylabel("bootstrap resamples")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="data/events_1_3.csv")
    ap.add_argument("--bars", default="data/bars_1m_all.pkl")
    ap.add_argument("--size", type=float, default=2500.0, help="$ per trade for the Monte Carlo")
    ap.add_argument("--horizon", type=int, default=200, help="trades per simulated path")
    ap.add_argument("--breach", type=float, default=3000.0, help="$ drawdown considered a breach")
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--out", default="data/robustness.png")
    args = ap.parse_args()

    trades = load_trades(args.events, args.bars)
    returns = trades.return_pct.to_numpy(float)
    print(f"loaded {len(returns)} trades, avg {returns.mean():.2f}%, win rate {(returns > 0).mean()*100:.1f}%")

    bs = bootstrap(returns, n_sims=args.sims)
    print(f"\nbootstrap ({bs['n_sims']:,} resamples of {bs['n_trades']} trades):")
    print(f"  mean return/trade: {bs['mean_pct']:.2f}%")
    print(f"  95% CI: [{bs['ci_low_pct']:.2f}%, {bs['ci_high_pct']:.2f}%]")
    print(f"  P(true mean <= 0): {bs['p_mean_nonpositive']*100:.1f}%")

    mc = monte_carlo(returns, args.size, args.horizon, args.breach, n_sims=args.sims)
    print(f"\nmonte carlo ({mc['n_sims']:,} paths, ${mc['size_dollars']:,.0f}/trade, {mc['horizon']} trades/path):")
    print(f"  final P&L: median ${mc['median_final']:,.0f}, "
          f"5th-95th pct [${mc['p5_final']:,.0f}, ${mc['p95_final']:,.0f}]")
    print(f"  max drawdown: median ${mc['median_max_dd']:,.0f}, 95th pct ${mc['p95_max_dd']:,.0f}")
    print(f"  P(breach ${args.breach:,.0f} drawdown): {mc['p_breach']*100:.1f}%")

    plot(bs, mc, args.breach, args.out)
    print(f"\nplot saved to {args.out}")


if __name__ == "__main__":
    main()
