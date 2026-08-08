"""Generates strategy_backtest.ipynb.

BQuant runs notebooks only — it cannot import a .py — so the libraries in
tracker_source.py and backtest_source.py are inlined into the notebook as cells
rather than imported. This script is the only place they are joined: edit the
source files or the runner cells below, re-run this, and commit the regenerated
notebook.

    python dev/build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "strategy_backtest.ipynb")

md = lambda s: dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(keepends=True))
code = lambda s: dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=s.strip("\n").splitlines(keepends=True))


def library_cell(filename, title, note):
    """A source file, wrapped as a notebook cell."""
    with open(os.path.join(HERE, filename)) as fh:
        src = fh.read()
    header = (
        "# =====================================================================\n"
        "# %s — generated, do not edit here\n"
        "# =====================================================================\n"
        "# Source of truth is dev/%s; this cell is produced by\n"
        "# dev/build_notebook.py. It lives inline because BQuant cannot import a\n"
        "# .py file — this notebook plus the dashboard notebook is all you need.\n"
        "# %s\n\n" % (title, filename, note))
    return code(header + src.strip("\n"))


cells = [
md("""
# Do these signals actually work?

A backtest of the dashboard's five signal engines over the past year. The dashboard says *what
looks like a trade now*; this asks whether those calls were right, on the whole population of
signals rather than a hand-picked few.

**Method.** Step through the year one week at a time. At each step, rewind the price and implied-vol
history to that date and re-run the dashboard's own `build_signals`. Every engine reads the last
row of each series, so a truncated frame reproduces exactly what the dashboard would have printed
that day — confidence scores included — with no look-ahead. Every signal on the board is recorded,
then marked forward over 5, 10 and 21 sessions.

**No sizing.** Every trade is one unit and the headline number is a hit rate. That keeps the result
about signal quality instead of about a position-sizing rule layered on top. Average P&L per trade
is reported alongside, in each engine's own unit (vol points or percent).

**The number that matters is the gap to baseline.** A hit rate alone proves nothing: "sell vol wins
78% of the time" is unimpressive if implied vol exceeds subsequent realized vol 76% of the time
anyway. So every trade is also scored as if it had been taken on *every* date in the sample,
regardless of whether the engine fired. That unconditional rate is the baseline, and the distance
between the two is the only thing that can fairly be called an edge.

**To run in BQuant:** upload this notebook next to `commodities_vol_rv_dashboard.ipynb` and run
every cell top to bottom. Nothing to install; nothing here writes to the dashboard. The Bloomberg
pull takes a couple of minutes and the replay about a minute.
"""),

library_cell("replay_source.py", "REPLAY LIBRARY",
             "Point-in-time replay, leg resolution and trade marking. Run once."),
library_cell("backtest_source.py", "BACKTEST LIBRARY",
             "Baselines, statistics, charts and the written summary. Run once."),

code("""
# =====================================================================
# SETUP — reuse the dashboard's engines without touching the dashboard
# =====================================================================
# Two ways this can run, both handled here:
#   * as its own notebook  -> find the dashboard .ipynb and exec its CONFIG /
#     ANALYTICS / DATA LAYER cells into NS. Its RENDERERS and CONTROLS cells are
#     skipped, so no widgets are built and no Bloomberg pull is triggered.
#   * pasted into the dashboard notebook -> the engines are already defined, so
#     NS is just this notebook's own namespace.
# Either way the signals being tested are the dashboard's own code, not a copy.
from IPython.display import HTML, display

if "build_signals" in globals():
    NS = globals()
    print("Using the engines already defined in this notebook.")
else:
    DASHBOARD = find_dashboard()
    NS = load_dashboard(DASHBOARD)
    print("Loaded %s (cells %s)." % (os.path.basename(DASHBOARD), NS["__loaded_cells__"]))

print("%d assets, %d with implied vol." % (len(NS["ALL_TICKERS"]), len(NS["IV_TICKERS"])))
"""),

code("""
# =====================================================================
# DATA — one Bloomberg pull, 3y of OHLC + implied vol
# =====================================================================
# Same fetch_all the dashboard uses. Takes a couple of minutes; the result is
# cached in DATA so everything below can be re-run without pulling again.
px, iv, px_fail, iv_fail = NS["fetch_all"]()
DATA = dict(px=px, iv=iv)

print("prices: %d assets to %s" % (px["close"].shape[1], px["close"].index[-1].date()))
print("implied vol: %d assets" % iv.shape[1])
if px_fail:
    print("no price:", ", ".join(NS["NAME"].get(t, t) for t in px_fail))
if iv_fail:
    print("no IV:", ", ".join(NS["NAME"].get(t, t) for t in iv_fail))
"""),

code("""
# =====================================================================
# SETTINGS
# =====================================================================
close = DATA["px"]["close"]

WEEKS    = 52                 # replays, walking back from the most recent data
STEP     = 5                  # sessions between replays (5 = weekly)
HORIZONS = (5, 10, 21)        # holding periods marked for every signal
PRIMARY  = 5                  # the horizon the report headlines

# At STEP=5 the 5-session holds are consecutive and non-overlapping. Longer
# horizons overlap each other by construction — they are shown to test whether
# an edge survives a longer hold, not as extra independent evidence.

# The dashboard's default control panel. Change these to backtest a different
# configuration — lookback=504 for 2y percentiles, min_conf=55 to test only the
# ideas the panel would have shown above 55% confidence, and so on.
CFG = dict(lookback=252, lb_name="1y",
           rv_estimator="Yang-Zhang (OHLC)", rv_window=21,
           iv_lo=10, iv_hi=90, disp_z=2.0, corr_z=1.0,
           corr_window=NS["CORR_WINDOW"], pair_win=NS["PAIR_WIN"],
           ll_window=NS["LEADLAG_WINDOW"], ll_r=0.30, ll_gap=1.5,
           min_quality="Fair", exclude_stale=True, min_conf=0)

TR = Tracker(NS, horizon=PRIMARY)
dates = replay_dates(close, weeks=WEEKS, step=STEP)
print("%d replays from %s to %s, holding %s sessions."
      % (len(dates), dates[0].date(), dates[-1].date(), "/".join(str(h) for h in HORIZONS)))
"""),

code("""
# =====================================================================
# RUN — replay the year (about a minute)
# =====================================================================
def _progress(n, total, asof, k):
    if n == 1 or n % 10 == 0 or n == total:
        print("  %3d/%d   %s   %d signals" % (n, total, asof.date(), k))

MARKS = run_backtest(TR, DATA["px"], DATA["iv"], CFG, weeks=WEEKS, step=STEP,
                     horizons=HORIZONS, progress=_progress)

print("\\n%d marked trades across %d replay dates and %d horizons."
      % (len(MARKS), MARKS["entry_date"].nunique(), MARKS["horizon"].nunique()))
"""),

code("""
# =====================================================================
# RESULTS — headline, takeaways, charts
# =====================================================================
display(HTML(report_html(MARKS, PRIMARY, CFG, theme=TR.theme)))
display(fig_calibration(MARKS, PRIMARY, theme=TR.theme))
display(fig_engines(MARKS, PRIMARY, theme=TR.theme))
display(fig_equity(MARKS, PRIMARY, theme=TR.theme))
display(fig_horizons(MARKS, theme=TR.theme))
display(HTML(table_html(MARKS, PRIMARY, theme=TR.theme)))
"""),

code("""
# =====================================================================
# EXPORT — one HTML file for the slides, one CSV for the numbers
# =====================================================================
# backtest_report.html is self-contained: open it, screenshot the charts you
# want. backtest_marks.csv is every marked trade, so any number in the deck can
# be traced back to the trades behind it.
figs = [fig_calibration(MARKS, PRIMARY, theme=TR.theme),
        fig_engines(MARKS, PRIMARY, theme=TR.theme),
        fig_equity(MARKS, PRIMARY, theme=TR.theme),
        fig_horizons(MARKS, theme=TR.theme)]

parts = [report_html(MARKS, PRIMARY, CFG, theme=TR.theme)]
for i, f in enumerate(figs):
    parts.append(f.to_html(full_html=False, include_plotlyjs=(True if i == 0 else False)))
parts.append("<div style='padding:14px 0'>%s</div>" % table_html(MARKS, PRIMARY, theme=TR.theme))

with open("backtest_report.html", "w") as fh:
    fh.write("<html><body style='margin:0;background:#0B0E14;padding:16px'>%s</body></html>"
             % "".join(parts))

out = MARKS.drop(columns=[c for c in ("legs", "sectors") if c in MARKS.columns])
out.to_csv("backtest_marks.csv", index=False)
print("wrote backtest_report.html and backtest_marks.csv (%d rows)" % len(out))

# The per-engine table as plain text, for pasting into slides or notes.
print()
print(table(MARKS, "engine", horizon=PRIMARY)[
    ["engine", "n", "hit", "baseline", "lift", "p", "pnl", "unit"]].to_string(index=False))
"""),

md("""
---
### How each trade is marked

Entry close to exit close, no costs, equal notional on both legs of a pair, one unit per trade.

| Engine | Wins when | P&L |
|---|---|---|
| IV mean-reversion | IV rises after BUY VOL, falls after SELL VOL | ±(IV_exit − IV_entry), vol pts |
| Variance risk premium | realized vol over the hold comes in under the implied quoted at entry (SELL VOL) | ±(IV_entry − RV_realized), vol pts |
| Vol dispersion | the rich/cheap IV spread converges | ΔIV_cheap − ΔIV_rich, vol pts |
| Correlation RV | the laggard closes the gap on the outperformer | ret_long − ret_short, % |
| Lead-lag catch-up | the follower moves the leader's way | ±ret_follower, % |

Variance-risk-premium trades are marked against **delivered** volatility rather than the IV
re-mark, because that is what an option seller is actually paid on.

### Reading the charts

- **Calibration.** Bars are realized hit rates by confidence bucket, with 95% Wilson intervals.
  Diamonds are the baseline for those same trades. A bucket only demonstrates skill to the extent
  its bar clears its diamond — a tall bar sitting on a tall diamond is a bet that wins anyway.
- **Engines.** Same idea per engine, with an exact binomial p-value against that engine's own
  baseline. Wide intervals mean too few trades to tell, and are drawn rather than hidden.
- **Cumulative wins − losses.** Unit-free, so engines marked in vol points and in percent share an
  axis. A straight climb is a persistent edge; a single step is one lucky month.
- **Horizons.** Edge over baseline at 5, 10 and 21 sessions. A real signal should not need a
  specific holding period to work.

### What this is not

- **Not a P&L.** No sizing, costs, slippage, margin or option greeks. A vol-point move is not a
  dollar, and a 55% hit rate on unsized bets does not mean 55% of anything is profit.
- **Not independent draws.** A setup that stays extreme for a month is re-recorded every week, so
  the trade count overstates the sample. Horizons longer than the replay step overlap outright.
- **One year, one universe, one parameter set.** The confidence score is itself fitted on each
  asset's own history, so a good calibration line says the score is internally consistent, not
  that it will hold on new data.

### Where the code lives

The two library cells are generated from `dev/tracker_source.py` and `dev/backtest_source.py` by
`python dev/build_notebook.py` — edit there and regenerate rather than editing the cells, or the
next rebuild overwrites your changes. `python dev/test_tracker.py` runs the whole path against a
generated market with no Bloomberg needed. Nothing in `dev/` needs to be uploaded to BQuant.
"""),
]

nb = dict(cells=cells, metadata=dict(
    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python", version="3.11")),
    nbformat=4, nbformat_minor=5)

with open(OUT, "w") as fh:
    json.dump(nb, fh, indent=1)
    fh.write("\n")
print("wrote %s — %d cells" % (OUT, len(cells)))
