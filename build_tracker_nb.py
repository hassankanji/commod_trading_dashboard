"""Generates trade_outcome_tracker.ipynb. Edit here, re-run, commit both."""
import json

md = lambda s: dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(keepends=True))
code = lambda s: dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=s.strip("\n").splitlines(keepends=True))

cells = [
md("""
# Trade-signal outcome tracker

Separate from the dashboard on purpose. The dashboard says *what looks like a trade now*;
this says *did the trades it flagged actually work*. It does not import, modify or depend on
the dashboard's UI — it re-runs the dashboard's own signal engines on truncated data.

**How it works**

1. Rewind the price/IV history to a past date `AS_OF` and re-run `build_signals`. Every engine
   reads the last row of each series, so a truncated frame reproduces exactly what the dashboard
   printed that day — confidence hit-rates included, with no look-ahead.
2. Take the **5 highest-confidence** signals, then top up so at least **3 touch metals or energy**
   even when nothing there scored well. Quota picks are labelled `QUOTA` in the report.
3. Mark each trade forward over the next `HORIZON` sessions, entry close to exit close, using the
   payoff that matches what the engine was betting on (see the notes at the bottom).

**To run:** open in BQuant with `commodities_vol_rv_dashboard.ipynb` and `trade_tracker.py` in the
same folder, then run every cell top to bottom. Nothing here writes to the dashboard.
"""),

code("""
# =====================================================================
# SETUP — reuse the dashboard's engines without touching the dashboard
# =====================================================================
# Execs only its CONFIG / ANALYTICS / DATA LAYER cells. The RENDERERS and
# CONTROLS cells are skipped, so no widgets are built and no data is pulled
# on import — the signals scored below are the same code the dashboard runs.
import importlib, os, sys
import numpy as np
import pandas as pd
from IPython.display import HTML, display

HERE = os.getcwd()
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import trade_tracker as tt
importlib.reload(tt)

DASHBOARD = os.path.join(HERE, "commodities_vol_rv_dashboard.ipynb")
NS = tt.load_dashboard(DASHBOARD)
print("Loaded dashboard cells %s — %d assets, %d with implied vol."
      % (NS["__loaded_cells__"], len(NS["ALL_TICKERS"]), len(NS["IV_TICKERS"])))
"""),

code("""
# =====================================================================
# DATA — one Bloomberg pull, 3y of OHLC + implied vol
# =====================================================================
# Same fetch_all the dashboard uses. Takes a couple of minutes; the result is
# cached in DATA so the cells below can be re-run freely.
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
# SETTINGS — what to replay, and with which dashboard settings
# =====================================================================
close = DATA["px"]["close"]

HORIZON  = 5                      # trading sessions held (one week)
AS_OF    = close.index[-(HORIZON + 1)]   # replay date = one week back on the data's own calendar
TOP_N    = 5                      # highest-confidence signals to track
SECTORS  = ("Energy", "Precious Metals", "Base Metals")
SECTOR_N = 3                      # ...topped up to at least this many metals/energy trades

# The dashboard's default control panel. Change these to score a different
# configuration — e.g. lookback=504 for 2y percentiles, or min_conf=55 to
# replay only the ideas the panel would have shown above 55% confidence.
CFG = dict(lookback=252, lb_name="1y",
           rv_estimator="Yang-Zhang (OHLC)", rv_window=21,
           iv_lo=10, iv_hi=90, disp_z=2.0, corr_z=1.0,
           corr_window=NS["CORR_WINDOW"], pair_win=NS["PAIR_WIN"],
           ll_window=NS["LEADLAG_WINDOW"], ll_r=0.30, ll_gap=1.5,
           min_quality="Fair", exclude_stale=True, min_conf=0)

print("replaying %s, marking to %s" % (AS_OF.date(), close.index[-1].date()))
"""),

code("""
# =====================================================================
# RUN — replay, pick, score
# =====================================================================
TR = tt.Tracker(NS, horizon=HORIZON)

SIGNALS = TR.signals_asof(DATA["px"], DATA["iv"], CFG, AS_OF)
print("%d signals were on the board on %s:" % (len(SIGNALS), AS_OF.date()))
if len(SIGNALS):
    print(SIGNALS.groupby("engine").size().to_string())

RESULTS, SUMMARY = TR.run(DATA["px"], DATA["iv"], CFG, AS_OF, horizon=HORIZON,
                          top_n=TOP_N, sectors=SECTORS, sector_n=SECTOR_N)
display(HTML(tt.report_html(RESULTS, SUMMARY, CFG, theme=TR.theme)))
"""),

code("""
# =====================================================================
# SAVE — plain-text report + CSV journal
# =====================================================================
# run_<date>.csv is this replay; ledger.csv accumulates across replays so the
# hit rate builds a sample over time. Re-running a date replaces its rows.
print(tt.report_text(RESULTS, SUMMARY, CFG))
path = tt.save_run(RESULTS)
print("\\nsaved -> %s" % path)
"""),

code("""
# =====================================================================
# OPTIONAL — walk several weeks back to build a real sample
# =====================================================================
# One week is five-ish trades: far too few to judge an engine. This replays the
# last WEEKS non-overlapping weeks and pools them. Each replay only ever sees
# data up to its own as-of date, so the pooled hit rate is still out-of-sample.
WEEKS = 8

frames = []
for k in range(1, WEEKS + 1):
    i = len(close) - 1 - k * HORIZON
    if i < 400:                                   # need history for percentiles
        break
    asof = close.index[i]
    r, s = TR.run(DATA["px"], DATA["iv"], CFG, asof, horizon=HORIZON,
                  top_n=TOP_N, sectors=SECTORS, sector_n=SECTOR_N)
    if r is None or r.empty:
        continue
    frames.append(r)
    tt.save_run(r)
    print("%s  %2d trades  hit %s" % (asof.date(), s["scored"],
          "%.0f%%" % s["hit"] if s["scored"] else "—"))

if frames:
    POOLED = pd.concat(frames, ignore_index=True, sort=False)
    PSUM = tt.summarize(POOLED, sectors=SECTORS)
    print("\\npooled over %d weeks: %d trades, hit rate %.0f%%, avg confidence %.0f%% "
          "(calibration %+.0f pts)"
          % (len(frames), PSUM["scored"], PSUM["hit"], PSUM["avg_conf"], PSUM["calibration"]))
    for e, b in PSUM["by_engine"].items():
        print("   %-24s %d/%d  %s" % (e, b["wins"], b["n"],
              "%.0f%%" % b["hit"] if b["n"] else "—"))
"""),

md("""
---
### How each trade is marked

All marks are entry close to exit close, no costs, no sizing, equal notional on both legs of a
pair. This measures whether the signal pointed the right way, not a tradable P&L.

| Engine | Wins when | P&L |
|---|---|---|
| IV mean-reversion | IV rises after BUY VOL, falls after SELL VOL | ±(IV_exit − IV_entry), vol pts |
| Variance risk premium | realized vol over the week comes in under the implied quoted at entry (SELL VOL) | ±(IV_entry − RV_realized), vol pts |
| Vol dispersion | the rich/cheap IV spread converges | ΔIV_cheap − ΔIV_rich, vol pts |
| Correlation RV | the laggard closes the gap on the outperformer | ret_long − ret_short, % |
| Lead-lag catch-up | the follower moves the leader's way | ±ret_follower, % |

Variance-risk-premium trades are marked against **delivered** volatility rather than the IV
re-mark, because that is what an option seller is actually paid on. It is the one engine whose
outcome is not simply "did the quote move".

### Reading the numbers

- **Calibration** is realized hit rate minus the average confidence the engines predicted at entry.
  Positive means the confidence scores were, if anything, conservative that week.
- Vol trades are in vol points and price trades in percent, so the two average-P&L figures are
  not additive and should not be summed into a portfolio return.
- One week is roughly five to eight trades. That is a log entry, not evidence: run the optional
  multi-week cell above before drawing any conclusion about an engine.
- Trades whose legs cannot be resolved, or whose IV is missing at either end, are reported as
  **N.A.** rather than silently dropped or marked at a guessed level.
- `trade_journal/ledger.csv` is the running record. It carries the full reason text and trigger
  for every tracked trade, so a stale entry can always be traced back to what the dashboard said.

### Verifying it without Bloomberg

`python test_trade_tracker.py` runs the whole replay → select → score → report path against a
generated market, checks the P&L signs by recomputing them independently, and asserts that a
replay cannot see data after its as-of date.
"""),
]

nb = dict(cells=cells, metadata=dict(
    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python", version="3.11")),
    nbformat=4, nbformat_minor=5)

with open("trade_outcome_tracker.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1)
    fh.write("\n")
print("wrote trade_outcome_tracker.ipynb — %d cells" % len(cells))
