"""Generates trade_outcome_tracker.ipynb.

BQuant runs notebooks only — it cannot import a .py — so the tracker library in
tracker_source.py is inlined into the notebook as a cell rather than imported.
This script is the only place the two are joined: edit tracker_source.py or the
cells below, re-run this, and commit the regenerated notebook.

    python dev/build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "trade_outcome_tracker.ipynb")

md = lambda s: dict(cell_type="markdown", metadata={}, source=s.strip("\n").splitlines(keepends=True))
code = lambda s: dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=s.strip("\n").splitlines(keepends=True))


def library_cell():
    """tracker_source.py, wrapped as a notebook cell."""
    with open(os.path.join(HERE, "tracker_source.py")) as fh:
        src = fh.read()
    header = (
        "# =====================================================================\n"
        "# TRACKER LIBRARY — generated, do not edit here\n"
        "# =====================================================================\n"
        "# Source of truth is dev/tracker_source.py; this cell is produced by\n"
        "# dev/build_notebook.py. It lives inline because BQuant cannot import a\n"
        "# .py file — this notebook plus the dashboard notebook is all you need.\n"
        "# Run it once, then carry on to the next cell.\n\n")
    return code(header + src.strip("\n"))


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
2. Take the **5 highest-confidence** signals, then top up so **every engine that fired gets at
   least one trade** (labelled `ENGINE`) and at least **3 trades touch metals or energy**
   (labelled `SECTOR`) — even when nothing in those buckets scored well. Without the engine
   pass, one engine's good week crowds the others out of the results entirely.
3. Mark each trade forward over the next `HORIZON` sessions, entry close to exit close, using the
   payoff that matches what the engine was betting on (see the notes at the bottom).

**To run in BQuant:** upload this notebook next to `commodities_vol_rv_dashboard.ipynb` and run
every cell top to bottom. There is nothing else to install — no `.py` files, no imports beyond
what the dashboard already uses. Nothing here writes to the dashboard.

If you would rather not keep two notebooks, paste cells 1–7 of this one onto the end of the
dashboard notebook instead; the setup cell detects that the engines are already defined and
skips loading them from disk.
"""),

library_cell(),

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
# Either way the signals scored below come from the dashboard's own code.
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
ENGINE_N = 1                      # ...topped up to this many per engine, so each one is judged
SECTORS  = ("Energy", "Precious Metals", "Base Metals")
SECTOR_N = 3                      # ...and to at least this many metals/energy trades

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
TR = Tracker(NS, horizon=HORIZON)

SIGNALS = TR.signals_asof(DATA["px"], DATA["iv"], CFG, AS_OF)
print("%d signals were on the board on %s:" % (len(SIGNALS), AS_OF.date()))
if len(SIGNALS):
    print(SIGNALS.groupby("engine").size().to_string())

RESULTS, SUMMARY = TR.run(DATA["px"], DATA["iv"], CFG, AS_OF, horizon=HORIZON,
                          top_n=TOP_N, engine_n=ENGINE_N, sectors=SECTORS, sector_n=SECTOR_N)
display(HTML(report_html(RESULTS, SUMMARY, CFG, theme=TR.theme)))
"""),

code("""
# =====================================================================
# SAVE — plain-text report + CSV journal
# =====================================================================
# run_<date>.csv is this replay; ledger.csv accumulates across replays so the
# hit rate builds a sample over time. Re-running a date replaces its rows.
print(report_text(RESULTS, SUMMARY, CFG))
path = save_run(RESULTS)
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
                  top_n=TOP_N, engine_n=ENGINE_N, sectors=SECTORS, sector_n=SECTOR_N)
    if r is None or r.empty:
        continue
    frames.append(r)
    save_run(r)
    print("%s  %2d trades  hit %s" % (asof.date(), s["scored"],
          "%.0f%%" % s["hit"] if s["scored"] else "—"))

if frames:
    POOLED = pd.concat(frames, ignore_index=True, sort=False)
    PSUM = summarize(POOLED, sectors=SECTORS)
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
- Every engine that fired is guaranteed a trade, so the by-engine table is never empty for an
  engine that had something to say. An engine with **no signals at all** that day is listed
  separately as *no signals on this date* — a blank week is not a bad week.
- Trades carrying an `ENGINE` or `SECTOR` badge got in on a quota, not on merit. They drag the
  headline hit rate toward 50% by design; the *by how it was picked* table separates them from
  the top-confidence picks so you can see both numbers.
- One week is roughly five to eight trades. That is a log entry, not evidence: run the optional
  multi-week cell above before drawing any conclusion about an engine.
- Trades whose legs cannot be resolved, or whose IV is missing at either end, are reported as
  **N.A.** rather than silently dropped or marked at a guessed level.
- `trade_journal/ledger.csv` is the running record. It carries the full reason text and trigger
  for every tracked trade, so a stale entry can always be traced back to what the dashboard said.

### Where the code lives

The **TRACKER LIBRARY** cell above is the whole implementation, inlined so that this notebook and
the dashboard notebook are the only two files BQuant needs. It is generated from
`dev/tracker_source.py` in the repo by `python dev/build_notebook.py` — edit it there and
regenerate rather than editing the cell, or the next rebuild will overwrite your changes. The
`dev/` folder is developer tooling only; nothing in it has to be uploaded to BQuant.

`python dev/test_tracker.py` runs the whole replay → select → score → report path against a
generated market with no Bloomberg needed, checks the P&L signs by recomputing them
independently, asserts a replay cannot see data after its as-of date, and executes this
notebook's own cells so the inlined library cannot drift from its source.
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
