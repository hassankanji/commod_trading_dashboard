# Commodities vol & relative-value dashboard

Two notebooks. **Upload both to BQuant, run either one.** There is nothing to install and no
`.py` file to import — BQuant runs notebooks only, so everything they need is inside them.

| File | What it is |
|---|---|
| `commodities_vol_rv_dashboard.ipynb` | The dashboard. Five signal engines over a commodity universe, with the control panel and charts. |
| `strategy_backtest.ipynb` | The evidence. Replays a year of history, asks the dashboard what it would have said each week, and measures whether those calls were right. Needs the dashboard notebook in the same folder — it reads the engines out of it rather than copying them. |

## What the backtest does

Steps back through the year one week at a time. At each step it rewinds the price and implied-vol
history to that date and re-runs the dashboard's own `build_signals`, so the signals being scored
are the ones the dashboard would actually have printed that day — with no look-ahead. Every signal
is recorded, not just the good ones, and each is marked forward over 5, 10 and 21 sessions.

Every trade is one unit; there is no sizing. The headline number is a hit rate, and next to it a
**baseline**: the same trade, same asset, same horizon, taken on every date in the sample whether
or not the engine fired. The gap between the two is the only thing that counts as an edge — "sell
vol wins 61% of the time" means nothing if that bet wins 54% of the time regardless.

Outputs: five summary cards, written key takeaways, four charts (confidence calibration, per-engine
edge with confidence intervals, cumulative wins through the year, edge by holding period), and a
per-engine table. The last cell writes `backtest_report.html` and `backtest_marks.csv` so the
charts can go straight into slides and any number can be traced back to the trades behind it.

Runtime: a couple of minutes for the Bloomberg pull, about a minute for 52 replays.

## dev/ — not needed in BQuant

Tooling for maintaining the repo. None of it is uploaded or run inside BQuant.

| File | What it is |
|---|---|
| `dev/replay_source.py` | Point-in-time replay: loading the dashboard's engines, resolving each signal's legs, marking trades forward. |
| `dev/backtest_source.py` | Baselines, Wilson intervals, exact binomial tests, charts, and the written summary. |
| `dev/build_notebook.py` | Inlines both sources into `strategy_backtest.ipynb` as library cells. |
| `dev/test_backtest.py` | Self-test. Runs the whole path against a generated market, no Bloomberg needed. |

To change the analysis, edit the source files, then:

```
python dev/build_notebook.py     # rebuild the notebook
python dev/test_backtest.py      # verify it, including the notebook's own cells
```

The test recomputes every P&L and baseline straight from the raw frames, checks the statistics
against hand-worked values, and executes the shipped notebook from a folder containing nothing but
the dashboard — so a stale library cell or a hidden `.py` dependency fails rather than passing
quietly. Editing a library cell inside the notebook works for a quick experiment, but the next
rebuild overwrites it and the test flags the drift.
