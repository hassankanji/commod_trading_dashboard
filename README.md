# Commodities vol & relative-value dashboard

Two notebooks. **Upload both to BQuant, run either one.** There is nothing to install and no
`.py` file to import — BQuant runs notebooks only, so everything they need is inside them.

| File | What it is |
|---|---|
| `commodities_vol_rv_dashboard.ipynb` | The dashboard. Five signal engines over a commodity universe, with the control panel and charts. |
| `trade_outcome_tracker.ipynb` | The outcome tracker. Replays what the dashboard suggested a week ago and marks how those trades did. Needs the dashboard notebook in the same folder — it reads the engines out of it. |

Run the dashboard on its own whenever you want current signals. Run the tracker when you want to
know whether last week's signals worked; it does its own Bloomberg pull and does not touch the
dashboard or require it to have been run.

## dev/ — not needed in BQuant

Tooling for maintaining the repo. None of it is uploaded or run inside BQuant.

| File | What it is |
|---|---|
| `dev/tracker_source.py` | Source of truth for the tracker library that is inlined into the tracker notebook. |
| `dev/build_notebook.py` | Regenerates `trade_outcome_tracker.ipynb` from the source above plus the runner cells. |
| `dev/test_tracker.py` | Self-test. Runs the whole tracker against a generated market with no Bloomberg needed. |

To change the tracker's logic, edit `dev/tracker_source.py`, then:

```
python dev/build_notebook.py     # rebuild the notebook
python dev/test_tracker.py       # verify it, including the notebook's own cells
```

Editing the library cell inside the notebook directly works for a quick experiment, but the next
rebuild overwrites it — and the test fails if the two have drifted apart.
