"""Self-test for the outcome tracker — runs anywhere, no Bloomberg needed.

Stubs the BQuant-only imports, execs the dashboard's real CONFIG + ANALYTICS
cells, and pushes a generated market through the whole
replay -> select -> score -> report path. Also executes the tracker notebook's
own code cells against that market, so the notebook cannot drift away from the
module without this failing.

    python test_trade_tracker.py
"""
import json
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DASHBOARD = os.path.join(HERE, "commodities_vol_rv_dashboard.ipynb")
TRACKER_NB = os.path.join(HERE, "trade_outcome_tracker.ipynb")


# ---------------------------------------------------------------------------
def stub_bquant():
    """Fake the modules that only exist inside BQuant."""
    bql = types.ModuleType("bql")

    class _Service:
        def execute(self, *a, **k):
            raise RuntimeError("no Bloomberg outside BQuant")

    bql.Service = _Service
    bql.Request = lambda *a, **k: None
    bql.func = types.SimpleNamespace(range=lambda *a, **k: None)
    bql.data = types.SimpleNamespace()
    sys.modules["bql"] = bql

    for m in ["plotly", "plotly.graph_objects", "plotly.subplots", "ipywidgets"]:
        mod = types.ModuleType(m)
        mod.__getattr__ = lambda n: types.SimpleNamespace()
        sys.modules[m] = mod
    sys.modules["plotly"].graph_objects = sys.modules["plotly.graph_objects"]
    sys.modules["plotly"].subplots = sys.modules["plotly.subplots"]

    disp = types.ModuleType("IPython.display")
    disp.HTML = lambda s: s
    disp.display = lambda *a, **k: None
    ipy = types.ModuleType("IPython")
    ipy.display = disp
    sys.modules.setdefault("IPython", ipy)
    sys.modules.setdefault("IPython.display", disp)


def synthetic_market(ns, seed=7, start="2023-08-07", end="2026-08-06"):
    """Generated OHLC + implied vol over the dashboard's real universe."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T = len(idx)
    cls_of = ns["ASSET_CLASS"]

    # One common factor per asset class plus idiosyncratic noise, so the
    # correlation and lead-lag engines have a real relationship to find.
    fac = {c: rng.normal(0, 0.009, T) for c in sorted(set(cls_of.values()))}
    close = {}
    for i, tk in enumerate(ns["ALL_TICKERS"]):
        vol = 0.008 + 0.012 * rng.random()
        r = 0.75 * fac[cls_of[tk]] + rng.normal(0, vol, T)
        if i % 4 == 0:                          # some names lag their factor by a day
            r[1:] = 0.5 * r[1:] + 0.5 * np.roll(fac[cls_of[tk]], 1)[1:]
        close[tk] = 50.0 * np.exp(np.cumsum(r))
    close = pd.DataFrame(close, index=idx)

    hi = close * (1 + np.abs(rng.normal(0, 0.006, close.shape)))
    lo = close * (1 - np.abs(rng.normal(0, 0.006, close.shape)))
    op = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, close.shape))
    px = {"open": op, "high": hi, "low": lo, "close": close}

    # IV mean-reverts around an asset-specific level and re-marks on ~75% of
    # days so vol_quality reads Good; a third is pushed to the top of its range
    # and a third to the bottom so engines 1-3 actually fire on the replay date.
    iv = {}
    for i, tk in enumerate(ns["IV_TICKERS"]):
        lvl = 18 + 14 * rng.random()
        s = np.zeros(T)
        s[0] = lvl
        for t in range(1, T):
            s[t] = s[t - 1] + 0.05 * (lvl - s[t - 1]) + rng.normal(0, 0.6)
        if i % 3 == 0:
            s[-60:] += np.linspace(0, 9, 60)
        elif i % 3 == 1:
            s[-60:] -= np.linspace(0, 7, 60)
        ser = pd.Series(np.maximum(s, 4.0), index=idx)
        ser[rng.random(T) < 0.25] = np.nan
        iv[tk] = ser.ffill()
    return px, pd.DataFrame(iv)


def default_cfg(ns):
    """The dashboard's control panel at its default settings."""
    return dict(lookback=252, lb_name="1y", rv_estimator="Yang-Zhang (OHLC)", rv_window=21,
                iv_lo=10, iv_hi=90, disp_z=2.0, corr_z=1.0,
                corr_window=ns["CORR_WINDOW"], pair_win=ns["PAIR_WIN"],
                ll_window=ns["LEADLAG_WINDOW"], ll_r=0.30, ll_gap=1.5,
                min_quality="Fair", exclude_stale=True, min_conf=0)


# ---------------------------------------------------------------------------
def check_tracker(tt, ns):
    px, iv = synthetic_market(ns)
    close, cfg = px["close"], default_cfg(ns)
    tr = tt.Tracker(ns, horizon=5)
    asof = close.index[-6]                      # one trading week before the last bar
    print("\nreplay as of %s -> data ends %s" % (asof.date(), close.index[-1].date()))

    sig = tr.signals_asof(px, iv, cfg, asof)
    print("signals on the board: %d" % len(sig))
    print(sig.groupby("engine").size().to_string() if len(sig) else "  (none)")
    assert len(sig), "synthetic market produced no signals — the fixture is broken"

    res, summ = tr.run(px, iv, cfg, asof)
    print("\n" + tt.report_text(res, summ, cfg))

    assert len(res) >= 5, "expected at least the top-5 picks"
    assert res["kind"].notna().all(), res[res["kind"].isna()][["engine", "name", "side"]]
    assert (res["outcome"] != "N.A.").all(), res[res["outcome"] == "N.A."][["name", "note"]]
    assert res["sessions"].iloc[0] == 5
    assert res["entry_date"].iloc[0] == asof

    n_sec = sum(any(c in tt.SECTORS for c in cs) for cs in res["sectors"])
    assert n_sec >= min(tt.SECTOR_N, len(res)), "metals/energy quota not filled"
    print("\nmetals/energy trades tracked: %d" % n_sec)

    # Every P&L recomputed independently from the raw frames.
    ivf = iv.reindex(close.index).ffill()
    entry, exit_ = res["entry_date"].iloc[0], res["exit_date"].iloc[0]
    checked = set()
    for _, r in res.iterrows():
        k = r["kind"]
        if k == "vol_single" and r["engine"] == "IV mean-reversion":
            tk, sgn = r["legs"][0]
            want = sgn * (ivf.at[exit_, tk] - ivf.at[entry, tk])
        elif k == "vol_single":                 # variance risk premium
            tk, sgn = r["legs"][0]
            rvf = tr.realized_vol(px, cfg["rv_estimator"], 5)
            want = sgn * (rvf.at[exit_, tk] - ivf.at[entry, tk])
        elif k == "vol_pair":
            (rich, _), (cheap, _) = r["legs"]
            want = ((ivf.at[exit_, cheap] - ivf.at[entry, cheap]) -
                    (ivf.at[exit_, rich] - ivf.at[entry, rich]))
        elif k == "px_pair":
            (a, sa), (b, sb) = r["legs"]
            want = (sa * (close.at[exit_, a] / close.at[entry, a] - 1) +
                    sb * (close.at[exit_, b] / close.at[entry, b] - 1)) * 100
        else:                                   # px_single
            tk, sgn = r["legs"][0]
            want = sgn * (close.at[exit_, tk] / close.at[entry, tk] - 1) * 100
        assert abs(want - r["pnl"]) < 1e-9, (r["engine"], r["name"], want, r["pnl"])
        checked.add(k)
    print("P&L independently reproduced for every trade (kinds: %s)" % ", ".join(sorted(checked)))

    # No look-ahead: truncating the frames at the as-of date changes nothing.
    sig_t = tr.signals_asof({k: v.loc[:asof] for k, v in px.items()}, iv.loc[:asof], cfg, asof)
    assert list(sig_t["name"]) == list(sig["name"]) and \
        np.allclose(sig_t["conf"].values, sig["conf"].values), "replay saw future data"
    print("no-look-ahead: identical signals from truncated data")

    # An unseasoned replay reports OPEN rather than inventing a mark.
    res_open, _ = tr.run(px, iv, cfg, close.index[-1])
    assert (res_open["outcome"] == "OPEN").all(), res_open["outcome"].tolist()
    print("unseasoned replay reported OPEN")

    # Leg parsing across every live signal, then against hand-built side strings
    # for the engines this tape happened not to fire.
    bad = [(r["engine"], r["side"]) for _, r in sig.iterrows()
           if not tr.legs(r["engine"], r["name"], r["side"])[1]]
    assert not bad, bad
    nm = ns["NAME"]
    cases = [("Correlation RV", "Gold / Silver",
              "BUY %s / SELL %s" % (nm["GC1 Comdty"], nm["SI1 Comdty"]),
              "px_pair", [("GC1 Comdty", 1.0), ("SI1 Comdty", -1.0)]),
             ("Correlation RV", "Gold / US Dollar (DXY)",
              "BUY %s / SELL %s" % (nm["DXY Curncy"], nm["GC1 Comdty"]),
              "px_pair", [("DXY Curncy", 1.0), ("GC1 Comdty", -1.0)]),
             ("Vol dispersion (pairs)", "WTI Crude / Brent Crude",
              "SELL %s VOL / BUY %s VOL" % (nm["CL1 Comdty"], nm["CO1 Comdty"]),
              "vol_pair", [("CL1 Comdty", -1.0), ("CO1 Comdty", 1.0)]),
             ("Lead-lag catch-up", "Copper leads Aluminium LME",
              "SELL %s" % nm["LMAHDS03 Comdty"], "px_single", [("LMAHDS03 Comdty", -1.0)]),
             ("IV mean-reversion", nm["NG1 Comdty"], "BUY VOL",
              "vol_single", [("NG1 Comdty", 1.0)]),
             ("Variance risk premium", nm["TIO1 Comdty"], "SELL VOL",
              "vol_single", [("TIO1 Comdty", -1.0)])]
    for eng, name, side, kind, legs in cases:
        assert tr.legs(eng, name, side) == (kind, legs), (eng, side)
    assert tr.legs("Correlation RV", "x / y", "BUY Nonexistent / SELL Gold") == (None, [])
    print("leg parsing: %d live signals + %d hand-built cases resolved" % (len(sig), len(cases)))

    # Sector quota: ask for a sector the top-5 misses and check the top-up.
    alt = ("Softs", "Livestock")
    picks_alt = tr.select(sig, top_n=5, sectors=alt, sector_n=2)
    quota = picks_alt[picks_alt["basis"] != "top confidence"]
    assert len(picks_alt) > 5 and len(quota) >= 1, picks_alt[["engine", "side", "basis"]]
    assert all(any(c in alt for c in cs) for cs in quota["sectors"])
    pool = [r["conf"] for _, r in sig.iterrows()
            if any(tr.ASSET_CLASS.get(t) in alt
                   for t, _ in tr.legs(r["engine"], r["name"], r["side"])[1])]
    assert quota["conf"].max() <= max(pool) + 1e-9, "quota did not take the best available"
    print("sector quota added %d trade(s) from %s" % (len(quota), "/".join(alt)))

    # Journal: writes, reloads, and replaces rather than duplicates.
    tmp = tempfile.mkdtemp(prefix="tracker-selftest-")
    out = os.path.join(tmp, "journal")
    path = tt.save_run(res, out)
    assert os.path.exists(path) and len(tt.load_ledger(out)) == len(res)
    tt.save_run(res, out)
    assert len(tt.load_ledger(out)) == len(res), "re-running a date duplicated the ledger"
    print("journal round-trip ok -> %s" % path)

    html = tt.report_html(res, summ, cfg)
    assert "Trade-signal outcomes" in html and len(html) > 2000
    open(os.path.join(tmp, "report.html"), "w").write(html)
    print("html report ok (%d chars) -> %s" % (len(html), os.path.join(tmp, "report.html")))
    return px, iv


def check_notebook(tt, ns, px, iv):
    """Execute the tracker notebook's code cells against the synthetic market."""
    with open(TRACKER_NB) as fh:
        nb = json.load(fh)
    cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    # Run it from a scratch copy of the folder: the notebook locates its inputs
    # relative to the working directory, and this keeps trade_journal/ out of
    # the repo when the test runs.
    sandbox = tempfile.mkdtemp(prefix="tracker-nb-")
    for f in ("commodities_vol_rv_dashboard.ipynb", "trade_tracker.py"):
        os.symlink(os.path.join(HERE, f), os.path.join(sandbox, f))
    g = {"__name__": "__nbtest__"}
    cwd = os.getcwd()
    os.chdir(sandbox)
    try:
        for i, src in enumerate(cells):
            exec(compile(src, "trade_outcome_tracker.ipynb#code%d" % i, "exec"), g)
            if i == 0:
                # cell 0 loads the dashboard; swap the Bloomberg pull for the fixture
                g["NS"]["fetch_all"] = lambda *a, **k: (px, iv, [], [])
    finally:
        os.chdir(cwd)
    assert not g["RESULTS"].empty and "hit" in g["SUMMARY"]
    assert (g["RESULTS"]["outcome"] != "N.A.").all()
    assert len(g["frames"]) >= 2, "multi-week replay produced nothing"
    print("notebook: %d code cells executed, %d trades scored, %d weeks pooled"
          % (len(cells), len(g["RESULTS"]), len(g["frames"])))


def main():
    stub_bquant()
    import trade_tracker as tt
    ns = tt.load_dashboard(DASHBOARD)
    print("loaded dashboard cells %s — %d assets, %d with IV"
          % (ns["__loaded_cells__"], len(ns["ALL_TICKERS"]), len(ns["IV_TICKERS"])))
    px, iv = check_tracker(tt, ns)
    check_notebook(tt, ns, px, iv)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
