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
ROOT = os.path.dirname(HERE)

DASHBOARD = os.path.join(ROOT, "commodities_vol_rv_dashboard.ipynb")
TRACKER_NB = os.path.join(ROOT, "trade_outcome_tracker.ipynb")


def notebook_code_cells(path):
    with open(path) as fh:
        nb = json.load(fh)
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def load_inlined_library():
    """Exec the notebook's TRACKER LIBRARY cell as if it were a module.

    Deliberately not `import tracker_source`: what ships to BQuant is the copy
    inlined in the notebook, so that copy is what gets tested.
    """
    cells = notebook_code_cells(TRACKER_NB)
    lib = [c for c in cells if "TRACKER LIBRARY" in c[:400]]
    assert len(lib) == 1, "expected exactly one library cell, found %d" % len(lib)
    mod = types.ModuleType("tracker_inlined")
    exec(compile(lib[0], "trade_outcome_tracker.ipynb#library", "exec"), mod.__dict__)

    with open(os.path.join(HERE, "tracker_source.py")) as fh:
        src = fh.read()
    assert src.strip("\n") in lib[0], \
        "the notebook's library cell is stale — re-run: python dev/build_notebook.py"
    return mod


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

    # Every engine that fired must be represented, and every engine that did not
    # must be reported as silent rather than quietly missing.
    fired = set(sig["engine"])
    tracked = set(res["engine"])
    assert fired <= tracked, "engines with signals but no tracked trade: %s" % (fired - tracked)
    assert set(summ["silent_engines"]) == set(tt.ENGINES) - fired, summ["silent_engines"]
    assert set(summ["by_engine"]) == tracked
    assert set(summ["by_engine"]) | set(summ["silent_engines"]) == set(tt.ENGINES)
    print("engine coverage: %d/%d engines fired, all tracked; silent: %s"
          % (len(fired), len(tt.ENGINES), ", ".join(summ["silent_engines"]) or "none"))

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
    picks_alt = tr.select(sig, top_n=5, sectors=alt, sector_n=2, engine_n=0)
    quota = picks_alt[picks_alt["basis"] == "metals/energy quota"]
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
    """Execute the tracker notebook's code cells against the synthetic market.

    This is what proves the shipped artefact works: the notebook has to find the
    dashboard on its own and run with no .py file anywhere near it.
    """
    cells = notebook_code_cells(TRACKER_NB)
    # A scratch folder holding only the dashboard notebook — no tracker_source.py,
    # nothing on sys.path — so an accidental `import` would fail loudly here.
    sandbox = tempfile.mkdtemp(prefix="tracker-nb-")
    os.symlink(DASHBOARD, os.path.join(sandbox, os.path.basename(DASHBOARD)))
    g = {"__name__": "__nbtest__"}
    cwd, path_before = os.getcwd(), list(sys.path)
    os.chdir(sandbox)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (HERE, ROOT)]
    try:
        for i, src in enumerate(cells):
            exec(compile(src, "trade_outcome_tracker.ipynb#code%d" % i, "exec"), g)
            if "NS" in g and "fetch_all" not in g.get("_patched", ()):
                # swap the Bloomberg pull for the fixture as soon as NS exists
                g["NS"]["fetch_all"] = lambda *a, **k: (px, iv, [], [])
                g["_patched"] = ("fetch_all",)
    finally:
        os.chdir(cwd)
        sys.path[:] = path_before

    assert g["DASHBOARD"] == os.path.join(sandbox, os.path.basename(DASHBOARD)), \
        "notebook did not locate the dashboard on its own: %s" % g.get("DASHBOARD")
    assert not g["RESULTS"].empty and "hit" in g["SUMMARY"]
    assert (g["RESULTS"]["outcome"] != "N.A.").all()
    assert len(g["frames"]) >= 2, "multi-week replay produced nothing"
    assert os.path.isdir(os.path.join(sandbox, "trade_journal")), "journal was not written"
    print("notebook: %d code cells executed standalone, %d trades scored, %d weeks pooled"
          % (len(cells), len(g["RESULTS"]), len(g["frames"])))


def check_inline_mode(tt, px, iv):
    """The other documented route: cells pasted onto the end of the dashboard.

    The engines are then already in the namespace, so the setup cell must reuse
    them and never go looking for a notebook on disk.
    """
    cells = notebook_code_cells(TRACKER_NB)
    g = {"__name__": "__inline__"}
    tt.load_dashboard(DASHBOARD, ns=g)          # stand in for the dashboard's own kernel
    assert "build_signals" in g

    sandbox = tempfile.mkdtemp(prefix="tracker-inline-")   # empty: nothing to find
    cwd = os.getcwd()
    os.chdir(sandbox)
    try:
        for i, src in enumerate(cells):
            exec(compile(src, "trade_outcome_tracker.ipynb#inline%d" % i, "exec"), g)
            if "NS" in g and "_patched" not in g:
                g["NS"]["fetch_all"] = lambda *a, **k: (px, iv, [], [])
                g["_patched"] = True
    finally:
        os.chdir(cwd)

    assert g["NS"] is g, "setup cell did not reuse the notebook's own namespace"
    assert "DASHBOARD" not in g, "setup cell went looking for a file it did not need"
    assert not g["RESULTS"].empty
    print("inline mode: %d cells ran inside the dashboard's namespace, %d trades scored"
          % (len(cells), len(g["RESULTS"])))


def main():
    stub_bquant()
    tt = load_inlined_library()
    ns = tt.load_dashboard(DASHBOARD)
    print("loaded dashboard cells %s — %d assets, %d with IV"
          % (ns["__loaded_cells__"], len(ns["ALL_TICKERS"]), len(ns["IV_TICKERS"])))
    px, iv = check_tracker(tt, ns)
    check_notebook(tt, ns, px, iv)
    check_inline_mode(tt, px, iv)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
