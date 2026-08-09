"""Self-test — runs anywhere, no Bloomberg needed.

Stubs the BQuant-only imports, execs the dashboard's real CONFIG + ANALYTICS
cells, and pushes a generated market through the whole
replay -> mark -> backtest -> report path. Statistics are checked against hand
values and baselines are recomputed straight from the raw frames, so a wrong
number fails rather than merely looking plausible.

It also executes the shipped notebook's own cells — from a folder containing
nothing but the dashboard notebook — so the inlined libraries cannot drift from
their sources and no hidden .py dependency can creep in.

    python dev/test_tracker.py
"""
import json
import math
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DASHBOARD = os.path.join(ROOT, "commodities_vol_rv_dashboard.ipynb")
TRACKER_NB = os.path.join(ROOT, "strategy_backtest.ipynb")


def notebook_code_cells(path):
    with open(path) as fh:
        nb = json.load(fh)
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


LIBRARIES = [("REPLAY LIBRARY", "replay_source.py"),
             ("BACKTEST LIBRARY", "backtest_source.py")]


def load_inlined_library():
    """Exec the notebook's library cells as one module.

    Deliberately not `import tracker_source`: what ships to BQuant is the copy
    inlined in the notebook, so that copy is what gets tested. The two cells
    share a namespace in the notebook, so they share one here too.
    """
    cells = notebook_code_cells(TRACKER_NB)
    mod = types.ModuleType("tracker_inlined")
    for title, filename in LIBRARIES:
        lib = [c for c in cells if title in c[:400]]
        assert len(lib) == 1, "expected one %s cell, found %d" % (title, len(lib))
        exec(compile(lib[0], "strategy_backtest.ipynb#%s" % filename, "exec"), mod.__dict__)
        with open(os.path.join(HERE, filename)) as fh:
            src = fh.read()
        assert src.strip("\n") in lib[0], \
            "%s is stale in the notebook — re-run: python dev/build_notebook.py" % filename
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

    # Stub plotly/ipywidgets only if they are genuinely absent. Where they are
    # installed, use the real thing — the charts are a deliverable, and a stub
    # would let a broken figure call pass the test.
    missing = []
    for m in ["plotly", "plotly.graph_objects", "plotly.subplots", "ipywidgets"]:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    for m in missing:
        mod = types.ModuleType(m)
        mod.__getattr__ = lambda n: types.SimpleNamespace()
        sys.modules[m] = mod
    if "plotly" in missing:
        sys.modules["plotly"].graph_objects = sys.modules["plotly.graph_objects"]
        sys.modules["plotly"].subplots = sys.modules["plotly.subplots"]

    try:
        __import__("IPython.display")
    except ImportError:
        disp = types.ModuleType("IPython.display")
        disp.HTML = lambda s: s
        disp.display = lambda *a, **k: None
        ipy = types.ModuleType("IPython")
        ipy.display = disp
        sys.modules["IPython"] = ipy
        sys.modules["IPython.display"] = disp
        missing.append("IPython.display")
    return dict(stubbed=missing)


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
def check_replay(tt, ns):
    """The replay half: right signals, right legs, right marks, no look-ahead."""
    px, iv = synthetic_market(ns)
    close, cfg = px["close"], default_cfg(ns)
    tr = tt.Tracker(ns, horizon=5)
    asof = close.index[-6]                      # one trading week before the last bar
    print("\nreplay as of %s -> data ends %s" % (asof.date(), close.index[-1].date()))

    sig = tr.signals_asof(px, iv, cfg, asof)
    print("signals on the board: %d" % len(sig))
    print(sig.groupby("engine").size().to_string() if len(sig) else "  (none)")
    assert len(sig), "synthetic market produced no signals — the fixture is broken"

    trades = tr.annotate(sig)
    assert len(trades) == len(sig), "annotate dropped signals"
    bad = trades[trades["kind"].isna()]
    assert bad.empty, "legs unresolved: %s" % bad[["engine", "name", "side"]].to_dict("records")

    res = tr.score(trades, px, iv, cfg, asof)
    assert (res["outcome"] != "N.A.").all(), res[res["outcome"] == "N.A."][["name", "note"]]
    assert (res["sessions"] == 5).all() and (res["entry_date"] == asof).all()

    # Every P&L recomputed independently from the raw frames.
    ivf = iv.reindex(close.index).ffill()
    rv5 = tr.realized_vol(px, cfg["rv_estimator"], 5)
    entry, exit_ = res["entry_date"].iloc[0], res["exit_date"].iloc[0]
    checked = set()
    for _, r in res.iterrows():
        k = r["kind"]
        if k == "vol_single" and r["engine"] == "IV mean-reversion":
            tk, sgn = r["legs"][0]
            want = sgn * (ivf.at[exit_, tk] - ivf.at[entry, tk])
        elif k == "vol_single":                 # variance risk premium
            tk, sgn = r["legs"][0]
            want = sgn * (rv5.at[exit_, tk] - ivf.at[entry, tk])
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
    print("P&L independently reproduced for all %d trades (kinds: %s)"
          % (len(res), ", ".join(sorted(checked))))

    # No look-ahead: truncating the frames at the as-of date changes nothing.
    sig_t = tr.signals_asof({k: v.loc[:asof] for k, v in px.items()}, iv.loc[:asof], cfg, asof)
    assert list(sig_t["name"]) == list(sig["name"]) and \
        np.allclose(sig_t["conf"].values, sig["conf"].values), "replay saw future data"
    print("no-look-ahead: identical signals from truncated data")

    # An unseasoned replay reports OPEN rather than inventing a mark.
    last = tr.annotate(tr.signals_asof(px, iv, cfg, close.index[-1]))
    res_open = tr.score(last, px, iv, cfg, close.index[-1])
    assert (res_open["outcome"] == "OPEN").all(), res_open["outcome"].unique().tolist()
    print("unseasoned replay reported OPEN")

    # Leg parsing across every live signal, then against hand-built side strings
    # for the engines this tape happened not to fire.
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
    return px, iv


def check_stats(tt):
    """The statistics have known answers — check them against hand values."""
    lo, hi = tt.wilson(50, 100)
    assert abs(lo - 40.4) < 0.2 and abs(hi - 59.6) < 0.2, (lo, hi)
    lo, hi = tt.wilson(1, 3)                     # tiny n must stay inside [0, 100]
    assert 0 <= lo < hi <= 100

    # P(X >= 50 | n=100, p=0.5) is just over a half; the extremes are exact.
    assert abs(tt.binom_p(50, 100, 0.5) - 0.5398) < 0.001, tt.binom_p(50, 100, 0.5)
    assert abs(tt.binom_p(100, 100, 0.5) - 0.5 ** 100) < 1e-30
    assert abs(tt.binom_p(0, 100, 0.5, side="less") - 0.5 ** 100) < 1e-30
    assert tt.binom_p(60, 100, 0.5) < 0.03 < tt.binom_p(55, 100, 0.5)
    # The two tails must together account for the whole distribution plus the
    # shared point mass at k.
    k, n, p = 57, 100, 0.5
    both = tt.binom_p(k, n, p) + tt.binom_p(k, n, p, side="less")
    assert abs(both - (1 + math.exp(tt._log_binom_pmf(k, n, p)))) < 1e-9, both
    print("statistics: Wilson intervals and exact binomial tails match hand values")


def _block_hit(tt, marks, horizon=5):
    return tt.headline(marks, horizon)["hit"]


def check_backtest(tt, ns, px, iv):
    """Run a short backtest and verify the numbers it reports."""
    cfg = default_cfg(ns)
    tr = tt.Tracker(ns, horizon=5)
    close = px["close"]
    weeks, horizons = 10, (5, 21)
    marks = tt.run_backtest(tr, px, iv, cfg, weeks=weeks, step=5, horizons=horizons)
    assert not marks.empty
    assert set(marks["horizon"]) == set(horizons)
    assert marks["entry_date"].nunique() == weeks, marks["entry_date"].nunique()

    # Every mark must be a real, closed, forward-looking hold.
    assert (marks["exit_date"] > marks["entry_date"]).all()
    assert (marks["exit_date"] <= close.index[-1]).all(), "marked past the end of the data"
    for h in horizons:
        sub = marks[marks["horizon"] == h]
        gap = [close.index.get_loc(b) - close.index.get_loc(a)
               for a, b in zip(sub["entry_date"], sub["exit_date"])]
        assert set(gap) == {h}, "horizon %d held %s sessions" % (h, sorted(set(gap)))

    # win/outcome/edge must agree with each other.
    assert ((marks["win"] == 1) == (marks["outcome"] == "WIN")).all()
    scored = marks[marks["baseline"].notna()]
    assert len(scored) > 0.8 * len(marks), "too many baselines failed to compute"
    assert np.allclose(scored["edge"], scored["win"] * 100.0 - scored["baseline"])

    # Recompute one baseline by hand, straight from the raw frames.
    ivf = iv.reindex(close.index).ffill()
    row = marks[(marks["engine"] == "IV mean-reversion") & (marks["horizon"] == 5)
                & marks["baseline"].notna()].iloc[0]
    tk, sgn = row["legs"][0]
    d = sgn * (ivf[tk].shift(-5) - ivf[tk])
    want = float((d.dropna() > 0).mean() * 100.0)
    assert abs(want - row["baseline"]) < 1e-9, (want, row["baseline"])

    # ...and one for the variance risk premium, whose baseline is the whole point:
    # implied at entry against the vol actually delivered, not against the re-mark.
    vrp = marks[(marks["engine"] == "Variance risk premium") & (marks["horizon"] == 5)
                & marks["baseline"].notna()]
    assert len(vrp), "no VRP baselines computed"
    row = vrp.iloc[0]
    tk, sgn = row["legs"][0]
    rv5 = tr.realized_vol(px, cfg["rv_estimator"], 5)
    d = sgn * (rv5[tk].shift(-5) - ivf[tk])
    want = float((d.dropna() > 0).mean() * 100.0)
    assert abs(want - row["baseline"]) < 1e-9, (want, row["baseline"])

    # Aggregation, calibration and the written summary.
    tab = tt.table(marks, "engine", horizon=5)
    assert set(["engine", "n", "hit", "lo", "hi", "baseline", "lift", "p"]) <= set(tab.columns)
    assert (tab["n"] > 0).any() and (tab["lo"] <= tab["hit"]).all() and (tab["hit"] <= tab["hi"]).all()
    assert abs(tab["n"].sum() - len(marks[marks["horizon"] == 5])) == 0

    cal = tt.calibration(marks, 5)
    assert len(cal) and cal["n"].sum() == len(marks[marks["horizon"] == 5])

    hd = tt.headline(marks, 5)
    assert 0 <= hd["hit"] <= 100 and hd["trades"] == len(marks[marks["horizon"] == 5])

    eq = tt.equity(marks, 5)
    assert len(eq) and eq.notna().all().all()

    points = tt.takeaways(marks, 5)
    assert len(points) >= 3 and all(isinstance(p, str) and len(p) > 20 for p in points)

    # Charts must actually build — a stubbed plotly would hide a broken call.
    for fig in (tt.fig_calibration(marks, 5), tt.fig_engines(marks, 5),
                tt.fig_equity(marks, 5), tt.fig_horizons(marks)):
        assert len(fig.data) >= 1 and fig.layout.title.text

    html = tt.report_html(marks, 5, cfg) + tt.table_html(marks, 5)
    assert "Do these signals actually work?" in html and len(html) > 3000

    # Confidence diagnostics: the numbers have to agree with each other.
    cd = tt.conf_diagnostics(marks, 5)
    f = cd["filter"]
    assert (f["n"].diff().dropna() <= 0).all(), "raising the threshold kept more trades"
    assert f.iloc[0]["n"] == len(marks[marks["horizon"] == 5]), "no-filter row dropped trades"
    for _, r in f.iterrows():
        sub = marks[(marks["horizon"] == 5) & (marks["conf"] >= r["threshold"])]
        assert r["n"] == len(sub) and abs(r["kept"] - 100.0 * len(sub) / f.iloc[0]["n"]) < 1e-9
    ce = cd["by_engine"]
    assert set(ce["engine"]) <= set(marks["engine"])
    for _, r in ce.iterrows():
        sub = marks[(marks["horizon"] == 5) & (marks["engine"] == r["engine"])
                    & (marks["conf_bin"] == r["conf_bin"])]
        assert r["n"] == len(sub), (r["engine"], r["conf_bin"], r["n"], len(sub))
    for k in ("rank_raw", "rank_within", "rank_adjusted"):
        assert cd[k] is not None and (np.isnan(cd[k]) or -1 <= cd[k] <= 1), (k, cd[k])
    assert len(tt.conf_takeaways(marks, 5)) >= 2
    for fig in (tt.fig_conf_by_engine(marks, 5), tt.fig_conf_filter(marks, 5)):
        assert fig.layout.title.text
    # ...and again in the Standard Chartered palette, which the deck exports use.
    sc = tt.fig_engines(marks, 5, theme=tt.SC_THEME, colors=tt.SC_ENGINE_COLOR)
    assert sc.layout.paper_bgcolor == tt.SC_THEME["BG"]

    # The export pack must write something usable even with no PNG backend.
    outdir = os.path.join(tempfile.mkdtemp(prefix="pack-"), "presentation")
    files = tt.export_pack(marks, 5, cfg, outdir=outdir)
    assert len(files) >= len(tt.FIGURES) + 6
    for name, _, _ in tt.FIGURES:
        assert os.path.exists(os.path.join(outdir, "%s.html" % name)), name
    numbers = open(os.path.join(outdir, "NUMBERS.md")).read()
    for heading in ("## Headline", "## By engine", "## Confidence calibration",
                    "## Does confidence rank outcomes?", "## How to read these numbers"):
        assert heading in numbers, heading
    assert "%.1f%%" % _block_hit(tt, marks) in numbers, "headline hit rate missing from NUMBERS.md"
    print("confidence diagnostics + export pack verified (%d files)" % len(files))

    print("backtest: %d marks over %d dates × %d horizons; baselines, aggregation, "
          "charts and takeaways verified" % (len(marks), weeks, len(horizons)))
    for line in points:
        print("   · %s" % line)
    return marks


def check_notebook(tt, ns, px, iv):
    """Execute the backtest notebook's code cells against the synthetic market.

    This is what proves the shipped artefact works: the notebook has to find the
    dashboard on its own and run with no .py file anywhere near it.
    """
    cells = notebook_code_cells(TRACKER_NB)
    # A scratch folder holding only the dashboard notebook — no source .py files,
    # nothing on sys.path — so an accidental `import` would fail loudly here.
    sandbox = tempfile.mkdtemp(prefix="backtest-nb-")
    os.symlink(DASHBOARD, os.path.join(sandbox, os.path.basename(DASHBOARD)))
    g = {"__name__": "__nbtest__"}
    cwd, path_before = os.getcwd(), list(sys.path)
    os.chdir(sandbox)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in (HERE, ROOT)]
    try:
        for i, src in enumerate(cells):
            exec(compile(src, "strategy_backtest.ipynb#code%d" % i, "exec"), g)
            if "NS" in g and "_data" not in g:
                # swap the Bloomberg pull for the fixture as soon as NS exists
                g["NS"]["fetch_all"] = lambda *a, **k: (px, iv, [], [])
                g["_data"] = True
            if "WEEKS" in g and "_weeks" not in g:
                g["WEEKS"] = 8            # keep the test quick; the notebook ships with 52
                g["_weeks"] = True
    finally:
        os.chdir(cwd)
        sys.path[:] = path_before

    assert g["DASHBOARD"] == os.path.join(sandbox, os.path.basename(DASHBOARD)), \
        "notebook did not locate the dashboard on its own: %s" % g.get("DASHBOARD")
    assert not g["MARKS"].empty and g["MARKS"]["entry_date"].nunique() == 8
    for f in ("backtest_report.html", "backtest_marks.csv"):
        p = os.path.join(sandbox, f)
        assert os.path.exists(p) and os.path.getsize(p) > 2000, "%s not written" % f
    print("notebook: %d code cells executed standalone, %d marks, exports written"
          % (len(cells), len(g["MARKS"])))


def check_inline_mode(tt, px, iv):
    """The other documented route: cells pasted onto the end of the dashboard.

    The engines are then already in the namespace, so the setup cell must reuse
    them and never go looking for a notebook on disk.
    """
    cells = notebook_code_cells(TRACKER_NB)
    g = {"__name__": "__inline__"}
    tt.load_dashboard(DASHBOARD, ns=g)          # stand in for the dashboard's own kernel
    assert "build_signals" in g

    sandbox = tempfile.mkdtemp(prefix="backtest-inline-")   # empty: nothing to find
    cwd = os.getcwd()
    os.chdir(sandbox)
    try:
        for i, src in enumerate(cells):
            exec(compile(src, "strategy_backtest.ipynb#inline%d" % i, "exec"), g)
            if "NS" in g and "_data" not in g:
                g["NS"]["fetch_all"] = lambda *a, **k: (px, iv, [], [])
                g["_data"] = True
            if "WEEKS" in g and "_weeks" not in g:
                g["WEEKS"] = 4
                g["_weeks"] = True
    finally:
        os.chdir(cwd)

    assert g["NS"] is g, "setup cell did not reuse the notebook's own namespace"
    assert "DASHBOARD" not in g, "setup cell went looking for a file it did not need"
    assert not g["MARKS"].empty
    print("inline mode: %d cells ran inside the dashboard's namespace, %d marks"
          % (len(cells), len(g["MARKS"])))


def main():
    stub_bquant()
    tt = load_inlined_library()
    ns = tt.load_dashboard(DASHBOARD)
    print("loaded dashboard cells %s — %d assets, %d with IV"
          % (ns["__loaded_cells__"], len(ns["ALL_TICKERS"]), len(ns["IV_TICKERS"])))
    px, iv = check_replay(tt, ns)
    check_stats(tt)
    check_backtest(tt, ns, px, iv)
    check_notebook(tt, ns, px, iv)
    check_inline_mode(tt, px, iv)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
