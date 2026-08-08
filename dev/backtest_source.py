"""
Backtest — do these signals actually work?
=========================================

The tracker answered "how did last week's ideas do". This answers the harder
question over a year of history, on the whole population of signals rather than
a handful of picks.

Method
------
Walk the year in steps of STEP sessions. At each step, rewind the price/IV
history to that date and re-run the dashboard's `build_signals` — every engine
reads the last row of each series, so the truncated frame reproduces exactly
what the dashboard would have printed that day, confidence hit-rates included,
with no look-ahead. Every signal on the board is recorded, not just the good
ones, and each is marked forward over several holding periods.

No sizing. Every trade is one unit and the headline number is a hit rate, which
keeps the result about signal quality rather than about a position-sizing rule
layered on top. Average P&L per trade is reported alongside, in each engine's
natural unit (vol points or percent).

Three things make the result honest
-----------------------------------
1. **Baselines.** A hit rate means nothing on its own. "Sell vol wins 78% of the
   time" is unimpressive if implied vol exceeds subsequent realized vol 76% of
   the time *anyway* — that is the structural variance premium, not a signal.
   So every trade gets an unconditional baseline: the same bet, same asset, same
   horizon, taken on every date in the sample regardless of whether the engine
   fired. The number that matters is the gap between the two.

2. **Confidence intervals.** Wilson score intervals on every hit rate, and a
   binomial p-value against the trade's own baseline. An engine with 12 trades
   is not evidence, and the chart shows that rather than hiding it.

3. **Overlap is disclosed, not hidden.** At STEP=5 the 5-day holds are
   consecutive and non-overlapping, but a signal that persists for weeks is
   re-recorded each week, so trades are not independent draws. Longer horizons
   overlap outright. Effective sample size is smaller than the trade count; the
   report says so where it matters.
"""

import math

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (5, 10, 21)
DEFAULT_STEP = 5                    # sessions between replays
DEFAULT_WEEKS = 52
MIN_HISTORY = 400                   # sessions needed before percentiles mean anything

CONF_BINS = [(0, 45), (45, 55), (55, 65), (65, 75), (75, 101)]
CONF_LABELS = ["<45", "45-55", "55-65", "65-75", "75+"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion. Behaves sanely at small n,
    which the normal approximation does not."""
    if not n:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0)


def _log_binom_pmf(k, n, p):
    if p <= 0:
        return 0.0 if k == 0 else -np.inf
    if p >= 1:
        return 0.0 if k == n else -np.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def _tail(k, n, p0, upper):
    lo, hi = (k, n) if upper else (0, k)
    terms = [_log_binom_pmf(i, n, p0) for i in range(lo, hi + 1)]
    m = max(terms)
    if not np.isfinite(m):
        return 0.0
    return float(min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms)))


def binom_p(k, n, p0, side="greater"):
    """Exact one-sided binomial p-value against a true rate of p0.

    `greater` asks whether the engine beat its baseline; `less` asks whether it
    did materially worse, which is its own finding — a setup that loses to its
    own baseline is a signal to fade, not noise.

    Exact rather than normal-approximated: several buckets here are small enough
    that the approximation overstates significance, which is the one failure
    mode this whole file exists to avoid.
    """
    if not n or not np.isfinite(p0):
        return np.nan
    p0 = min(max(float(p0), 1e-9), 1 - 1e-9)
    k = int(round(k))
    if side == "less":
        return _tail(k, n, p0, upper=False)
    if k <= 0:
        return 1.0
    return _tail(k, n, p0, upper=True)


# ---------------------------------------------------------------------------
# Baselines — what this bet pays with no signal at all
# ---------------------------------------------------------------------------
class Baselines:
    """Unconditional win rate for a given bet, computed once and cached.

    The key is (kind, engine, horizon, legs-with-signs): the same trade
    structure on the same assets in the same direction, evaluated on *every*
    date in the sample rather than only the dates the engine fired. Comparing a
    signal's hit rate against this is what separates an edge from a structural
    drift the engine happened to sit on top of.
    """

    def __init__(self, close, ivf, rv_by_h):
        self.close = close
        self.ivf = ivf
        self.rv_by_h = rv_by_h
        self._cache = {}

    def _series_win(self, kind, engine, h, legs):
        """Boolean series: would this bet have won, entered on each date?"""
        h = int(h)
        if kind in ("vol_single", "vol_pair"):
            total = None
            for tk, sgn in legs:
                if tk not in self.ivf.columns:
                    return None
                iv0 = self.ivf[tk]
                if engine == "Variance risk premium":
                    rv = self.rv_by_h.get(h)
                    if rv is None or tk not in rv.columns:
                        return None
                    # delivered vol over (t, t+h] against the implied quoted at t
                    leg = sgn * (rv[tk].shift(-h) - iv0)
                else:
                    leg = sgn * (iv0.shift(-h) - iv0)
                total = leg if total is None else total + leg
            return total
        # price legs
        total = None
        for tk, sgn in legs:
            if tk not in self.close.columns:
                return None
            p = self.close[tk]
            leg = sgn * (p.shift(-h) / p - 1.0) * 100.0
            total = leg if total is None else total + leg
        return total

    def rate(self, kind, engine, h, legs):
        """-> unconditional win rate in %, or NaN if it cannot be computed."""
        if not legs or kind is None:
            return np.nan
        key = (kind, engine, int(h), tuple(sorted((t, float(s)) for t, s in legs)))
        if key in self._cache:
            return self._cache[key]
        pnl = self._series_win(kind, engine, h, legs)
        if pnl is None:
            out = np.nan
        else:
            # A DataFrame here means a leg was broadcast across every column
            # instead of being indexed — a bug, not missing data, and it would
            # otherwise disappear as a silent NaN baseline.
            if not isinstance(pnl, pd.Series):
                raise TypeError("baseline for %s/%s produced %s, expected a Series"
                                % (engine, kind, type(pnl).__name__))
            pnl = pnl.dropna()
            out = float((pnl > 0).mean() * 100.0) if len(pnl) >= 30 else np.nan
        self._cache[key] = out
        return out


# ---------------------------------------------------------------------------
# The backtest
# ---------------------------------------------------------------------------
def replay_dates(close, weeks=DEFAULT_WEEKS, step=DEFAULT_STEP, end=None,
                 min_history=MIN_HISTORY):
    """As-of dates, oldest first. Each needs `min_history` sessions behind it."""
    idx = close.index
    last = len(idx) - 1 if end is None else int(idx.searchsorted(pd.Timestamp(end), "right")) - 1
    out = []
    for k in range(1, int(weeks) + 1):
        i = last - k * int(step)
        if i < min_history:
            break
        out.append(idx[i])
    return list(reversed(out))


def run_backtest(tracker, px, iv, cfg, weeks=DEFAULT_WEEKS, step=DEFAULT_STEP,
                 horizons=DEFAULT_HORIZONS, end=None, progress=None):
    """Replay the year and mark every signal at every horizon.

    -> DataFrame, one row per (as-of date, signal, horizon).
    """
    close = px["close"]
    ivf = iv.reindex(close.index).ffill()
    horizons = tuple(int(h) for h in horizons)
    rv_by_h = tracker.realized_by_horizon(px, cfg, horizons)   # once, not per replay
    base = Baselines(close, ivf, rv_by_h)
    dates = replay_dates(close, weeks=weeks, step=step, end=end)
    if not dates:
        raise ValueError("not enough history for a backtest — need %d sessions before "
                         "the first replay date" % MIN_HISTORY)

    frames = []
    for n, asof in enumerate(dates, 1):
        sig = tracker.signals_asof(px, iv, cfg, asof)
        if progress:
            progress(n, len(dates), asof, len(sig))
        if sig is None or sig.empty:
            continue
        trades = tracker.annotate(sig)
        for h in horizons:
            entry, exit_, held = tracker.entry_exit(close, asof, h)
            if exit_ is None or held < h:
                continue                       # not enough forward data: drop, never guess
            marked = tracker.mark(trades, close, ivf, rv_by_h[h], entry, exit_, held)
            if marked.empty:
                continue
            marked["horizon"] = h
            frames.append(marked)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out[out["outcome"].isin(["WIN", "LOSS", "FLAT"])].reset_index(drop=True)
    out["win"] = (out["outcome"] == "WIN").astype(int)
    out["baseline"] = [base.rate(k, e, h, l) for k, e, h, l
                       in zip(out["kind"], out["engine"], out["horizon"], out["legs"])]
    out["edge"] = out["win"] * 100.0 - out["baseline"]
    out["conf_bin"] = pd.cut(out["conf"], bins=[b[0] for b in CONF_BINS] + [CONF_BINS[-1][1]],
                             labels=CONF_LABELS, right=False)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _block(df):
    """Hit rate, interval, baseline and significance for a set of trades."""
    n = len(df)
    if not n:
        return dict(n=0, wins=0, hit=np.nan, lo=np.nan, hi=np.nan, baseline=np.nan,
                    lift=np.nan, p=np.nan, p_low=np.nan, pnl=np.nan, unit="", conf=np.nan)
    wins = int(df["win"].sum())
    hit = wins / n * 100.0
    lo, hi = wilson(wins, n)
    baseline = float(df["baseline"].mean()) if df["baseline"].notna().any() else np.nan
    units = df["unit"].dropna().unique()
    pnl_unit = units[0] if len(units) == 1 else ""
    ok = np.isfinite(baseline)
    return dict(n=n, wins=wins, hit=hit, lo=lo, hi=hi, baseline=baseline,
                lift=(hit - baseline) if ok else np.nan,
                p=binom_p(wins, n, baseline / 100.0) if ok else np.nan,
                p_low=binom_p(wins, n, baseline / 100.0, side="less") if ok else np.nan,
                pnl=float(df["pnl"].mean()) if pnl_unit else np.nan, unit=pnl_unit,
                conf=float(df["conf"].mean()))


def table(marks, by, horizon=None, order=None):
    """Aggregate `marks` by a column -> DataFrame of one row per group."""
    d = marks if horizon is None else marks[marks["horizon"] == int(horizon)]
    rows = []
    keys = list(order) if order else sorted(d[by].dropna().unique())
    for k in keys:
        sub = d[d[by] == k]
        if sub.empty and order is None:
            continue
        r = _block(sub)
        r[by] = k
        rows.append(r)
    cols = [by, "n", "wins", "hit", "lo", "hi", "baseline", "lift", "p", "p_low",
            "pnl", "unit", "conf"]
    return pd.DataFrame(rows, columns=cols)


def headline(marks, horizon):
    """Top-level numbers for the summary cards."""
    d = marks[marks["horizon"] == int(horizon)]
    b = _block(d)
    b.update(trades=len(d), dates=d["entry_date"].nunique(),
             engines=d["engine"].nunique(),
             start=d["entry_date"].min(), end=d["exit_date"].max(),
             assets=len(set(t for legs in d["legs"] for t, _ in (legs or []))))
    # Does confidence rank outcomes at all? Spearman on (confidence, win).
    if len(d) > 20 and d["conf"].nunique() > 3:
        b["conf_corr"] = float(d["conf"].rank().corr(d["win"].rank()))
    else:
        b["conf_corr"] = np.nan
    return b


def calibration(marks, horizon):
    """Predicted confidence vs realized hit rate, per confidence bucket."""
    t = table(marks, "conf_bin", horizon=horizon, order=CONF_LABELS)
    return t[t["n"] > 0].reset_index(drop=True)


def equity(marks, horizon, by="engine"):
    """Cumulative net wins (wins − losses) through time.

    Unit-free on purpose: engines are marked in vol points or percent, which
    cannot be added together, but a win is a win in any unit. A rising line is
    an engine that keeps being right; a flat one is a coin flip.
    """
    d = marks[marks["horizon"] == int(horizon)].sort_values("entry_date")
    out = {}
    for k, sub in d.groupby(by):
        s = sub.groupby("entry_date")["win"].agg(lambda w: int((w == 1).sum() - (w == 0).sum()))
        out[k] = s.sort_index().cumsum()
    return pd.DataFrame(out).ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# Key takeaways — written from the numbers, not by hand
# ---------------------------------------------------------------------------
def takeaways(marks, horizon, max_points=6):
    """Plain-English findings, ordered by how much they matter."""
    h = int(horizon)
    hd = headline(marks, h)
    eng = table(marks, "engine", horizon=h).sort_values("lift", ascending=False)
    cal = calibration(marks, h)
    out = []

    out.append("%d signals replayed across %d dates (%s to %s), held %d sessions each."
               % (hd["trades"], hd["dates"], pd.Timestamp(hd["start"]).date(),
                  pd.Timestamp(hd["end"]).date(), h))

    if np.isfinite(hd["baseline"]):
        verdict = "beat" if hd["lift"] > 0 else "trail"
        out.append("Overall hit rate %.1f%%, against a %.1f%% baseline for the same trades taken "
                   "on every date — the signals %s doing nothing by %+.1f points (p=%.3f)."
                   % (hd["hit"], hd["baseline"], verdict, hd["lift"], hd["p"]))

    good = eng[(eng["n"] >= 30) & (eng["lift"] > 0) & (eng["p"] < 0.05)]
    if len(good):
        r = good.iloc[0]
        out.append("Best engine: %s, %.0f%% vs %.0f%% baseline (%+.1f pts, n=%d, p=%.3f)."
                   % (r["engine"], r["hit"], r["baseline"], r["lift"], r["n"], r["p"]))
    bad = eng[(eng["n"] >= 30) & (eng["lift"] < -2)]
    if len(bad):
        r = bad.iloc[-1]
        out.append("Weakest: %s adds nothing — %.0f%% against a %.0f%% baseline (%+.1f pts, n=%d)."
                   % (r["engine"], r["hit"], r["baseline"], r["lift"], r["n"]))

    if len(cal) >= 3:
        lo, hi = cal.iloc[0], cal.iloc[-1]
        raw = hi["hit"] - lo["hit"]
        net = hi["lift"] - lo["lift"]
        if not np.isfinite(net):
            out.append("Confidence buckets: %s hit %.1f%%, %s hit %.1f%% (%.1f-point spread)."
                       % (lo["conf_bin"], lo["hit"], hi["conf_bin"], hi["hit"], raw))
        elif abs(net) < 3:
            # The important case: a big raw spread that is really the baseline
            # doing the work, because high-confidence buckets are dominated by
            # bets that win anyway.
            out.append("Confidence looks predictive — %s signals hit %.1f%% against %.1f%% for %s "
                       "— but almost all of that %.0f-point gap is baseline, not skill: net of "
                       "what those same trades pay on any date the spread is only %+.1f points. "
                       "The score is mostly identifying bets with a favourable base rate."
                       % (hi["conf_bin"], hi["hit"], lo["hit"], lo["conf_bin"], raw, net))
        else:
            out.append("Confidence carries real information: net of baseline, %s signals run "
                       "%+.1f points of edge against %+.1f for %s — a %.1f-point spread that "
                       "survives the base rate."
                       % (hi["conf_bin"], hi["lift"], lo["lift"], lo["conf_bin"], net))

    hs = sorted(marks["horizon"].unique())
    if len(hs) > 1:
        best = max(hs, key=lambda x: (_block(marks[marks["horizon"] == x])["lift"]
                                      if np.isfinite(_block(marks[marks["horizon"] == x])["lift"])
                                      else -np.inf))
        parts = []
        for x in hs:
            b = _block(marks[marks["horizon"] == x])
            parts.append("%dd %+.1f" % (x, b["lift"]))
        out.append("Edge over baseline by holding period: %s (points). Holds longer than the "
                   "%d-session replay step overlap each other, so the longer horizons are fewer "
                   "independent bets than their trade counts suggest."
                   % (", ".join(parts), DEFAULT_STEP))

    out.append("Every trade is one unit and unsized, so this measures whether the signals point "
               "the right way, not a tradable P&L. Trades overlap where the same setup persists "
               "week to week, so treat the sample as smaller than the trade count suggests.")
    return out[:max_points]


# ---------------------------------------------------------------------------
# Charts — plotly, styled to match the dashboard
# ---------------------------------------------------------------------------
import plotly.graph_objects as go

# Supplied by the tracker library cell, which always runs first in the notebook.
# The fallbacks keep this file readable and runnable on its own.
try:
    THEME, ENGINE_COLOR, ENGINES
except NameError:
    THEME = dict(BG="#0B0E14", PANEL="#151B26", GRID="#2C3644", TXT="#F2F6FC",
                 MUTED="#95A3B8", GREEN="#25D07A", RED="#FF5B5B", AMBER="#FFC44D",
                 BLUE="#4DB6FF", PURPLE="#C58CFF", TEAL="#2FD9C6")
    ENGINE_COLOR = {"IV mean-reversion": "#4DB6FF", "Variance risk premium": "#FFC44D",
                    "Vol dispersion (pairs)": "#2FD9C6", "Correlation RV": "#C58CFF",
                    "Lead-lag catch-up": "#25D07A"}
    ENGINES = tuple(ENGINE_COLOR)

FONT = "Inter, Segoe UI, Arial"


def _layout(fig, T, title, sub="", height=420, **kw):
    fig.update_layout(
        title=dict(text="<b>%s</b>%s" % (title, "<br><span style='font-size:12px;color:%s'>%s</span>"
                                         % (T["MUTED"], sub) if sub else ""),
                   font=dict(size=17, color=T["TXT"], family=FONT), x=0, xanchor="left"),
        paper_bgcolor=T["BG"], plot_bgcolor=T["BG"], height=height,
        font=dict(family=FONT, color=T["MUTED"], size=12),
        margin=dict(l=60, r=30, t=70 if sub else 55, b=45),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=T["MUTED"])), **kw)
    fig.update_xaxes(gridcolor=T["GRID"], zerolinecolor=T["GRID"], linecolor=T["GRID"])
    fig.update_yaxes(gridcolor=T["GRID"], zerolinecolor=T["GRID"], linecolor=T["GRID"])
    return fig


def _err(tab):
    return dict(type="data", symmetric=False,
                array=(tab["hi"] - tab["hit"]).tolist(),
                arrayminus=(tab["hit"] - tab["lo"]).tolist(),
                color=THEME["MUTED"], thickness=1.2, width=4)


def fig_calibration(marks, horizon, theme=None):
    """The headline question: do higher-confidence signals actually win more?"""
    T = dict(THEME, **(theme or {}))
    cal = calibration(marks, horizon)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cal["conf_bin"].astype(str), y=cal["hit"], error_y=_err(cal),
        marker_color=[T["GREEN"] if l > 0 else T["RED"] for l in cal["lift"].fillna(0)],
        text=["n=%d" % n for n in cal["n"]], textposition="inside",
        insidetextanchor="start",           # sits at the foot of the bar, clear of the whiskers
        textfont=dict(color=T["BG"], size=11), name="realized hit rate",
        hovertemplate="confidence %{x}<br>hit %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=cal["conf_bin"].astype(str), y=cal["baseline"], mode="markers",
        marker=dict(symbol="diamond", size=11, color=T["TXT"],
                    line=dict(color=T["BG"], width=1)),
        name="baseline (same trades, any date)",
        hovertemplate="baseline %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=cal["conf_bin"].astype(str), y=cal["conf"], mode="lines+markers",
        line=dict(color=T["MUTED"], dash="dot", width=1.5), marker=dict(size=6),
        name="what the score promised",
        hovertemplate="promised %{y:.1f}%<extra></extra>"))
    _layout(fig, T, "Does the confidence score predict anything?",
            "Bars are what actually happened, with 95%% Wilson intervals. Diamonds are the same "
            "trades taken on every date — the bar has to beat the diamond to be an edge. "
            "%d-session hold." % horizon, height=440,
            barmode="group", yaxis_title="win rate  (%)")
    return fig


def fig_engines(marks, horizon, theme=None):
    """Per-engine verdict: hit rate against its own baseline, with intervals."""
    T = dict(THEME, **(theme or {}))
    tab = table(marks, "engine", horizon=horizon,
                order=[e for e in ENGINES if e in set(marks["engine"])])
    tab = tab[tab["n"] > 0].sort_values("lift")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=tab["engine"], x=tab["hit"], orientation="h",
        marker_color=[ENGINE_COLOR.get(e, T["BLUE"]) for e in tab["engine"]],
        error_x=dict(type="data", symmetric=False,
                     array=(tab["hi"] - tab["hit"]).tolist(),
                     arrayminus=(tab["hit"] - tab["lo"]).tolist(),
                     color=T["MUTED"], thickness=1.2, width=4),
        showlegend=False,                   # the colours already name the engines
        hovertemplate="%{y}<br>hit %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        y=tab["engine"], x=tab["baseline"], mode="markers",
        marker=dict(symbol="diamond", size=12, color=T["TXT"],
                    line=dict(color=T["BG"], width=1)),
        name="baseline: same trades, any date",
        hovertemplate="baseline %{x:.1f}%<extra></extra>"))

    # Numbers in a right-hand column rather than trailing each bar, so they line
    # up with each other instead of with the whiskers.
    label_x = 128
    for _, r in tab.iterrows():
        sig = ""
        if np.isfinite(r["p"]) and r["n"] >= 30:
            sig = "  p=%.3f" % r["p"] if r["lift"] > 0 else "  p=%.3f" % r["p_low"]
        fig.add_annotation(
            x=label_x, y=r["engine"], xanchor="right", showarrow=False,
            text="<b style='color:%s'>%+.1f pts</b>   %.0f%% vs %.0f%%   n=%d%s"
                 % (T["GREEN"] if r["lift"] > 0 else T["RED"], r["lift"], r["hit"],
                    r["baseline"], r["n"], sig),
            font=dict(color=T["MUTED"], size=11), bgcolor=T["BG"])
    _layout(fig, T, "Which engines beat doing nothing?",
            "Bar past the diamond = genuine edge. Bar short of it = the engine is riding a drift "
            "that was there anyway. %d-session hold." % horizon,
            height=380, xaxis_title="win rate  (%)")
    fig.update_xaxes(range=[0, label_x + 2], tickvals=[0, 20, 40, 60, 80, 100])
    fig.update_yaxes(showgrid=False)        # per-category lines read as strikethrough
    return fig


def fig_equity(marks, horizon, theme=None):
    """Cumulative net wins through the year — is the edge steady or one lucky patch?"""
    T = dict(THEME, **(theme or {}))
    eq = equity(marks, horizon)
    fig = go.Figure()
    for col in eq.columns:
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq[col], mode="lines", name=col,
            line=dict(color=ENGINE_COLOR.get(col, T["BLUE"]), width=2),
            hovertemplate="%{x|%d %b %Y}<br>" + col + " %{y:+.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T["MUTED"], width=1, dash="dot"))
    _layout(fig, T, "Is the edge steady, or one good month?",
            "Running total of wins minus losses. Unit-free, so engines marked in vol points and "
            "in percent sit on the same axis. Flat = coin flip. %d-session hold." % horizon,
            height=420, yaxis_title="cumulative wins − losses")
    return fig


def fig_horizons(marks, theme=None):
    """Does the edge survive holding longer?"""
    T = dict(THEME, **(theme or {}))
    hs = sorted(marks["horizon"].unique())
    engines = [e for e in ENGINES if e in set(marks["engine"])]
    fig = go.Figure()
    for h in hs:
        tab = table(marks, "engine", horizon=h, order=engines)
        fig.add_trace(go.Bar(
            x=tab["engine"], y=tab["lift"], name="%d sessions" % h,
            marker_color={hs[0]: T["BLUE"], hs[-1]: T["PURPLE"]}.get(h, T["TEAL"]),
            hovertemplate="%{x}<br>%{y:+.1f} pts vs baseline<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T["MUTED"], width=1))
    _layout(fig, T, "Does the edge survive a longer hold?",
            "Win rate minus baseline, in percentage points. Above zero is edge; the bars should "
            "not depend on how long you hold if the signal is real.",
            height=380, barmode="group", yaxis_title="edge over baseline  (pts)")
    return fig


# ---------------------------------------------------------------------------
# HTML summary
# ---------------------------------------------------------------------------
def cards_html(marks, horizon, theme=None):
    T = dict(THEME, **(theme or {}))
    h = headline(marks, horizon)
    eng = table(marks, "engine", horizon=horizon).sort_values("lift", ascending=False)
    best = eng.iloc[0] if len(eng) else None

    def col(v, good=0.0):
        if v is None or not np.isfinite(v):
            return T["MUTED"]
        return T["GREEN"] if v > good else (T["RED"] if v < good else T["AMBER"])

    cards = [
        ("Trades marked", "%d" % h["trades"],
         "%d replay dates · %d assets" % (h["dates"], h["assets"]), T["BLUE"]),
        ("Hit rate", "%.1f%%" % h["hit"], "95%% CI %.0f–%.0f%%" % (h["lo"], h["hi"]),
         col(h["hit"] - 50)),
        ("Baseline", "%.1f%%" % h["baseline"], "same trades, any date", T["MUTED"]),
        ("Edge", "%+.1f pts" % h["lift"],
         "p = %.3f" % h["p"] if np.isfinite(h["p"]) else "—", col(h["lift"])),
        ("Best engine", (best["engine"].split(" (")[0] if best is not None else "—"),
         ("%+.1f pts, n=%d" % (best["lift"], best["n"])) if best is not None else "",
         col(best["lift"] if best is not None else np.nan)),
    ]
    return "".join(
        "<div style='background:%s;border:1px solid %s;border-radius:10px;padding:11px 15px;"
        "min-width:145px'><div style='font:700 10px %s;color:%s;letter-spacing:.08em'>%s</div>"
        "<div style='font:800 21px %s;color:%s;padding:2px 0'>%s</div>"
        "<div style='font:400 11px %s;color:%s'>%s</div></div>"
        % (T["PANEL"], T["GRID"], FONT, T["MUTED"], k.upper(), FONT, c, v, FONT, T["MUTED"], sub)
        for k, v, sub, c in cards)


def report_html(marks, horizon, cfg=None, theme=None):
    """Header, cards and the auto-written takeaways — everything but the charts."""
    T = dict(THEME, **(theme or {}))
    h = headline(marks, horizon)
    points = "".join(
        "<li style='margin:5px 0;line-height:1.55'>%s</li>" % p
        for p in takeaways(marks, horizon))
    setting = ""
    if cfg:
        setting = ("<div style='font:400 11px %s;color:%s;padding-top:8px'>Dashboard settings "
                   "replayed: %s percentiles · %s realized vol over %dd · cheap ≤ %dth, rich ≥ "
                   "%dth pctile · vol quality ≥ %s</div>"
                   % (FONT, T["MUTED"], cfg.get("lb_name"), cfg.get("rv_estimator"),
                      cfg.get("rv_window"), cfg.get("iv_lo"), cfg.get("iv_hi"),
                      cfg.get("min_quality")))
    return (
        "<div style='background:%s;padding:18px;border-radius:12px'>"
        "<div style='font:800 24px %s;color:%s'>Do these signals actually work?</div>"
        "<div style='font:400 12px %s;color:%s;padding:4px 0 12px'>"
        "%s replays between %s and %s, every signal recorded, held %d sessions, no sizing.</div>"
        "<div style='display:flex;gap:10px;flex-wrap:wrap'>%s</div>"
        "<div style='font:700 11px %s;color:%s;letter-spacing:.08em;margin:18px 0 4px'>"
        "KEY TAKEAWAYS</div>"
        "<ul style='font:400 13px %s;color:%s;margin:0;padding-left:18px'>%s</ul>%s</div>"
        % (T["BG"], FONT, T["TXT"], FONT, T["MUTED"], h["dates"],
           pd.Timestamp(h["start"]).date(), pd.Timestamp(h["end"]).date(), horizon,
           cards_html(marks, horizon, theme), FONT, T["MUTED"], FONT, T["TXT"], points, setting))


def table_html(marks, horizon, theme=None):
    """Per-engine numbers as a table, for the slide that needs figures not shapes."""
    T = dict(THEME, **(theme or {}))
    tab = table(marks, "engine", horizon=horizon,
                order=[e for e in ENGINES if e in set(marks["engine"])])
    tab = tab[tab["n"] > 0]
    head = ["Engine", "Trades", "Hit rate", "95% CI", "Baseline", "Edge", "p", "Avg P&L", "Avg conf"]
    th = "".join("<th style='text-align:%s;padding:7px 11px;font:700 10px %s;color:%s;"
                 "letter-spacing:.07em'>%s</th>"
                 % ("left" if i == 0 else "right", FONT, T["MUTED"], c.upper())
                 for i, c in enumerate(head))
    rows = ""
    for _, r in tab.iterrows():
        lift_c = T["GREEN"] if r["lift"] > 0 else T["RED"]
        cells = [
            ("<span style='color:%s'>%s</span>" % (ENGINE_COLOR.get(r["engine"], T["TXT"]),
                                                   r["engine"]), "left"),
            ("%d" % r["n"], "right"),
            ("<b>%.1f%%</b>" % r["hit"], "right"),
            ("%.0f–%.0f" % (r["lo"], r["hi"]), "right"),
            ("%.1f%%" % r["baseline"] if np.isfinite(r["baseline"]) else "—", "right"),
            ("<b style='color:%s'>%+.1f</b>" % (lift_c, r["lift"])
             if np.isfinite(r["lift"]) else "—", "right"),
            ("%.3f" % r["p"] if np.isfinite(r["p"]) else "—", "right"),
            ("%+.2f %s" % (r["pnl"], r["unit"]) if np.isfinite(r["pnl"]) else "—", "right"),
            ("%.0f%%" % r["conf"], "right")]
        rows += "<tr style='border-top:1px solid %s'>%s</tr>" % (T["GRID"], "".join(
            "<td style='text-align:%s;padding:8px 11px;font:400 12px %s;color:%s'>%s</td>"
            % (al, FONT, T["TXT"], v) for v, al in cells))
    return ("<table style='border-collapse:collapse;width:100%%;background:%s;border:1px solid %s;"
            "border-radius:10px;overflow:hidden'><tr style='background:%s'>%s</tr>%s</table>"
            % (T["PANEL"], T["GRID"], T["BG"], th, rows))
