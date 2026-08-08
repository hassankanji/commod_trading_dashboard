"""
Point-in-time replay
====================

The half of the backtest that reproduces history: rewind the data to a past
date, ask the dashboard what it would have said, and work out what each of
those calls was actually worth. The statistics that judge the results live in
backtest_source.py.

Nothing here changes the dashboard — it *reuses* its engines by exec'ing the
notebook's non-UI code cells, so the signals being scored are the ones the
dashboard itself prints, not a reimplementation that could quietly diverge.

How the replay works
--------------------
Every engine in `build_signals` reads the LAST row of each series. So handing
it price/IV frames truncated at date T reproduces exactly what the dashboard
would have shown on T — including the confidence hit-rates, which are computed
from `shift(-h)` forward windows and therefore only ever see data inside the
truncated frame. There is no look-ahead in the replay.

    signals_asof(T)  ->  annotate  ->  mark forward over the next N sessions

`annotate` resolves each signal's legs: the engines describe a trade only as
display text ("SELL Gold VOL / BUY Silver VOL"), so the text is parsed back
into tickers and position signs. A signal whose legs cannot be resolved is
marked N.A. rather than guessed at.

Outcome definitions, per engine
-------------------------------
Each engine bets on a different thing, so each gets the P&L that matches the
bet. All are marked from the entry close to the exit close, no costs, no
sizing — this measures signal direction, not a tradable P&L.

    IV mean-reversion   BUY VOL wins if IV rose;  SELL VOL wins if IV fell.
                        P&L = ±(IV_exit - IV_entry), in vol points.

    Variance risk prem. The real test: did realized vol over the hold come in
                        under the implied vol quoted at entry?
                        P&L = ±(IV_entry - RV_realized_fwd), in vol points.
                        (SELL VOL is long that spread, BUY VOL is short it.)

    Vol dispersion      Short the rich leg's vol, long the cheap leg's.
                        P&L = (ΔIV_cheap - ΔIV_rich), in vol points — positive
                        when the spread converged.

    Correlation RV      Long the laggard, short the outperformer, equal notional.
                        P&L = ret(long) - ret(short), in %. Positive = gap closed.

    Lead-lag catch-up   Directional in the follower.
                        P&L = ±ret(follower), in %.

Holding periods are counted in trading sessions, not calendar days, so a
holiday cannot silently shorten a hold.
"""

import json
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HORIZON_DAYS = 5                                     # one trading week

# Same palette as the dashboard so the report drops into the same environment
# without looking like a different tool. Overridden by the host namespace when
# one is supplied (see Tracker.__init__).
THEME = dict(BG="#0B0E14", PANEL="#151B26", GRID="#2C3644", TXT="#F2F6FC",
             MUTED="#95A3B8", GREEN="#25D07A", RED="#FF5B5B", AMBER="#FFC44D",
             BLUE="#4DB6FF", PURPLE="#C58CFF", TEAL="#2FD9C6")

ENGINE_COLOR = {"IV mean-reversion": "#4DB6FF", "Variance risk premium": "#FFC44D",
                "Vol dispersion (pairs)": "#2FD9C6", "Correlation RV": "#C58CFF",
                "Lead-lag catch-up": "#25D07A"}

# The dashboard's five engines, in its own display order.
ENGINES = tuple(ENGINE_COLOR)

# Markers identifying the dashboard cells worth importing. The RENDERERS and
# CONTROLS cells are skipped on purpose — they build widgets and fire a
# Bloomberg pull on import.
DASHBOARD_SECTIONS = ("CONFIG —", "ANALYTICS —", "DATA LAYER —")

# A cell containing any of these builds or wires the UI: never exec it, whatever
# its header says. `on_click` in particular is what triggers the Bloomberg pull.
UI_MARKERS = ("widgets.Tab(", ".on_click(", ".observe(")

# ...and these identify the cells we do want even if their headers were renamed.
ENGINE_MARKERS = ("def build_signals", "bq = bql.Service()", "def fetch_all")


def _wanted_cell(src, sections):
    if any(u in src for u in UI_MARKERS):
        return False
    return any(s in src[:400] for s in sections) or any(k in src for k in ENGINE_MARKERS)


def load_dashboard(path, ns=None, sections=DASHBOARD_SECTIONS):
    """Exec the dashboard notebook's non-UI code cells into a namespace dict."""
    with open(path) as fh:
        nb = json.load(fh)
    ns = {} if ns is None else ns
    loaded = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if not _wanted_cell(src, sections):
            continue
        exec(compile(src, "%s#cell%d" % (os.path.basename(path), i), "exec"), ns)
        loaded.append(i)
    if "build_signals" not in ns:
        raise RuntimeError(
            "%s has no cell defining build_signals — is that the dashboard notebook?" % path)
    ns.setdefault("__loaded_cells__", loaded)
    return ns


def _defines_engines(path):
    try:
        with open(path) as fh:
            return "def build_signals" in fh.read()
    except (OSError, UnicodeDecodeError, ValueError):
        return False


def find_dashboard(preferred="commodities_vol_rv_dashboard.ipynb", extra_dirs=()):
    """Locate the dashboard notebook on disk.

    BQuant does not guarantee which directory a notebook's kernel starts in, so
    rather than assuming a path this looks for any .ipynb that actually defines
    build_signals — which survives the file being renamed or moved a folder up.
    Searched: the working directory, its parents, the home directory, and one
    level of subfolders under each.
    """
    cwd = os.getcwd()
    roots, p = [cwd] + list(extra_dirs) + [os.path.expanduser("~")], cwd
    for _ in range(3):
        p = os.path.dirname(p) or os.sep
        roots.append(p)

    searched, hits = [], []
    for root in roots:
        if not root or not os.path.isdir(root) or root in searched:
            continue
        searched.append(root)
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        dirs = [os.path.join(root, e) for e in entries
                if not e.startswith(".") and os.path.isdir(os.path.join(root, e))]
        for folder in [root] + dirs[:40]:
            try:
                names = sorted(os.listdir(folder))
            except OSError:
                continue
            for fn in names:
                if fn.endswith(".ipynb") and _defines_engines(os.path.join(folder, fn)):
                    hits.append(os.path.join(folder, fn))

    if not hits:
        raise FileNotFoundError(
            "Could not find the dashboard notebook (no .ipynb defining build_signals).\n"
            "Searched: %s\n"
            "Fix: put the dashboard notebook in the same folder as this one, or load it "
            "explicitly with  NS = load_dashboard('/full/path/to/dashboard.ipynb')"
            % ", ".join(searched))
    for h in hits:
        if os.path.basename(h) == preferred:
            return h
    return hits[0]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class Tracker:
    """Replays the dashboard's signal engines as of a past date and scores them.

    `ns` is the namespace produced by load_dashboard() (or the dashboard
    notebook's own globals() if you are running inside it).
    """

    def __init__(self, ns, horizon=HORIZON_DAYS):
        missing = [k for k in ("build_signals", "realized_vol", "NAME", "ASSET_CLASS")
                   if k not in ns]
        if missing:
            raise RuntimeError("namespace is missing %s — load the dashboard's "
                               "CONFIG and ANALYTICS cells first" % ", ".join(missing))
        self.ns = ns
        self.horizon = int(horizon)
        self.NAME = ns["NAME"]
        self.ASSET_CLASS = ns["ASSET_CLASS"]
        self.build_signals = ns["build_signals"]
        self.realized_vol = ns["realized_vol"]
        self.by_name = {v: k for k, v in self.NAME.items()}
        self.theme = {k: ns.get(k, v) for k, v in THEME.items()}

    # -- dates -------------------------------------------------------------
    def _pos(self, index, when):
        """Index position of the last session on or before `when` (-1 if none)."""
        return int(index.searchsorted(pd.Timestamp(when), side="right")) - 1

    def entry_exit(self, close, asof, horizon=None):
        """(entry_date, exit_date, sessions_held). exit is None if unseasoned."""
        h = self.horizon if horizon is None else int(horizon)
        idx = close.index
        i = self._pos(idx, asof)
        if i < 0:
            raise ValueError("no price history on or before %s" % asof)
        j = min(i + h, len(idx) - 1)
        held = j - i
        return idx[i], (idx[j] if held > 0 else None), held

    # -- replay ------------------------------------------------------------
    def signals_asof(self, px, iv, cfg, asof):
        """Every signal the dashboard would have printed on `asof`, one per row."""
        px_a = {k: v.loc[:pd.Timestamp(asof)] for k, v in px.items()}
        iv_a = iv.loc[:pd.Timestamp(asof)]
        res = self.build_signals(px_a, iv_a, cfg)
        frames = []
        for engine, df in res["engines"].items():
            if df is None or df.empty:
                continue
            d = df.copy()
            d["engine"] = engine
            frames.append(d)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True, sort=False)
        return out.sort_values(["conf", "score"], ascending=[False, False]).reset_index(drop=True)

    # -- leg resolution ----------------------------------------------------
    def _tk(self, nm):
        return self.by_name.get((nm or "").strip())

    def legs(self, engine, name, side):
        """-> (kind, [(ticker, signed weight), ...]).

        Signs are position signs: +1 long (long vol / long the asset), -1 short.
        The engines emit their legs only as display text, so this parses the
        `side` string they build; an unparseable row is returned as (None, [])
        and scored as N.A. rather than guessed at.
        """
        side = (side or "").strip()
        if engine in ("IV mean-reversion", "Variance risk premium"):
            tk = self._tk(name)
            sgn = 1.0 if side.upper().startswith("BUY") else -1.0
            return ("vol_single", [(tk, sgn)]) if tk else (None, [])
        if engine == "Vol dispersion (pairs)":
            m = re.match(r"^SELL (.+?) VOL / BUY (.+?) VOL$", side)
            if not m:
                return (None, [])
            rich, cheap = self._tk(m.group(1)), self._tk(m.group(2))
            return ("vol_pair", [(rich, -1.0), (cheap, 1.0)]) if rich and cheap else (None, [])
        if engine == "Correlation RV":
            m = re.match(r"^BUY (.+?) / SELL (.+?)$", side)
            if not m:
                return (None, [])
            lng, sht = self._tk(m.group(1)), self._tk(m.group(2))
            return ("px_pair", [(lng, 1.0), (sht, -1.0)]) if lng and sht else (None, [])
        if engine == "Lead-lag catch-up":
            m = re.match(r"^(BUY|SELL) (.+?)$", side)
            if not m:
                return (None, [])
            tk = self._tk(m.group(2))
            sgn = 1.0 if m.group(1) == "BUY" else -1.0
            return ("px_single", [(tk, sgn)]) if tk else (None, [])
        return (None, [])

    def classes(self, legs):
        return [self.ASSET_CLASS.get(tk, "Other") for tk, _ in legs]

    # -- selection ---------------------------------------------------------
    # -- scoring -----------------------------------------------------------
    def annotate(self, sig):
        """Attach legs/kind/sectors to every signal, with no selection applied.

        `select` picks a handful of trades to follow; the backtest wants the
        whole population, so this is the same annotation without the filtering.
        """
        if sig is None or sig.empty:
            return pd.DataFrame()
        rows = []
        for _, r in sig.iterrows():
            kind, legs = self.legs(r["engine"], r["name"], r["side"])
            d = dict(r)
            d.update(basis="all signals", kind=kind, legs=legs, sectors=self.classes(legs))
            rows.append(d)
        return pd.DataFrame(rows).reset_index(drop=True)

    def realized_by_horizon(self, px, cfg, horizons):
        """{h: realized-vol frame} — the vol delivered over each trailing h days.

        Read at a trade's exit date this is the vol delivered while the trade
        was on. It depends only on the price history and the horizon, never on
        the as-of date, so the backtest computes it once instead of once per
        replay.
        """
        return {int(h): self.realized_vol(px, cfg["rv_estimator"], max(2, int(h)))
                for h in horizons}

    def score(self, trades, px, iv, cfg, asof, horizon=None):
        """Mark every selected trade from entry close to exit close."""
        if trades is None or trades.empty:
            return pd.DataFrame()
        h = self.horizon if horizon is None else int(horizon)
        close = px["close"]
        entry, exit_, held = self.entry_exit(close, asof, h)
        ivf = iv.reindex(close.index).ffill()

        # Realized vol measured over a window exactly as long as the holding
        # period: its value AT the exit date is the vol actually delivered
        # between entry and exit — the number a vol seller is marked against.
        rv_fwd = self.realized_vol(px, cfg["rv_estimator"], max(2, held)) if held else None
        return self.mark(trades, close, ivf, rv_fwd, entry, exit_, held)

    def mark(self, trades, close, ivf, rv_fwd, entry, exit_, held):
        """Mark annotated trades between two dates. Frames are passed in so a
        backtest can prepare them once and reuse them across every replay."""
        out = []

        def lvl(frame, tk, when):
            try:
                v = frame.at[when, tk]
            except (KeyError, IndexError):
                return np.nan
            return float(v) if pd.notna(v) else np.nan

        for _, t in trades.iterrows():
            d = dict(t)
            d.update(asof=entry, entry_date=entry, exit_date=exit_, sessions=held,
                     pnl=np.nan, unit="", outcome="N.A.", detail="", note="")
            legs, kind = t["legs"], t["kind"]

            if not legs or kind is None:
                d["note"] = "could not resolve the trade's legs from the signal text"
                out.append(d); continue
            if exit_ is None:
                d["outcome"] = "OPEN"
                d["note"] = "no sessions after %s yet" % pd.Timestamp(entry).date()
                out.append(d); continue

            if kind in ("vol_single", "vol_pair"):
                d["unit"] = "vol pts"
                parts, pnl, bad = [], 0.0, False
                for tk, sgn in legs:
                    iv0, iv1 = lvl(ivf, tk, entry), lvl(ivf, tk, exit_)
                    if np.isnan(iv0) or np.isnan(iv1):
                        bad = True
                        break
                    if t["engine"] == "Variance risk premium":
                        # marked against delivered vol, not against the IV re-mark
                        rvf = lvl(rv_fwd, tk, exit_)
                        if np.isnan(rvf):
                            bad = True
                            break
                        pnl += sgn * (rvf - iv0)
                        parts.append("%s: IV %.1f at entry vs %.1f realized over the hold"
                                     % (self.NAME.get(tk, tk), iv0, rvf))
                    else:
                        pnl += sgn * (iv1 - iv0)
                        parts.append("%s IV %.1f → %.1f (%+.1f)"
                                     % (self.NAME.get(tk, tk), iv0, iv1, iv1 - iv0))
                if bad:
                    d["note"] = "implied vol missing at entry or exit"
                    out.append(d); continue
                d["pnl"], d["detail"] = pnl, "; ".join(parts)

            else:  # price legs
                d["unit"] = "%"
                parts, pnl, bad = [], 0.0, False
                for tk, sgn in legs:
                    p0, p1 = lvl(close, tk, entry), lvl(close, tk, exit_)
                    if np.isnan(p0) or np.isnan(p1) or p0 == 0:
                        bad = True
                        break
                    r = (p1 / p0 - 1.0) * 100.0
                    pnl += sgn * r
                    parts.append("%s %s %.2f → %.2f (%+.2f%%)"
                                 % ("long" if sgn > 0 else "short",
                                    self.NAME.get(tk, tk), p0, p1, r))
                if bad:
                    d["note"] = "price missing at entry or exit"
                    out.append(d); continue
                d["pnl"], d["detail"] = pnl, "; ".join(parts)

            d["outcome"] = "WIN" if d["pnl"] > 1e-9 else ("LOSS" if d["pnl"] < -1e-9 else "FLAT")
            out.append(d)

        res = pd.DataFrame(out)
        return res
