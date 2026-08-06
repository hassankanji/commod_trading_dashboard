"""
Trade-signal outcome tracker
============================

Deliberately separate from the dashboard. The dashboard answers "what looks
like a trade right now"; this answers "did the trades it flagged actually
work". Nothing in here changes the dashboard — it *reuses* its engines by
exec'ing the notebook's non-UI code cells, so the signals being scored are
byte-identical to the ones the dashboard prints.

How the replay works
--------------------
Every engine in `build_signals` reads the LAST row of each series. So handing
it price/IV frames truncated at date T reproduces exactly what the dashboard
would have shown on T — including the confidence hit-rates, which are computed
from `shift(-h)` forward windows and therefore only ever see data inside the
truncated frame. There is no look-ahead in the replay.

    signals_asof(T)  ->  pick trades  ->  score them over the next N sessions

Outcome definitions, per engine
-------------------------------
Each engine bets on a different thing, so each gets the P&L that matches the
bet. All are marked from the entry close to the exit close, no costs, no
sizing — this measures signal direction, not a tradable P&L.

    IV mean-reversion   BUY VOL wins if IV rose;  SELL VOL wins if IV fell.
                        P&L = ±(IV_exit - IV_entry), in vol points.

    Variance risk prem. The real test: did realized vol over the holding week
                        come in under the implied vol quoted at entry?
                        P&L = ±(IV_entry - RV_realized_fwd), in vol points.
                        (SELL VOL is long that spread, BUY VOL is short it.)

    Vol dispersion      Short the rich leg's vol, long the cheap leg's.
                        P&L = (ΔIV_cheap - ΔIV_rich), in vol points — positive
                        when the spread converged.

    Correlation RV      Long the laggard, short the outperformer, equal notional.
                        P&L = ret(long) - ret(short), in %. Positive = gap closed.

    Lead-lag catch-up   Directional in the follower.
                        P&L = ±ret(follower), in %.

A "week" is HORIZON_DAYS trading sessions, not calendar days, so holidays
don't silently shorten the holding period.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HORIZON_DAYS = 5                                     # one trading week
TOP_N = 5                                            # highest-confidence picks
SECTORS = ("Energy", "Precious Metals", "Base Metals")
SECTOR_N = 3                                         # metals/energy floor

VOL_ENGINES = ("IV mean-reversion", "Variance risk premium", "Vol dispersion (pairs)")

# Same palette as the dashboard so the report drops into the same environment
# without looking like a different tool. Overridden by the host namespace when
# one is supplied (see Tracker.__init__).
THEME = dict(BG="#0B0E14", PANEL="#151B26", GRID="#2C3644", TXT="#F2F6FC",
             MUTED="#95A3B8", GREEN="#25D07A", RED="#FF5B5B", AMBER="#FFC44D",
             BLUE="#4DB6FF", PURPLE="#C58CFF", TEAL="#2FD9C6")

ENGINE_COLOR = {"IV mean-reversion": "#4DB6FF", "Variance risk premium": "#FFC44D",
                "Vol dispersion (pairs)": "#2FD9C6", "Correlation RV": "#C58CFF",
                "Lead-lag catch-up": "#25D07A"}

# Markers identifying the dashboard cells worth importing. The RENDERERS and
# CONTROLS cells are skipped on purpose — they build widgets and fire a
# Bloomberg pull on import.
DASHBOARD_SECTIONS = ("CONFIG —", "ANALYTICS —", "DATA LAYER —")


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
        if not any(s in src[:400] for s in sections):
            continue
        exec(compile(src, "%s#cell%d" % (os.path.basename(path), i), "exec"), ns)
        loaded.append(i)
    if not loaded:
        raise RuntimeError("No matching code cells found in %s" % path)
    ns.setdefault("__loaded_cells__", loaded)
    return ns


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
    def select(self, sig, top_n=TOP_N, sectors=SECTORS, sector_n=SECTOR_N):
        """Top `top_n` by confidence, then top up so >= `sector_n` touch metals/energy.

        Deduped on the set of underlyings: the same asset flagged by two engines
        is one bet, not two, so only its higher-confidence version is tracked.
        Sector top-ups are taken in confidence order *within* metals/energy —
        they are included even when their confidence is poor, which is the
        point: a coin-flip signal that the desk would still look at deserves a
        recorded outcome.
        """
        if sig is None or sig.empty:
            return pd.DataFrame()
        rows, seen = [], set()

        def add(r, basis):
            kind, legs = self.legs(r["engine"], r["name"], r["side"])
            key = frozenset(tk for tk, _ in legs) if legs else ("?", r["engine"], r["name"])
            if key in seen:
                return False
            seen.add(key)
            d = dict(r)
            d.update(basis=basis, kind=kind, legs=legs, sectors=self.classes(legs))
            rows.append(d)
            return True

        for _, r in sig.iterrows():
            if len(rows) >= top_n:
                break
            add(r, "top confidence")

        def in_sector(d):
            return any(c in sectors for c in d["sectors"])

        for _, r in sig.iterrows():
            if sum(in_sector(d) for d in rows) >= sector_n:
                break
            kind, legs = self.legs(r["engine"], r["name"], r["side"])
            if not legs or not any(self.ASSET_CLASS.get(tk) in sectors for tk, _ in legs):
                continue
            add(r, "metals/energy quota")

        out = pd.DataFrame(rows)
        return out.reset_index(drop=True)

    # -- scoring -----------------------------------------------------------
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

        def lvl(frame, tk, when):
            try:
                v = frame.at[when, tk]
            except (KeyError, IndexError):
                return np.nan
            return float(v) if pd.notna(v) else np.nan

        out = []
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
                        parts.append("%s: IV %.1f at entry vs %.1f realized over the week"
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

    # -- one-shot ----------------------------------------------------------
    def run(self, px, iv, cfg, asof, horizon=None, top_n=TOP_N,
            sectors=SECTORS, sector_n=SECTOR_N):
        """Replay `asof`, pick the trades, score them. -> (results, summary)."""
        sig = self.signals_asof(px, iv, cfg, asof)
        picks = self.select(sig, top_n=top_n, sectors=sectors, sector_n=sector_n)
        res = self.score(picks, px, iv, cfg, asof, horizon)
        return res, summarize(res, sectors=sectors)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(res, sectors=SECTORS):
    """Hit rate overall / by engine / by basis, plus confidence calibration."""
    if res is None or res.empty:
        return dict(n=0, scored=0)
    live = res[res["outcome"].isin(["WIN", "LOSS", "FLAT"])]
    wins = int((live["outcome"] == "WIN").sum())
    n = len(live)

    def blk(df):
        if df.empty:
            return dict(n=0, wins=0, hit=np.nan, vol_pnl=np.nan, px_pnl=np.nan)
        v = df[df["unit"] == "vol pts"]["pnl"]
        p = df[df["unit"] == "%"]["pnl"]
        return dict(n=len(df), wins=int((df["outcome"] == "WIN").sum()),
                    hit=(df["outcome"] == "WIN").mean() * 100.0,
                    vol_pnl=v.mean() if len(v) else np.nan,
                    px_pnl=p.mean() if len(p) else np.nan)

    by_engine = {e: blk(live[live["engine"] == e]) for e in sorted(live["engine"].unique())}
    by_basis = {b: blk(live[live["basis"] == b]) for b in sorted(live["basis"].unique())}
    in_sec = live[live["sectors"].apply(lambda cs: any(c in sectors for c in cs))]

    s = dict(n=len(res), scored=n, wins=wins,
             hit=(wins / n * 100.0) if n else np.nan,
             avg_conf=float(live["conf"].mean()) if n else np.nan,
             entry=res["entry_date"].iloc[0], exit=res["exit_date"].iloc[0],
             sessions=int(res["sessions"].iloc[0]),
             by_engine=by_engine, by_basis=by_basis,
             sector=blk(in_sec), overall=blk(live),
             unscored=int(len(res) - n))
    s["calibration"] = (s["hit"] - s["avg_conf"]) if n else np.nan
    return s


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _fmt_pnl(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return "%+.2f %s" % (v, unit)


def report_text(res, summary, cfg=None):
    """Plain-text report — copy/pasteable, identical numbers to the HTML one."""
    if res is None or res.empty:
        return "No signals were open on the replay date, so there is nothing to score."
    L = []
    L.append("TRADE-SIGNAL OUTCOMES")
    L.append("entry %s  ->  exit %s   (%d trading sessions)"
             % (pd.Timestamp(summary["entry"]).date(),
                pd.Timestamp(summary["exit"]).date() if summary["exit"] is not None else "—",
                summary["sessions"]))
    if cfg:
        L.append("settings: %s RV %s / %dd, cheap<=%d rich>=%d, quality>=%s"
                 % (cfg.get("lb_name"), cfg.get("rv_estimator"), cfg.get("rv_window"),
                    cfg.get("iv_lo"), cfg.get("iv_hi"), cfg.get("min_quality")))
    L.append("")
    for _, r in res.iterrows():
        L.append("[%s] %s — %s" % (r["engine"], r["name"], r["side"]))
        L.append("    picked as: %s | confidence %.0f%% (n=%d) | trigger: %s"
                 % (r["basis"], r["conf"], int(r.get("conf_n") or 0), r["trigger"]))
        L.append("    outcome: %-5s  %s" % (r["outcome"], _fmt_pnl(r["pnl"], r["unit"])))
        if r["detail"]:
            L.append("    marks: %s" % r["detail"])
        if r["note"]:
            L.append("    note: %s" % r["note"])
        L.append("")
    o = summary["overall"]
    L.append("RESULTS")
    L.append("  hit rate            %s (%d of %d)"
             % ("%.0f%%" % summary["hit"] if summary["scored"] else "—",
                summary["wins"], summary["scored"]))
    L.append("  avg predicted conf  %.0f%%   -> calibration %+.0f pts"
             % (summary["avg_conf"], summary["calibration"]))
    if not np.isnan(o["vol_pnl"]):
        L.append("  avg vol-trade P&L   %+.2f vol pts" % o["vol_pnl"])
    if not np.isnan(o["px_pnl"]):
        L.append("  avg price-trade P&L %+.2f%%" % o["px_pnl"])
    L.append("")
    L.append("  by engine:")
    for e, b in summary["by_engine"].items():
        L.append("    %-24s %d/%d  %s" % (e, b["wins"], b["n"],
                 "%.0f%%" % b["hit"] if b["n"] else "—"))
    L.append("  by how it was picked:")
    for b_, b in summary["by_basis"].items():
        L.append("    %-24s %d/%d  %s" % (b_, b["wins"], b["n"],
                 "%.0f%%" % b["hit"] if b["n"] else "—"))
    sec = summary["sector"]
    L.append("  metals/energy only:      %d/%d  %s"
             % (sec["wins"], sec["n"], "%.0f%%" % sec["hit"] if sec["n"] else "—"))
    if summary["unscored"]:
        L.append("  %d trade(s) could not be marked — see notes above." % summary["unscored"])
    return "\n".join(L)


def report_html(res, summary, cfg=None, theme=None):
    """Dark-themed HTML report matching the dashboard's look."""
    T = dict(THEME, **(theme or {}))
    if res is None or res.empty:
        return ("<div style='font:400 13px Inter,Arial;color:%s'>No signals were open "
                "on the replay date, so there is nothing to score.</div>" % T["MUTED"])

    def oc_color(o):
        return {"WIN": T["GREEN"], "LOSS": T["RED"], "FLAT": T["AMBER"]}.get(o, T["MUTED"])

    def conf_color(c):
        if c is None or (isinstance(c, float) and np.isnan(c)):
            return T["MUTED"]
        return T["GREEN"] if c >= 65 else (T["TEAL"] if c >= 55 else
                                           (T["AMBER"] if c >= 45 else T["RED"]))

    head = ("<div style='font:800 22px Inter,Segoe UI,Arial;color:%s'>Trade-signal outcomes</div>"
            "<div style='font:400 12px Inter,Arial;color:%s;padding:3px 0 10px'>"
            "signals as they stood on <b style='color:%s'>%s</b>, marked to "
            "<b style='color:%s'>%s</b> — %d trading sessions. Entry-close to exit-close, "
            "no costs or sizing.</div>"
            % (T["TXT"], T["MUTED"], T["TXT"], pd.Timestamp(summary["entry"]).date(),
               T["TXT"], pd.Timestamp(summary["exit"]).date() if summary["exit"] is not None else "—",
               summary["sessions"]))

    def sign_color(v):
        """Green up, red down, muted when there were no trades of that kind."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return T["MUTED"]
        return T["GREEN"] if v > 0 else (T["RED"] if v < 0 else T["AMBER"])

    o = summary["overall"]
    cards = [("Hit rate", "%.0f%%" % summary["hit"] if summary["scored"] else "—",
              "%d of %d marked" % (summary["wins"], summary["scored"]),
              sign_color(summary["hit"] - 50 if summary["scored"] else np.nan)),
             ("Predicted", "%.0f%%" % summary["avg_conf"], "avg confidence at entry", T["BLUE"]),
             ("Calibration", "%+.0f pts" % summary["calibration"], "realized minus predicted",
              sign_color(summary["calibration"])),
             ("Vol trades", _fmt_pnl(o["vol_pnl"], "vol pts"), "average P&L",
              sign_color(o["vol_pnl"])),
             ("Price trades", _fmt_pnl(o["px_pnl"], "%"), "average P&L",
              sign_color(o["px_pnl"]))]
    card_html = "".join(
        "<div style='background:%s;border:1px solid %s;border-radius:10px;padding:10px 14px;"
        "min-width:130px'><div style='font:700 10px Inter,Arial;color:%s;letter-spacing:.08em'>%s</div>"
        "<div style='font:800 20px Inter,Arial;color:%s;padding:2px 0'>%s</div>"
        "<div style='font:400 11px Inter,Arial;color:%s'>%s</div></div>"
        % (T["PANEL"], T["GRID"], T["MUTED"], k.upper(), c, v, T["MUTED"], sub)
        for k, v, sub, c in cards)

    rows = ""
    for _, r in res.iterrows():
        ec = ENGINE_COLOR.get(r["engine"], T["MUTED"])
        badge = ("<span style='font:700 9px Inter,Arial;color:%s;border:1px solid %s;"
                 "border-radius:4px;padding:1px 5px'>%s</span>"
                 % (T["AMBER"], T["AMBER"], "QUOTA")) if r["basis"] != "top confidence" else ""
        rows += (
            "<tr style='border-top:1px solid %s'>"
            "<td style='padding:10px 12px;vertical-align:top;white-space:nowrap'>"
            "<div style='font:700 12px Inter,Arial;color:%s'>%s</div>"
            "<div style='font:400 10px Inter,Arial;color:%s;padding-top:2px'>%s %s</div></td>"
            "<td style='padding:10px 12px;vertical-align:top'>"
            "<div style='font:600 13px Inter,Arial;color:%s'>%s</div>"
            "<div style='font:400 11px Inter,Arial;color:%s;padding-top:3px'>%s</div>"
            "<div style='font:400 11px Inter,Arial;color:%s;padding-top:4px'>%s</div></td>"
            "<td style='padding:10px 12px;vertical-align:top;text-align:right;white-space:nowrap'>"
            "<div style='font:800 14px Inter,Arial;color:%s'>%.0f%%</div>"
            "<div style='font:400 10px Inter,Arial;color:%s'>n=%d</div></td>"
            "<td style='padding:10px 12px;vertical-align:top;text-align:right;white-space:nowrap'>"
            "<div style='font:800 14px Inter,Arial;color:%s'>%s</div>"
            "<div style='font:700 11px Inter,Arial;color:%s'>%s</div></td></tr>"
            % (T["GRID"], ec, r["engine"], T["MUTED"], r["basis"], badge,
               T["TXT"], "%s — %s" % (r["name"], r["side"]),
               T["MUTED"], r["trigger"],
               T["MUTED"], (r["detail"] or r["note"] or ""),
               conf_color(r["conf"]), r["conf"], T["MUTED"], int(r.get("conf_n") or 0),
               oc_color(r["outcome"]), r["outcome"],
               T["TXT"], _fmt_pnl(r["pnl"], r["unit"])))

    table = ("<table style='border-collapse:collapse;width:100%%;background:%s;"
             "border:1px solid %s;border-radius:10px;overflow:hidden;margin-top:12px'>"
             "<tr style='background:%s'>"
             "<th style='text-align:left;padding:8px 12px;font:700 10px Inter,Arial;color:%s;"
             "letter-spacing:.08em'>ENGINE</th>"
             "<th style='text-align:left;padding:8px 12px;font:700 10px Inter,Arial;color:%s;"
             "letter-spacing:.08em'>TRADE</th>"
             "<th style='text-align:right;padding:8px 12px;font:700 10px Inter,Arial;color:%s;"
             "letter-spacing:.08em'>CONF</th>"
             "<th style='text-align:right;padding:8px 12px;font:700 10px Inter,Arial;color:%s;"
             "letter-spacing:.08em'>OUTCOME</th></tr>%s</table>"
             % (T["PANEL"], T["GRID"], T["BG"], T["MUTED"], T["MUTED"], T["MUTED"],
                T["MUTED"], rows))

    def brk(title, mapping):
        items = "".join(
            "<div style='font:400 12px Inter,Arial;color:%s;padding:3px 0'>%s "
            "<b style='color:%s'>%d/%d</b> <span style='color:%s'>%s</span></div>"
            % (T["MUTED"], k, T["TXT"], b["wins"], b["n"],
               sign_color(b["hit"] - 50 if b["n"] else np.nan),
               "%.0f%%" % b["hit"] if b["n"] else "—")
            for k, b in mapping.items())
        return ("<div style='min-width:260px'><div style='font:700 10px Inter,Arial;color:%s;"
                "letter-spacing:.08em;padding-bottom:4px'>%s</div>%s</div>"
                % (T["MUTED"], title.upper(), items))

    sec = summary["sector"]
    breakdown = ("<div style='display:flex;gap:28px;flex-wrap:wrap;margin-top:14px'>%s%s%s</div>"
                 % (brk("By engine", summary["by_engine"]),
                    brk("By how it was picked", summary["by_basis"]),
                    brk("Sector", {"Metals / energy": sec})))

    foot = ("<div style='font:400 11px Inter,Arial;color:%s;margin-top:14px;line-height:1.6'>"
            "Vol trades are marked in vol points, relative-value and directional trades in "
            "percent, so the two averages are not additive. Variance-risk-premium trades are "
            "marked against the volatility actually delivered over the holding week, not against "
            "the implied-vol re-mark. One week of signals is a handful of observations — read "
            "the calibration line as a sanity check, not as evidence about the engines."
            "</div>" % T["MUTED"])

    return ("<div style='background:%s;padding:16px;border-radius:12px'>%s"
            "<div style='display:flex;gap:10px;flex-wrap:wrap'>%s</div>%s%s%s</div>"
            % (T["BG"], head, card_html, table, breakdown, foot))


# ---------------------------------------------------------------------------
# Journal — so outcomes accumulate instead of being recomputed each time
# ---------------------------------------------------------------------------
JOURNAL_DIR = "trade_journal"
JOURNAL_COLS = ["entry_date", "exit_date", "sessions", "engine", "name", "side", "basis",
                "cls", "conf", "conf_n", "conf_raw", "tier", "trigger", "outcome",
                "pnl", "unit", "detail", "note", "reason"]


def save_run(res, directory=JOURNAL_DIR):
    """Append this run to the journal and return the per-run file path."""
    if res is None or res.empty:
        return None
    os.makedirs(directory, exist_ok=True)
    d = res.copy()
    d["legs"] = d["legs"].apply(lambda ls: "|".join("%s%s" % ("+" if s > 0 else "-", t)
                                                    for t, s in (ls or [])))
    d["sectors"] = d["sectors"].apply(lambda cs: "|".join(cs or []))
    cols = [c for c in JOURNAL_COLS if c in d.columns] + ["legs", "sectors"]
    d = d[cols]
    stamp = pd.Timestamp(res["entry_date"].iloc[0]).date()
    path = os.path.join(directory, "run_%s.csv" % stamp)
    d.to_csv(path, index=False)

    # Re-running the same as-of date replaces that date's rows rather than
    # appending a second copy, so the ledger stays one row per tracked trade.
    ledger = os.path.join(directory, "ledger.csv")
    if os.path.exists(ledger):
        old = pd.read_csv(ledger)
        if "entry_date" in old.columns:
            same = pd.to_datetime(old["entry_date"], errors="coerce").dt.date == stamp
            old = old[~same.fillna(False)]
        d = pd.concat([old, d], ignore_index=True, sort=False)
    d.to_csv(ledger, index=False)
    return path


def load_ledger(directory=JOURNAL_DIR):
    path = os.path.join(directory, "ledger.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
