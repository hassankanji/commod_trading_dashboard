"""Slide charts drawn from the real backtest output.

Numbers transcribed from presentation/NUMBERS.md: 1,970 signals, 52 weekly
replays over trading days, 2025-08-11 to 2026-08-10, five-day holds.
Standard Chartered Prosper palette on white, sized for a 13.33x7.5in slide.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY, BLUE, LBLUE, GREEN, LGREEN = "#020B43", "#0473EA", "#7BB6F5", "#38D200", "#92E773"
GREY, GRID, RED = "#525355", "#DCE1E7", "#D0021B"
FONT = "SC Prosper Sans, Arial, Helvetica, sans-serif"

# engine, trades, hit, ci_lo, ci_hi, baseline, edge, p, avg_conf
ENGINES = [
    ("Vol dispersion (pairs)", 192, 64.1, 57, 71, 48.2, 15.9, 0.0000, 60),
    ("Variance risk premium",  335, 59.4, 54, 65, 50.5,  8.9, 0.0007, 56),
    ("IV mean-reversion",      339, 52.5, 47, 58, 44.8,  7.7, 0.0027, 68),
    ("Correlation RV",         506, 53.0, 49, 57, 49.7,  3.3, 0.0741, 51),
    ("Lead-lag catch-up",      598, 50.3, 46, 54, 49.1,  1.2, 0.2890, 51),
]
COLOR = {"Vol dispersion (pairs)": NAVY, "Variance risk premium": BLUE,
         "IV mean-reversion": LBLUE, "Correlation RV": "#B9C2CC",
         "Lead-lag catch-up": "#B9C2CC"}

HOLD = [(1, 3.3), (2, 2.6), (3, 2.7), (5, 5.6), (8, 6.7), (13, 9.1), (21, 11.6)]

ASSETS_BEST = [("Cocoa", 45, 21.0), ("S&P 500", 61, 18.6), ("Soybean Meal", 74, 17.8),
               ("Copper", 52, 14.2), ("Palladium", 114, 11.9), ("Soybeans", 127, 11.8)]
ASSETS_WORST = [("US 10Y future", 131, -3.9), ("Natural Gas", 82, -1.3), ("Silver", 181, -0.4),
                ("Brent Crude", 175, 1.0), ("US Dollar", 79, 1.5), ("Sugar", 61, 1.7)]
LIQUIDITY = [("Options re-price<br>most days", 898, 11.9),
             ("Options re-price<br>rarely", 160, 6.2),
             ("Price engines<br>(no options)", 1610, 2.5)]

# engine -> {asset class: (edge, n)}, ordered as the notebook ordered it
GRID_ROWS = ["Vol dispersion (pairs)", "IV mean-reversion", "Variance risk premium",
             "Correlation RV", "Lead-lag catch-up"]
GRID_COLS = ["Livestock", "Softs", "Base Metals", "Grains / Oilseeds",
             "Precious Metals", "Energy", "Macro"]
ENGINE_GRID = {
    "Vol dispersion (pairs)": {"Softs": (11.0, 36), "Grains / Oilseeds": (22.0, 56),
                               "Precious Metals": (2.7, 47), "Energy": (23.1, 49)},
    "IV mean-reversion": {"Livestock": (39.0, 15), "Softs": (-3.7, 79), "Base Metals": (24.3, 16),
                          "Grains / Oilseeds": (14.4, 84), "Precious Metals": (4.6, 77),
                          "Energy": (5.3, 68)},
    "Variance risk premium": {"Livestock": (7.4, 30), "Softs": (27.5, 50), "Base Metals": (1.9, 22),
                              "Grains / Oilseeds": (5.1, 92), "Precious Metals": (3.9, 73),
                              "Energy": (8.6, 68)},
    "Correlation RV": {"Base Metals": (7.1, 39), "Grains / Oilseeds": (7.1, 114),
                       "Precious Metals": (1.2, 156), "Energy": (0.8, 188), "Macro": (-0.3, 128)},
    "Lead-lag catch-up": {"Livestock": (-8.3, 48), "Softs": (6.1, 58), "Base Metals": (7.6, 36),
                          "Grains / Oilseeds": (1.7, 131), "Precious Metals": (19.3, 62),
                          "Energy": (-11.2, 129), "Macro": (3.9, 134)},
}

CONF_BUCKETS = [("<45", 317, 9.7), ("45-55", 788, 4.4), ("55-65", 418, 3.2),
                ("65-75", 268, 9.3), ("75+", 179, 3.7)]
CONF_HALVES = [("Least confident half<br>of each engine", 1068, 6.7),
               ("Most confident half<br>of each engine", 902, 4.3)]


def base(fig, title, sub, height, margin=None):
    fig.update_layout(
        title=dict(text="<b>%s</b><br><span style='font-size:13px;color:%s'>%s</span>"
                        % (title, GREY, sub),
                   font=dict(size=20, color=NAVY, family=FONT), x=0, xanchor="left", y=0.94),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=height,
        font=dict(family=FONT, color=GREY, size=13), showlegend=False,
        margin=margin or dict(l=170, r=40, t=85, b=55))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    return fig


def fig_engines():
    d = sorted(ENGINES, key=lambda r: r[6])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[r[0] for r in d], x=[r[2] for r in d], orientation="h", width=0.6,
        marker_color=[COLOR[r[0]] for r in d],
        error_x=dict(type="data", symmetric=False,
                     array=[r[4] - r[2] for r in d], arrayminus=[r[2] - r[3] for r in d],
                     color=GREY, thickness=1.3, width=5)))
    fig.add_trace(go.Scatter(
        y=[r[0] for r in d], x=[r[5] for r in d], mode="markers",
        marker=dict(symbol="diamond", size=13, color="#FFFFFF", line=dict(color=NAVY, width=2))))
    for r in d:
        strong = r[7] < 0.05
        fig.add_annotation(x=104, y=r[0], xanchor="left", showarrow=False,
                           text="<b style='color:%s'>%+.1f pts</b>   n=%d   p=%.3f"
                                % (GREEN if strong else GREY, r[6], r[1], r[7]),
                           font=dict(color=GREY, size=12))
    base(fig, "Three of the five engines beat doing nothing",
         "Bars: win rate with 95% confidence interval. Diamonds: the same trades taken on every "
         "date, signal or not. Five-day hold, 1,970 trades.", 470,
         margin=dict(l=175, r=20, t=85, b=55))
    fig.update_xaxes(range=[0, 143], tickvals=[0, 20, 40, 60, 80, 100],
                     ticktext=["0", "20%", "40%", "60%", "80%", "100%"], title_text="win rate")
    fig.update_yaxes(showgrid=False)
    return fig


def fig_hold():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[h for h, _ in HOLD], y=[e for _, e in HOLD], mode="lines+markers",
        line=dict(color=BLUE, width=3), marker=dict(size=10, color=BLUE)))
    for h, e in HOLD:
        fig.add_annotation(x=h, y=e, text="%+.1f" % e, showarrow=False, yshift=18,
                           font=dict(color=NAVY, size=12))
    fig.add_vrect(x0=4.2, x1=5.8, fillcolor=LBLUE, opacity=0.18, line_width=0)
    fig.add_annotation(x=5, y=0.6, text="the one we report", showarrow=False,
                       font=dict(color=GREY, size=11))
    base(fig, "These trades take weeks, not days",
         "Edge over baseline by how long the position is held. Five days is the only holding "
         "period where consecutive trades do not overlap, so it is the conservative one to quote.",
         430, margin=dict(l=80, r=50, t=85, b=60))
    fig.update_xaxes(title_text="trading days held", tickvals=[h for h, _ in HOLD])
    fig.update_yaxes(title_text="edge over baseline  (pts)", range=[0, 14])
    return fig


def fig_edge_lives():
    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.16,
                        subplot_titles=("Best and worst underlyings",
                                        "How liquid are the options?"))
    rows = sorted(ASSETS_WORST + ASSETS_BEST, key=lambda r: r[2])
    fig.add_trace(go.Bar(
        y=[r[0] for r in rows], x=[r[2] for r in rows], orientation="h", width=0.68,
        marker_color=[GREEN if r[2] > 3 else (RED if r[2] < 0 else "#B9C2CC") for r in rows],
        text=["n=%d" % r[1] for r in rows], textposition="outside",
        textfont=dict(color=GREY, size=10), cliponaxis=False), row=1, col=1)
    fig.add_trace(go.Bar(
        x=[r[0] for r in LIQUIDITY], y=[r[2] for r in LIQUIDITY], width=0.5,
        marker_color=[NAVY, LBLUE, "#B9C2CC"],
        text=["%+.1f pts<br>n=%d" % (r[2], r[1]) for r in LIQUIDITY], textposition="outside",
        textfont=dict(color=NAVY, size=12), cliponaxis=False), row=1, col=2)
    base(fig, "The edge is in agriculture, and in liquid options",
         "Edge over baseline in percentage points. Pair trades count against both legs, so counts "
         "exceed the 1,970 headline.", 470, margin=dict(l=125, r=40, t=115, b=60))
    fig.update_xaxes(title_text="edge over baseline  (pts)", range=[-8, 27], row=1, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(range=[0, 15], row=1, col=2)
    for a in fig.layout.annotations[:2]:
        a.font.size = 14
        a.font.color = GREY
    return fig


def fig_confidence():
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.15,
                        subplot_titles=("Edge by confidence bucket",
                                        "Split each engine at its own median"))
    fig.add_trace(go.Bar(
        x=[c[0] for c in CONF_BUCKETS], y=[c[2] for c in CONF_BUCKETS], width=0.62,
        marker_color=LBLUE, text=["%+.1f" % c[2] for c in CONF_BUCKETS],
        textposition="outside", textfont=dict(color=NAVY, size=13), cliponaxis=False),
        row=1, col=1)
    fig.add_trace(go.Bar(
        x=[c[0] for c in CONF_HALVES], y=[c[2] for c in CONF_HALVES], width=0.45,
        marker_color=[GREEN, "#B9C2CC"],
        text=["%+.1f pts" % c[2] for c in CONF_HALVES], textposition="outside",
        textfont=dict(color=NAVY, size=13), cliponaxis=False), row=1, col=2)
    base(fig, "The confidence score does not pick winners",
         "Edge over baseline. If the score worked, the left panel would climb and the right bar "
         "would be taller than the left one. Rank correlation inside an engine: -0.002.", 450,
         margin=dict(l=75, r=40, t=118, b=75))
    fig.update_yaxes(title_text="edge over baseline  (pts)", range=[0, 12.5], row=1, col=1)
    fig.update_yaxes(range=[0, 12.5], row=1, col=2)
    fig.update_xaxes(title_text="confidence the dashboard assigned", row=1, col=1)
    for a in fig.layout.annotations[:2]:
        a.font.size = 14
        a.font.color = GREY
    return fig


def fig_grid():
    """Engine by asset class. Blank where the cell held fewer than 15 trades."""
    z, text = [], []
    for eng in GRID_ROWS:
        zrow, trow = [], []
        for cls in GRID_COLS:
            cell = ENGINE_GRID[eng].get(cls)
            if cell is None:
                zrow.append(None)
                trow.append("")
            else:
                edge, n = cell
                zrow.append(edge)
                trow.append("<b>%+.0f</b><br><span style='font-size:10px'>n=%d</span>" % (edge, n))
        z.append(zrow)
        text.append(trow)

    fig = go.Figure(go.Heatmap(
        z=z, x=GRID_COLS, y=GRID_ROWS, text=text, texttemplate="%{text}",
        textfont=dict(size=13), zmid=0, zmin=-25, zmax=25,
        colorscale=[[0.0, RED], [0.5, "#F2F4F7"], [1.0, GREEN]], xgap=3, ygap=3,
        colorbar=dict(title=dict(text="edge<br>(pts)", side="right"), thickness=12, len=0.72),
        hovertemplate="%{y} on %{x}<br>edge %{z:+.1f} pts<extra></extra>"))
    base(fig, "Vol dispersion travels; lead-lag does not",
         "Edge over baseline in percentage points. Blank cells held fewer than 15 trades. "
         "A pair spanning two classes counts in both.", 460,
         margin=dict(l=195, r=90, t=90, b=65))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


if __name__ == "__main__":
    for name, fig in [("engines", fig_engines()), ("hold", fig_hold()),
                      ("edge_lives", fig_edge_lives()), ("confidence", fig_confidence()),
                      ("grid", fig_grid())]:
        fig.write_html("/tmp/deck/assets/%s.html" % name, include_plotlyjs=True)
    print("ok")
