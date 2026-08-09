"""Charts from the user's REAL backtest table (transcribed from their screenshot).
SC Prosper palette, white ground, sized for a 13.33x7.5in slide."""
import plotly.graph_objects as go

NAVY, BLUE, LBLUE, GREEN, LGREEN = "#020B43", "#0473EA", "#7BB6F5", "#38D200", "#92E773"
GREY, GRID, RED = "#525355", "#DCE1E7", "#D0021B"
FONT = "SC Prosper Sans, Arial, Helvetica, sans-serif"

# engine, trades, hit, ci_lo, ci_hi, baseline, edge, p, avg_conf
# Real 5-day run: 2,102 marked trades, 52 replays, 2025-11-22 to 2026-08-09.
DATA = [
    ("IV mean-reversion",      290, 51.0, 45, 57, 44.9,  6.1, 0.0206, 69),
    ("Variance risk premium",  268, 61.2, 55, 67, 49.6, 11.6, 0.0001, 57),
    ("Vol dispersion (pairs)", 201, 58.2, 51, 65, 48.2, 10.1, 0.0027, 64),
    ("Correlation RV",         540, 49.1, 45, 53, 49.8, -0.7, 0.6432, 51),
    ("Lead-lag catch-up",      803, 51.3, 48, 55, 49.7,  1.6, 0.1965, 51),
]

# Pooled confidence buckets, and the within-engine comparison that reinterprets them.
CALIB = [("<45", 359, 48.2, 46.0, 2.2), ("45-55", 811, 51.7, 49.1, 2.6),
         ("55-65", 465, 53.1, 50.2, 3.0), ("65-75", 283, 58.3, 50.4, 7.9),
         ("75+", 184, 55.4, 48.4, 7.0)]
WITHIN = [("Low-confidence half<br>of each engine", 1103, 3.4),
          ("High-confidence half<br>of each engine", 999, 4.0)]
COLOR = {"IV mean-reversion": BLUE, "Variance risk premium": NAVY,
         "Vol dispersion (pairs)": LBLUE, "Correlation RV": GREEN,
         "Lead-lag catch-up": LGREEN}


def base(fig, title, sub, height):
    fig.update_layout(
        title=dict(text="<b>%s</b><br><span style='font-size:13px;color:%s'>%s</span>"
                        % (title, GREY, sub),
                   font=dict(size=20, color=NAVY, family=FONT), x=0, xanchor="left", y=0.94),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=height,
        font=dict(family=FONT, color=GREY, size=13),
        margin=dict(l=170, r=40, t=80, b=50), showlegend=False)
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    return fig


def fig_engines():
    d = sorted(DATA, key=lambda r: r[6])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[r[0] for r in d], x=[r[2] for r in d], orientation="h",
        marker_color=[COLOR[r[0]] for r in d], width=0.6,
        error_x=dict(type="data", symmetric=False,
                     array=[r[4] - r[2] for r in d], arrayminus=[r[2] - r[3] for r in d],
                     color=GREY, thickness=1.3, width=5)))
    fig.add_trace(go.Scatter(
        y=[r[0] for r in d], x=[r[5] for r in d], mode="markers",
        marker=dict(symbol="diamond", size=13, color="#FFFFFF",
                    line=dict(color=NAVY, width=2))))
    for r in d:
        col = GREEN if r[7] < 0.05 else GREY
        fig.add_annotation(x=104, y=r[0], xanchor="left", showarrow=False,
                           text="<b style='color:%s'>%+.1f pts</b>   n=%d   p=%.3f"
                                % (col, r[6], r[1], r[7]),
                           font=dict(color=GREY, size=12))
    base(fig, "Three of the five engines beat doing nothing",
         "Bars: win rate with 95% confidence interval. Diamonds: the same trades taken on every "
         "date, signal or not. 5-session hold.", 470)
    fig.update_xaxes(range=[0, 143], tickvals=[0, 20, 40, 60, 80, 100],
                     ticktext=["0", "20%", "40%", "60%", "80%", "100%"])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=170, r=20, t=85, b=50), xaxis_title="win rate")
    return fig


def fig_confidence():
    """Pooled buckets rise; within engine the score is flat. That gap is the finding."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.13,
                        subplot_titles=("Pooled across engines: edge rises with confidence",
                                        "Within each engine: it does not"))
    fig.add_trace(go.Bar(
        x=[c[0] for c in CALIB], y=[c[4] for c in CALIB], marker_color=BLUE, width=0.62,
        text=["%+.1f" % c[4] for c in CALIB], textposition="outside",
        textfont=dict(color=NAVY, size=13), cliponaxis=False), row=1, col=1)
    fig.add_trace(go.Bar(
        x=[w[0] for w in WITHIN], y=[w[2] for w in WITHIN], marker_color=LBLUE, width=0.5,
        text=["%+.1f" % w[2] for w in WITHIN], textposition="outside",
        textfont=dict(color=NAVY, size=13), cliponaxis=False), row=1, col=2)
    base(fig, "Confidence sorts engines, not trades",
         "Edge over baseline, in percentage points. The left panel looks like a working score; "
         "it is engine composition — the high buckets fill with the engines that work.", 470)
    fig.update_layout(margin=dict(l=70, r=40, t=115, b=70), showlegend=False)
    fig.update_yaxes(title_text="edge over baseline  (pts)", range=[0, 9.6], row=1, col=1)
    fig.update_yaxes(range=[0, 9.6], row=1, col=2)
    fig.update_xaxes(title_text="confidence bucket", row=1, col=1)
    for a in fig.layout.annotations:
        a.font.size = 14
        a.font.color = GREY
    return fig


if __name__ == "__main__":
    for name, fig in [("engines", fig_engines()), ("confidence", fig_confidence())]:
        fig.write_html("/tmp/deck/assets/%s.html" % name, include_plotlyjs=True)
    print("ok")
