"""Charts from the user's REAL backtest table (transcribed from their screenshot).
SC Prosper palette, white ground, sized for a 13.33x7.5in slide."""
import plotly.graph_objects as go

NAVY, BLUE, LBLUE, GREEN, LGREEN = "#020B43", "#0473EA", "#7BB6F5", "#38D200", "#92E773"
GREY, GRID, RED = "#525355", "#DCE1E7", "#D0021B"
FONT = "SC Prosper Sans, Arial, Helvetica, sans-serif"

# engine, trades, hit, ci_lo, ci_hi, baseline, edge, p, avg_conf
DATA = [
    ("IV mean-reversion",      290, 46.6, 41, 52, 44.6,  2.0, 0.266, 69),
    ("Variance risk premium",  269, 59.9, 54, 66, 50.1,  9.8, 0.001, 57),
    ("Vol dispersion (pairs)", 190, 58.4, 51, 65, 48.0, 10.4, 0.003, 64),
    ("Correlation RV",         537, 50.5, 46, 55, 49.7,  0.8, 0.376, 51),
    ("Lead-lag catch-up",      805, 50.8, 47, 54, 49.6,  1.2, 0.261, 51),
]
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
    base(fig, "Two of the five engines beat doing nothing",
         "Bars: win rate with 95% confidence interval. Diamonds: the same trades taken on every "
         "date, signal or not. 5-session hold.", 470)
    fig.update_xaxes(range=[0, 143], tickvals=[0, 20, 40, 60, 80, 100],
                     ticktext=["0", "20%", "40%", "60%", "80%", "100%"])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=170, r=20, t=85, b=50), xaxis_title="win rate")
    return fig


def fig_conf_vs_edge():
    fig = go.Figure()
    for name, n, hit, lo, hi, bl, edge, p, conf in DATA:
        fig.add_trace(go.Scatter(
            x=[conf], y=[edge], mode="markers",
            marker=dict(size=[max(16, min(46, n / 18.0))], color=COLOR[name],
                        line=dict(color="#FFFFFF", width=2), opacity=0.95)))
        # Correlation RV and Lead-lag sit almost on top of each other at ~51%
        # confidence and ~+1 pt of edge, so their labels are placed by hand.
        pos = {"Variance risk premium": (0, 30, "center"),
               "Vol dispersion (pairs)": (0, -30, "center"),
               "IV mean-reversion": (0, 28, "center"),
               "Lead-lag catch-up": (-14, 34, "right"),
               "Correlation RV": (-14, -26, "right")}[name]
        fig.add_annotation(x=conf, y=edge, text=name, showarrow=False,
                           xshift=pos[0], yshift=pos[1], xanchor=pos[2],
                           font=dict(color=NAVY, size=13))
    fig.add_hline(y=0, line=dict(color=GREY, width=1, dash="dot"))
    base(fig, "The score the dashboard was most sure about had the least edge",
         "Each bubble is one engine. Horizontal: average confidence the dashboard assigned. "
         "Vertical: edge actually delivered over baseline. Bubble size: number of trades.", 470)
    fig.update_layout(margin=dict(l=80, r=50, t=85, b=60),
                      xaxis_title="average confidence at entry", yaxis_title="edge over baseline  (pts)")
    fig.update_xaxes(range=[47.5, 73], ticksuffix="%")
    fig.update_yaxes(range=[-2.5, 13.5])
    return fig


if __name__ == "__main__":
    for name, fig in [("engines", fig_engines()), ("conf_vs_edge", fig_conf_vs_edge())]:
        fig.write_html("/tmp/deck/assets/%s.html" % name, include_plotlyjs=True)
    print("ok")
