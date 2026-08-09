# Prompt + data pack for building the presentation

Paste everything below the line into Claude for PowerPoint, and attach the Standard Chartered
template plus the files from `presentation/` that the backtest notebook writes.

Two things make this work better than a bare prompt: the **methodology section** is written so the
tool explains the project the way you would defend it, and every **number is real and sourced**, so
nothing on a slide is invented. Replace any figure below if your latest run differs — the notebook
writes the current set to `presentation/NUMBERS.md`.

---

## The ask

Build a 10-slide deck for a 10-minute final internship presentation, using the attached Standard
Chartered "Prosper" template — use its own layouts, fonts and colours, not new ones. The audience
is a commodities trading desk: they know markets, they do not know this project. They will not
read dense slides, so keep each slide to one idea with a headline that states the finding rather
than naming the topic ("Two of the five engines beat doing nothing", not "Backtest results").

Put the detail in speaker notes, not on the slides. I have to be able to explain every number I
show, so the notes should tell me what to say and what to expect if I am challenged.

## What the project is

A daily monitor over 29 commodity and macro instruments, built in Python in Bloomberg BQuant. It
scans for places where volatility or a relationship has moved outside its normal range, and turns
each one into a specific trade with an action, a trigger and a plain-English reason. Six commodity
classes — energy, precious metals, base metals, grains/oilseeds, softs, livestock — plus S&P 500,
the dollar index and 10-year note futures as macro overlays. 23 of the 29 have listed options, so
they carry implied volatility as well as price. Three years of history.

Then a second piece: a backtest that replays a year of history and measures whether the signals
were right.

## The methodology, in the words I want to be able to use

**How a signal is born.** Every engine asks one question: *is this number unusual for this asset,
judged by its own history?* Each quantity is ranked as a percentile against its own past year; the
top or bottom 10% counts as stretched. Percentiles rather than absolute levels, because 20 vol is
cheap for natural gas and expensive for gold — ranking each asset against itself makes one
threshold work across the whole universe. The bet underneath all five engines is mean reversion:
stretched things revert more often than they stretch further.

Worked example to use on the "how it works" slide: WTI 30-day implied volatility sitting at the
8th percentile of its own year. Options are pricing less risk than this market has priced on 92%
of the past year's days. Nothing is being forecast about the oil price — the claim is only that
volatility is cheap relative to how this market normally prices it. The trade is to buy volatility.

**The five engines.**

| Engine | What it looks for | The trade | Marked in |
|---|---|---|---|
| IV mean-reversion | Implied vol at the top or bottom of its own 1-year range | Buy vol when cheap, sell when rich | vol points |
| Variance risk premium | Implied vol far above the volatility the asset is actually delivering | Sell vol, collect the premium | vol points |
| Vol dispersion (pairs) | Two related assets whose implied-vol spread has stretched from its norm | Sell the expensive vol, buy the cheap one | vol points |
| Correlation RV | Two assets that normally track each other, now decoupled | Buy the laggard, sell the outperformer | percent |
| Lead-lag catch-up | One asset that reliably moves a few days before another | Trade the follower in the leader's direction | percent |

Correlation RV is the one people misread: it is a convergence trade, not momentum. When two
normally-linked assets pull apart it buys the one that lagged and sells the one that ran,
expecting the gap to close. Lead-lag is the opposite — it follows the leader's direction.

**How the backtest works.** Step back through the past year one week at a time, 52 replays. At
each date the price and implied-vol history is truncated to that date and the dashboard's *own*
signal code is re-run — not a reimplementation — so the signals being scored are the ones it would
genuinely have printed that day. Every engine reads only the last row of each series, so a
truncated frame reproduces that day exactly, and nothing after the replay date is visible.

Every signal on the board is recorded, not a hand-picked few: 2,091 trades. Each is held five
trading sessions and marked entry close to exit close. One unit per trade, no sizing — the result
measures whether the signal pointed the right way, not a P&L.

**The part that matters most — the baseline.** A hit rate on its own proves nothing. Selling
volatility wins about 60% of the time in this test, which sounds excellent until you notice that
selling volatility on a random day with no signal at all wins about 50% of the time, because
implied vol tends to exceed realized vol as a rule. So every trade was also scored as if it had
been taken on *every* date in the year, signal or not. That unconditional rate is the baseline,
and the gap between the two is the only thing that can honestly be called an edge. Without this
comparison the project would have reported five successful engines; with it, three disappear.

Hit rates carry 95% Wilson confidence intervals and an exact binomial p-value against their own
baseline, so thin samples read as thin rather than as findings.

## The results — every number below is real

**Overall:** 2,091 trades across 52 weekly replays, 5-session holds. Hit rate **52.0%** (95% CI
49.8–54.1) against a **48.9%** baseline — an edge of **+3.1 points**, p = 0.002.

**By engine:**

| Engine | Trades | Hit rate | 95% CI | Baseline | Edge | p | Avg P&L | Avg confidence |
|---|---|---|---|---|---|---|---|---|
| Variance risk premium | 269 | 59.9% | 54–66 | 50.1% | **+9.8** | 0.001 | +7.97 vol pts | 57% |
| Vol dispersion (pairs) | 190 | 58.4% | 51–65 | 48.0% | **+10.4** | 0.003 | +5.72 vol pts | 64% |
| IV mean-reversion | 290 | 46.6% | 41–52 | 44.6% | +2.0 | 0.266 | +1.76 vol pts | 69% |
| Lead-lag catch-up | 805 | 50.8% | 47–54 | 49.6% | +1.2 | 0.261 | −0.13 % | 51% |
| Correlation RV | 537 | 50.5% | 46–55 | 49.7% | +0.8 | 0.376 | −0.09 % | 51% |

Two engines work — the variance risk premium and vol dispersion, both around +10 points and both
significant. Three do not: correlation RV and lead-lag are inside noise, and IV mean-reversion is
+2 points and not significant.

**Do not skip the IV mean-reversion line.** Its win rate of 46.6% is below a coin flip, which
looks like a broken engine — but its baseline is 44.6%. Buying volatility loses more often than it
wins by nature, because vol drifts lower most of the time and pays off in rare bursts. Judged
against the right benchmark it is roughly neutral, not a disaster. It is the clearest single
example of why the baseline matters, and it belongs in the notes.

**The confidence score did not work.** The dashboard attaches a confidence % to every idea: on
past days when this setup was at least this extreme, how often did the bet work. The backtest says
it does not rank trades. The engine it was most sure about, IV mean-reversion at 69% average
confidence, delivered the least edge; the two engines that actually worked carried lower
confidence. Raising the minimum-confidence filter discards trades without improving the edge on
what is left.

The diagnosis is more interesting than the failure: the score measures how often a setup worked,
not how much more often it worked than the same bet on any other day — so an engine sitting on a
rich base rate scores high with no skill in it. It is measuring the base rate and calling it
confidence. The fix is to score confidence as the gap to the base rate, which needs a
point-in-time base rate to be deployable.

## Limitations — put these on a slide, do not bury them

- No costs, no sizing, no option greeks. This measures direction, not profit.
- Trades are not independent: a setup that stays extreme for weeks is recorded at every replay, so
  the effective sample is smaller than 2,091 and the +3.1 points should be read as "small and
  real", not "proven".
- One year, one universe, one parameter set.

## Structure I want

1. **Title** — project name, my name, internship 2026.
2. **What I built** — one paragraph, screenshot of the Signals tab.
3. **Every engine asks the same question** — the percentile idea plus the WTI worked example.
4. **The five engines** — the table above, three columns, no more.
5. **Two supporting views** — Vol Monitor and Correlation & Pairs tabs, screenshot.
6. **How I tested it** — replay, every signal recorded, the baseline.
7. **Results** — the per-engine chart (`presentation/02_engines.png`).
8. **The confidence score didn't rank** — the confidence chart (`presentation/05_confidence_by_engine.png` or the confidence-vs-edge bubble chart).
9. **What I'd change / what this doesn't prove** — two columns.
10. **Closing statement** — "A hit rate means nothing until you know the baseline."

Five engines on five separate slides would eat the whole ten minutes on setup and leave nothing
for the result, which is the interesting half. One slide for the shared idea and one for the five
engines is the right trade.

## Design constraints

- Use the template's own layouts — title slide, 1-column and 2-column content, statement, and the
  title-and-image layouts for screenshots. Do not invent a layout.
- Standard Chartered Prosper palette only: navy `#020B43`, blue `#0473EA`, green `#38D200`, light
  blue `#7BB6F5`, light green `#92E773`, body grey `#525355`. Fonts: SC Prosper Sans, with SC
  Prosper Sans Light for headings.
- No accent bars or coloured stripes under titles.
- Headline every slide with the finding, not the topic.
- Substantial speaker notes on every slide: what to say, and what to answer if challenged.
