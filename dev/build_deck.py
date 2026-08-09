"""Builds the intern presentation on the Standard Chartered Prosper template.

    SC_TEMPLATE=/path/to/template.pptx python dev/build_deck.py

The template is not committed (corporate asset); point SC_TEMPLATE at your copy.

Only the template's own layouts and placeholders are used — no hand-positioned
shapes, no invented colours, no accent bars — so the visual design is the
template's rather than something bolted on top. Charts are dropped in at the
content placeholder's own geometry.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import PP_PLACEHOLDER

import os
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.environ.get("SC_TEMPLATE", os.path.join(HERE, "template_standard_chartered.pptx"))
OUT = os.environ.get("DECK_OUT", os.path.join(os.path.dirname(HERE), "commodities_dashboard_presentation.pptx"))
ASSETS = os.environ.get("DECK_ASSETS", os.path.join(HERE, "assets"))

NAVY = "020B43"
CHROME = (PP_PLACEHOLDER.DATE, PP_PLACEHOLDER.FOOTER)

prs = Presentation(TEMPLATE)
M0, M1 = prs.slide_masters[0], prs.slide_masters[1]

# Start from an empty deck: the template ships 92 example slides.
ids = prs.slides._sldIdLst
for sld in list(ids):
    prs.part.drop_rel(sld.rId)
    ids.remove(sld)


def add(layout):
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        if ph.placeholder_format.type in CHROME:
            ph._element.getparent().remove(ph._element)
    return s


def drop(shape):
    shape._element.getparent().remove(shape._element)


def bullets(ph, items, size=None):
    """One paragraph per item, inheriting the layout's list style."""
    tf = ph.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        bold, text = (it if isinstance(it, tuple) else (False, it))
        run = para.add_run()
        run.text = text
        run.font.bold = bold
        if size:
            run.font.size = Pt(size)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def picture(slide, ph, path, pad=0.0):
    """Place an image at a placeholder's geometry, preserving aspect ratio."""
    from PIL import Image
    left, top, cw, ch = ph.left, ph.top, ph.width, ph.height
    drop(ph)
    iw, ih = Image.open(path).size
    scale = min((cw - Inches(pad)) / iw, (ch - Inches(pad)) / ih)
    w, h = int(iw * scale), int(ih * scale)
    slide.shapes.add_picture(path, left + (cw - w) // 2, top + (ch - h) // 2, w, h)


# ---------------------------------------------------------------- 1. Title
s = add(M0.slide_layouts[0])                      # Title slide - Prosper Blue
s.placeholders[0].text_frame.paragraphs[0].add_run().text = \
    "Commodities Vol & Relative-Value Dashboard"
s.placeholders[1].text_frame.paragraphs[0].add_run().text = \
    "Hassan Kanji  |  Summer Internship 2026"
drop(s.placeholders[10])
notes(s, """
Ten minutes, three questions: what I built, how it decides a trade is worth looking at, and
whether those trades were actually right. The last one is where most of my time went, and it is
the part that changed what I think of the tool.
""")

# ------------------------------------------------------- 2. What I built
s = add(M1.slide_layouts[28])                     # Title & image right - Prosper Blue
s.placeholders[0].text_frame.paragraphs[0].add_run().text = "What I built"
bullets(s.placeholders[2], [
    "A daily monitor over 29 commodity and macro instruments. It flags where volatility or a "
    "relationship has moved outside its normal range, and turns each one into a specific trade "
    "with the reason attached — the action, the trigger, and what has to happen for it to work."])
notes(s, """
Screenshot of the Signals tab goes in the picture placeholder on the right.

The universe is six commodity classes — energy, precious metals, base metals, grains, softs,
livestock — plus S&P, dollar and ten-year futures as macro overlays, so commodities can be tested
against the things that actually move them. 23 of the 29 have listed options, so they carry
implied volatility as well as price.

The point of the tool is not to predict prices. It is to notice when something is priced unusually
relative to its own history, and to say so in the language of a trade rather than a statistic.
""")

# ------------------------------------------------- 3. The shared idea
s = add(M1.slide_layouts[8])                      # 1 Column content option A
s.placeholders[0].text_frame.paragraphs[0].add_run().text = \
    "Every engine asks the same question"
bullets(s.placeholders[1], [
    (True, "Is this number unusual for this asset, judged by its own history?"),
    "Each quantity is ranked against its own past year. The top or bottom 10% counts as stretched.",
    (True, "Worked example — WTI implied volatility at the 8th percentile"),
    "Options on WTI are pricing less risk than this market has priced on 92% of the past year's "
    "days. Nothing is forecast about the oil price: the claim is that volatility is cheap relative "
    "to how this market normally prices it, so the trade is to buy volatility.",
    (True, "The bet underneath all five engines is mean reversion"),
    "Stretched things revert more often than they stretch further. Every engine is that same test "
    "applied to a different quantity — and the backtest later is a test of exactly this assumption.",
], size=16)
notes(s, """
This is the slide to slow down on. If the audience takes one thing away, it is that the tool
measures a quantity against its own history and calls the extremes.

Why percentiles rather than absolute levels: 20 vol is cheap for natural gas and expensive for
gold. Ranking each asset against itself makes them comparable, and makes one threshold work across
the whole universe.

If asked why 10%: it is a control on the panel, not a constant. Loosening it prints more ideas of
lower quality. That trade-off is exactly what the backtest measures.
""")

# --------------------------------------------------- 4. Five engines
s = add(M1.slide_layouts[8])
s.placeholders[0].text_frame.paragraphs[0].add_run().text = \
    "Five engines, five kinds of dislocation"
ph = s.placeholders[1]
rows = [("Engine", "What it looks for", "The trade"),
        ("IV mean-reversion",
         "Implied vol at the top or bottom of its own 1-year range",
         "Buy vol when cheap, sell it when rich"),
        ("Variance risk premium",
         "Implied vol far above the volatility actually being delivered",
         "Sell vol and collect the premium"),
        ("Vol dispersion",
         "Two related assets whose vol spread has stretched from its norm",
         "Sell the expensive vol, buy the cheap one"),
        ("Correlation RV",
         "Two assets that normally track, now decoupled",
         "Buy the laggard, sell the outperformer"),
        ("Lead-lag catch-up",
         "One asset that reliably moves a few days before another",
         "Trade the follower in the leader's direction")]
# This template's content placeholders are plain body placeholders, so the table
# goes in as a shape at the placeholder's own geometry rather than through it.
left, top, width, height = ph.left, ph.top, ph.width, ph.height
drop(ph)
tbl = s.shapes.add_table(len(rows), 3, left, top, width, height).table
tbl.columns[0].width = Inches(3.0)
tbl.columns[1].width = Inches(5.3)
tbl.columns[2].width = Inches(4.36)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        para = cell.text_frame.paragraphs[0]
        run = para.add_run()
        run.text = val
        run.font.size = Pt(14 if r else 13)
        run.font.bold = bool(r == 0 or c == 0)
notes(s, """
Do not read the table out. Take two engines and explain them properly, then say the other three
follow the same pattern.

The two worth explaining are the variance risk premium — the option market charges more for
volatility than the asset ends up delivering, which is a well-documented premium and turns out to
be where the edge is — and correlation RV, because it is the one people ask about. Correlation RV
is not a momentum trade: when two normally-linked assets pull apart, it buys the one that has
lagged and sells the one that has run, expecting the gap to close.

The first three trade volatility itself, so they are marked in vol points. The last two trade
price, so they are marked in percent. That is why the results table has two different units.
""")

# ------------------------------------------ 5. Supporting views
s = add(M1.slide_layouts[31])                     # Title & image left - white
s.placeholders[0].text_frame.paragraphs[0].add_run().text = "Two supporting views"
bullets(s.placeholders[2], [
    "Vol Monitor ranks every asset by implied vol, realized vol and the gap between them, with a "
    "data-quality flag so illiquid option marks do not masquerade as signals.\n\n"
    "Correlation & Pairs shows how tightly two assets move together, whether that link is "
    "currently stretched, and which of the two moves first."])
notes(s, """
Screenshot of the Correlation & Pairs tab goes in the picture placeholder.

Worth mentioning: the data-quality flag is computed, not assigned by hand. It measures how often
an implied-vol series actually re-prices — on this feed even liquid commodity options only print a
fresh mark on 50 to 70% of days, and genuinely illiquid ones barely move at all. Signals built on
a frozen mark are excluded rather than shown with a caveat.

The lead-lag chart is the one to point at: it shows the correlation at every lag, so a flat curve
means there is no timing edge, only ordinary correlation.
""")

# --------------------------------------------- 6. The test
s = add(M1.slide_layouts[8])
s.placeholders[0].text_frame.paragraphs[0].add_run().text = \
    "Do the ideas actually work? How I tested"
bullets(s.placeholders[1], [
    (True, "Replayed a year, one week at a time"),
    "52 replays. At each date the data is rewound and the dashboard's own code is re-run, so the "
    "signals scored are the ones it would genuinely have printed that day. Nothing after the "
    "replay date is visible to it.",
    (True, "Recorded every signal — 2,091 trades — not the good ones"),
    "Each is held five sessions and marked entry close to exit close. One unit per trade, no "
    "sizing, so the result measures whether the signal pointed the right way, not a P&L.",
    (True, "Scored every trade against a baseline"),
    "A hit rate alone proves nothing. So each trade was also scored as if taken on every date in "
    "the year, signal or not. That unconditional rate is the baseline, and the gap to it is the "
    "only thing that can honestly be called an edge.",
], size=15)
notes(s, """
The baseline is the idea to sell here, because it is what makes the rest of the numbers mean
anything.

Concrete version: selling volatility wins about 60% of the time in this test. That sounds
excellent until you check that selling volatility on a random day, with no signal at all, wins
about 50% of the time — because implied vol tends to exceed realized vol as a rule. The signal is
worth the 10-point gap, not the 60%.

Without that comparison I would have reported five successful engines. With it, three of them
disappear.

If asked about look-ahead: the confidence scores inside the dashboard are computed from forward
windows, but those windows only ever see data inside the truncated frame, and the test asserts
that a replay produces identical signals whether or not future data is present in memory.
""")

# --------------------------------------------- 7. Results
s = add(M1.slide_layouts[8])
s.placeholders[0].text_frame.paragraphs[0].add_run().text = \
    "Two of the five engines beat doing nothing"
picture(s, s.placeholders[1], "%s/engines.png" % ASSETS)
notes(s, """
Across all 2,091 trades the signals won 52.0% of the time against a 48.9% baseline — a 3.1 point
edge, 95% interval 49.8 to 54.1, p = 0.002. Real, but small, and it is not spread evenly.

The variance risk premium and vol dispersion carry it: roughly +10 points each, both significant
against their own baselines. Correlation RV and lead-lag are inside noise. IV mean-reversion is
+2 points and not significant.

The IV mean-reversion line is the one to explain rather than skip. Its win rate is 46.6%, below a
coin flip, which looks like a broken engine — but its baseline is 44.6%. Buying volatility loses
more often than it wins by nature, because vol drifts down most of the time and pays off in rare
bursts. Judged against the right benchmark it is roughly neutral, not a disaster. That is the
clearest example of why the baseline matters.
""")

# --------------------------------------------- 8. Confidence
s = add(M1.slide_layouts[8])
s.placeholders[0].text_frame.paragraphs[0].add_run().text = \
    "The confidence score did not rank the trades"
picture(s, s.placeholders[1], "%s/conf_vs_edge.png" % ASSETS)
notes(s, """
The dashboard attaches a confidence percentage to every idea: on past days when this setup was at
least this extreme, how often did the bet work.

The backtest says it does not do its job. The engine it was most sure about — IV mean-reversion at
69% average confidence — delivered the least edge. The two engines that actually worked carried
the lowest and middling confidence. Raising the minimum-confidence filter throws away trades
without improving the edge on what is left.

The diagnosis is more interesting than the failure. The score measures how often a setup worked,
not how much more often it worked than the same bet on any other day — so an engine sitting on a
rich base rate scores high without any skill in it. It is measuring the base rate and calling it
confidence.

So: not delete it, fix it. Score confidence as the gap to the base rate rather than the raw hit
rate. I have the diagnostic version of that in the backtest; a deployable one needs a
point-in-time base rate rather than one computed over the whole sample.
""")

# --------------------------------------------- 9. Limits and next
s = add(M1.slide_layouts[12])                     # 2 Column content option A
s.placeholders[0].text_frame.paragraphs[0].add_run().text = "What I would change, and what this does not prove"
bullets(s.placeholders[1], [
    (True, "What I would change"),
    "Rebuild confidence as the gap to a base rate, not a raw hit rate.",
    "Concentrate on the two vol engines; retire or rework correlation RV and lead-lag, which show "
    "no measurable edge over a year.",
    "Report every signal against its baseline in the dashboard itself, not only in the backtest.",
], size=15)
bullets(s.placeholders[13], [
    (True, "What this does not prove"),
    "No costs, no sizing, no option greeks — this measures direction, not profit.",
    "A setup that stays extreme for weeks is recorded at every replay, so the trades are not "
    "independent and the real sample is smaller than 2,091.",
    "One year, one universe, one set of parameters.",
], size=15)
notes(s, """
Say the limitations before anyone asks — it is the difference between a result and a claim.

The overlap point is the one a sharp listener will raise. Weekly replays with five-session holds
do not overlap in time, but the same signal reappearing for six weeks is six correlated records,
not six independent ones. It inflates confidence in the result, which is why the honest read of
+3.1 points is "small and real" rather than "proven".

If asked what I would do with more time: walk the test across several years to see whether the two
working engines hold up outside this particular year, and test the rebuilt confidence score
properly out of sample.
""")

# --------------------------------------------- 10. Closing statement
s = add(M1.slide_layouts[2])                      # Statement - Prosper Blue
s.placeholders[0].text_frame.paragraphs[0].add_run().text = \
    "A hit rate means nothing until you know the baseline"
bullets(s.placeholders[13], [
    "Three of five engines, and the confidence score, only looked good until "
    "there was something to compare them against."])
notes(s, """
Close on this rather than a summary. The technical work is the dashboard; the judgement is the
baseline.

Three things I would name if asked what I learned: a hit rate without a benchmark is not evidence;
building the thing that tests your work matters as much as building the work; and the most useful
result of the project was the one that told me part of it did not work.
""")

prs.save(OUT)
print("wrote %s — %d slides" % (OUT, len(prs.slides)))
