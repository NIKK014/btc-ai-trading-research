const pptxgen = require("pptxgenjs");

const INK = "1A1F2B";        // near-black navy, dominant on dark slides
const PAPER = "FFFFFF";
const GREEN = "1F9D55";      // same greens/reds as the live dashboard
const RED = "D64545";
const BLUE = "3B82F6";
const MUTED = "8A8F98";
const FAINT = "F4F5F7";

const H = "Cambria";         // serif headers
const B = "Calibri";         // sans body

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";    // 13.3 x 7.5
p.author = "Nico Mooney";
p.title = "Bitcoin AI Trading Research";

const W = 13.3, HT = 7.5, M = 0.7;

// Repeated motif: a small filled circle carrying a letter or number.
function badge(s, x, y, text, fill, txtColor) {
  s.addShape(p.ShapeType.ellipse, { x, y, w: 0.42, h: 0.42, fill: { color: fill } });
  s.addText(text, {
    x, y, w: 0.42, h: 0.42, align: "center", valign: "middle",
    fontFace: B, fontSize: 13, bold: true, color: txtColor || PAPER, margin: 0,
  });
}

function titleSlide(s, text, kicker) {
  if (kicker) {
    s.addText(kicker, {
      x: M, y: 0.42, w: W - 2 * M, h: 0.3, fontFace: B, fontSize: 12,
      bold: true, color: BLUE, charSpacing: 2, margin: 0,
    });
  }
  s.addText(text, {
    x: M, y: 0.75, w: W - 2 * M, h: 0.8, fontFace: H, fontSize: 34,
    bold: true, color: INK, margin: 0,
  });
}

/* ---------------------------------------------------------- 1. Title */
{
  const s = p.addSlide();
  s.background = { color: INK };
  s.addText("Can AI improve a Bitcoin trading strategy?", {
    x: M, y: 2.25, w: 10.4, h: 1.5, fontFace: H, fontSize: 42, bold: true,
    color: PAPER, margin: 0,
  });
  s.addText(
    "A rule-based strategy, a machine-learning filter and an LLM judge, " +
    "tested against the same out-of-sample period.",
    { x: M, y: 3.85, w: 9.6, h: 0.8, fontFace: B, fontSize: 16, color: "C7CBD4", margin: 0 }
  );
  s.addShape(p.ShapeType.rect, { x: M, y: 4.95, w: 1.1, h: 0.035, fill: { color: BLUE } });
  s.addText("Nico Mooney   ·   Ironhack AI Engineering   ·   Final Project", {
    x: M, y: 5.25, w: 9, h: 0.35, fontFace: B, fontSize: 13, color: MUTED, margin: 0,
  });
  s.addText("BTCUSDT · 4h · 2020–2026 · paper trading only", {
    x: M, y: 5.62, w: 9, h: 0.35, fontFace: B, fontSize: 12, color: "6C727E",
    italic: true, margin: 0,
  });
  s.addNotes(
    "Eight minutes. The honest headline: the machine-learning filter helped a little, " +
    "the LLM did not, and most of the work went into being able to prove that."
  );
}

/* --------------------------------------------- 2. Why this is hard */
{
  const s = p.addSlide();
  titleSlide(s, "Most backtests are wrong", "THE PROBLEM");
  s.addText(
    "Three failure modes destroy more trading research than bad strategies do. " +
    "Each one had to be designed out before any result could mean anything.",
    { x: M, y: 1.62, w: 11.9, h: 0.5, fontFace: B, fontSize: 14, color: "4A5160", margin: 0 }
  );

  const cards = [
    ["Look-ahead bias", "Using information the moment could not have had. Every indicator is causal; splits are chronological with an embargo at each seam.", "1"],
    ["Selection bias", "Search enough configurations and one looks brilliant by chance. 651 were scored — so the winner is a survivor, not a discovery.", "2"],
    ["Fees and slippage", "A 15-minute strategy trades itself to death. Costs are charged on every fill at both ends.", "3"],
  ];
  cards.forEach(([h, body, n], i) => {
    const x = M + i * 4.05;
    s.addShape(p.ShapeType.roundRect, {
      x, y: 2.4, w: 3.75, h: 3.35, fill: { color: FAINT }, rectRadius: 0.08,
    });
    badge(s, x + 0.3, 2.75, n, BLUE);
    s.addText(h, {
      x: x + 0.3, y: 3.35, w: 3.15, h: 0.4, fontFace: B, fontSize: 17, bold: true,
      color: INK, margin: 0,
    });
    s.addText(body, {
      x: x + 0.3, y: 3.85, w: 3.15, h: 1.7, fontFace: B, fontSize: 12.5,
      color: "4A5160", margin: 0, lineSpacingMultiple: 1.15,
    });
  });

  s.addText(
    "Nothing here is exotic. It is just the part that is usually skipped.",
    { x: M, y: 6.25, w: 11.9, h: 0.4, fontFace: B, fontSize: 13.5, color: MUTED,
      italic: true, margin: 0 }
  );
  s.addNotes("Don't linger. This slide buys credibility for the numbers later.");
}

/* ------------------------------------------------- 3. The experiment */
{
  const s = p.addSlide();
  titleSlide(s, "One thing changed at a time", "METHOD");
  s.addText(
    "Every arm trades the same signals, on the same data, through the same risk model. " +
    "Only the decision-maker changes — which is what makes this an experiment rather " +
    "than three separately-tuned demos.",
    { x: M, y: 1.62, w: 11.9, h: 0.6, fontFace: B, fontSize: 14, color: "4A5160", margin: 0 }
  );

  const rows = [
    ["A", "Rules only", "EMA crossover with RSI confirmation", BLUE],
    ["B", "Rules + ML filter", "Random forest must agree before a trade opens", BLUE],
    ["C", "Rules + ML + LLM judge", "GPT reviews each entry and may veto it", BLUE],
    ["✓", "Control: approve everything", "Must reproduce System A exactly — tests the harness", MUTED],
    ["✓", "Control: four lines of arithmetic", "The bar the LLM has to clear", MUTED],
  ];
  rows.forEach(([tag, name, desc, colour], i) => {
    const y = 2.55 + i * 0.8;
    badge(s, M, y, tag, colour);
    s.addText(name, {
      x: M + 0.62, y, w: 3.9, h: 0.42, fontFace: B, fontSize: 14.5, bold: true,
      color: INK, valign: "middle", margin: 0,
    });
    s.addText(desc, {
      x: M + 4.6, y, w: 7.3, h: 0.42, fontFace: B, fontSize: 13, color: "4A5160",
      valign: "middle", margin: 0,
    });
  });
  s.addText(
    "The two controls exist because any filter that removes trades changes the results, " +
    "whether or not it understands anything.",
    { x: M, y: 6.65, w: 11.9, h: 0.5, fontFace: B, fontSize: 13, color: MUTED,
      italic: true, margin: 0 }
  );
  s.addNotes(
    "The control arms are the part I'd defend hardest. Without them, 'System C beat " +
    "System A' is not evidence of reasoning — it's evidence that filtering changes things."
  );
}

/* ------------------------------------------------- 4. Validation */
{
  const s = p.addSlide();
  titleSlide(s, "On validation data, it looked excellent", "THE TEMPTING SLIDE");

  const stats = [
    ["Sharpe ratio", "1.92", GREEN],
    ["Total return", "+34.6%", GREEN],
    ["Max drawdown", "7.6%", INK],
    ["Trades", "157", INK],
  ];
  stats.forEach(([label, value, colour], i) => {
    const x = M + i * 3.05;
    s.addText(value, {
      x, y: 2.15, w: 2.8, h: 1.0, fontFace: H, fontSize: 46, bold: true,
      color: colour, margin: 0,
    });
    s.addText(label, {
      x, y: 3.2, w: 2.8, h: 0.35, fontFace: B, fontSize: 13, color: MUTED, margin: 0,
    });
  });

  s.addShape(p.ShapeType.roundRect, {
    x: M, y: 4.0, w: 11.9, h: 1.75, fill: { color: FAINT }, rectRadius: 0.08,
  });
  s.addText("This is where a lot of projects stop.", {
    x: M + 0.4, y: 4.25, w: 11.1, h: 0.45, fontFace: B, fontSize: 18, bold: true,
    color: INK, margin: 0,
  });
  s.addText(
    "A Sharpe near 2 on eighteen months of data is a good-looking result. It is also the " +
    "best of 651 configurations measured on the data used to choose it — so the only " +
    "honest thing to do next is spend the test set.",
    { x: M + 0.4, y: 4.75, w: 11.1, h: 0.8, fontFace: B, fontSize: 13.5,
      color: "4A5160", margin: 0, lineSpacingMultiple: 1.15 }
  );
  s.addNotes("Deliver this straight, with no irony. The turn comes on the next slide.");
}

/* ------------------------------------------------- 5. The collapse */
{
  const s = p.addSlide();
  titleSlide(s, "Out of sample, it fell apart", "THE RESULT");
  s.addText(
    "Same strategy, same parameters, untouched data from 30 June 2025 to 11 August 2026.",
    { x: M, y: 1.62, w: 11.9, h: 0.4, fontFace: B, fontSize: 14, color: "4A5160", margin: 0 }
  );

  s.addChart(
    p.ChartType.bar,
    [{ name: "Sharpe ratio", labels: ["Validation", "Test"], values: [1.92, -0.91] }],
    {
      x: M, y: 2.15, w: 6.0, h: 4.1,
      barDir: "col", chartColors: [GREEN, RED], showValue: true,
      dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
      dataLabelFontFace: B, dataLabelFontSize: 14, dataLabelFontBold: true,
      showTitle: true, title: "Sharpe ratio", titleFontFace: B, titleFontSize: 13,
      titleColor: MUTED,
      showLegend: false, catAxisLabelColor: INK, catAxisLabelFontSize: 13,
      catAxisLabelFontFace: B, valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
      valGridLine: { color: "E6E8EC", size: 1 }, catGridLine: { style: "none" },
      valAxisMinVal: -1.5, valAxisMaxVal: 2.5,
    }
  );

  s.addShape(p.ShapeType.roundRect, {
    x: 7.15, y: 2.15, w: 5.45, h: 4.1, fill: { color: INK }, rectRadius: 0.08,
  });
  s.addText("+34.6%  →  −10.5%", {
    x: 7.5, y: 2.6, w: 4.8, h: 0.6, fontFace: H, fontSize: 26, bold: true,
    color: PAPER, margin: 0,
  });
  s.addText("total return, validation to test", {
    x: 7.5, y: 3.2, w: 4.8, h: 0.3, fontFace: B, fontSize: 12, color: "9AA1AE", margin: 0,
  });
  s.addText(
    "The strategy did not break. It was never as good as the validation number said — " +
    "that number was the high-water mark of a 651-configuration search, and it did not " +
    "survive contact with data it had not chosen itself.",
    { x: 7.5, y: 3.95, w: 4.8, h: 2.0, fontFace: B, fontSize: 13, color: "C7CBD4",
      margin: 0, lineSpacingMultiple: 1.2 }
  );
  s.addNotes(
    "This is the centre of the talk. Say plainly: the honest result is a negative one, " +
    "and the reason I can tell you that is the test set was only ever touched once."
  );
}

/* --------------------------------------- 6. All systems, out of sample */
{
  const s = p.addSlide();
  titleSlide(s, "Did the filters help?", "SYSTEM COMPARISON");
  s.addText(
    "Total return over the same out-of-sample period. Bitcoin fell 41% across this window.",
    { x: M, y: 1.62, w: 11.9, h: 0.4, fontFace: B, fontSize: 14, color: "4A5160", margin: 0 }
  );

  s.addChart(
    p.ChartType.bar,
    [{
      name: "Total return",
      labels: ["Buy and hold", "A — rules", "C — + LLM judge", "B — + ML filter"],
      values: [-40.97, -10.549, -0.347, 1.63],
    }],
    {
      x: M, y: 2.2, w: 7.6, h: 4.0,
      barDir: "bar", chartColors: [RED, RED, RED, GREEN],
      varyColors: true, showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: '0.0"%"', dataLabelFontFace: B, dataLabelFontSize: 12,
      dataLabelFontBold: true,
      showLegend: false, catAxisLabelColor: INK, catAxisLabelFontSize: 12.5,
      catAxisLabelFontFace: B, catAxisLabelPos: "low", valAxisLabelColor: MUTED, valAxisLabelFontSize: 10,
      valGridLine: { color: "E6E8EC", size: 1 }, catGridLine: { style: "none" },
      valAxisMinVal: -50, valAxisMaxVal: 15,
    }
  );

  const notes = [
    ["The ML filter earned its place", "It cut trades from 129 to 64 and turned −10.5% into +1.6%.", GREEN],
    ["But it is not a good strategy", "Sharpe 0.22 on 64 trades. The confidence interval contains zero.", INK],
    ["Beating buy-and-hold here is faint praise", "Holding lost 41%. Not losing money in a falling market is most of the gap.", INK],
  ];
  notes.forEach(([h, body, colour], i) => {
    const y = 2.45 + i * 1.25;
    s.addShape(p.ShapeType.ellipse, { x: 8.6, y: y + 0.06, w: 0.13, h: 0.13, fill: { color: colour } });
    s.addText(h, {
      x: 8.88, y, w: 3.75, h: 0.32, fontFace: B, fontSize: 13.5, bold: true,
      color: INK, margin: 0,
    });
    s.addText(body, {
      x: 8.88, y: y + 0.34, w: 3.75, h: 0.7, fontFace: B, fontSize: 12,
      color: "4A5160", margin: 0, lineSpacingMultiple: 1.12,
    });
  });
  s.addNotes(
    "If asked why B works: it refuses trades the model disagrees with, and in a falling " +
    "market that means it sits out. That is a real effect, but 64 trades is a small sample."
  );
}

/* ------------------------------------------------- 7. The LLM verdict */
{
  const s = p.addSlide();
  titleSlide(s, "The LLM deliberated. It did not add skill.", "THE AI QUESTION");

  const stats = [
    ["130", "entry decisions judged"],
    ["37.7%", "agreed with the strategy"],
    ["62.3%", "vetoed the trade"],
    ["67", "mean self-reported confidence"],
  ];
  stats.forEach(([v, l], i) => {
    const x = M + i * 3.05;
    s.addText(v, {
      x, y: 1.85, w: 2.8, h: 0.75, fontFace: H, fontSize: 36, bold: true,
      color: INK, margin: 0,
    });
    s.addText(l, {
      x, y: 2.62, w: 2.8, h: 0.5, fontFace: B, fontSize: 12, color: MUTED, margin: 0,
    });
  });

  s.addShape(p.ShapeType.roundRect, {
    x: M, y: 3.5, w: 11.9, h: 1.5, fill: { color: FAINT }, rectRadius: 0.08,
  });
  s.addText("It is not distinguishable from four lines of arithmetic.", {
    x: M + 0.4, y: 3.75, w: 11.1, h: 0.42, fontFace: B, fontSize: 17, bold: true,
    color: INK, margin: 0,
  });
  s.addText(
    "A deterministic control judge — approve if the model agrees and clears 35% confidence — " +
    "was given the identical inputs. The difference between it and the LLM has a confidence " +
    "interval that contains zero.",
    { x: M + 0.4, y: 4.2, w: 11.1, h: 0.7, fontFace: B, fontSize: 13,
      color: "4A5160", margin: 0, lineSpacingMultiple: 1.15 }
  );

  s.addText(
    "Neither degenerate, which is what makes it interesting: a judge agreeing 100% of the " +
    "time is a rubber stamp, one agreeing 50% is a coin flip. This one genuinely " +
    "deliberated — and still added nothing measurable.",
    { x: M, y: 5.25, w: 11.9, h: 0.8, fontFace: B, fontSize: 13, color: MUTED,
      italic: true, margin: 0, lineSpacingMultiple: 1.15 }
  );
  s.addNotes(
    "Expect a question here. The prompt contains no dates and no absolute prices, so the " +
    "model cannot recall what happened next — leakage through the weights, not the dataframe."
  );
}

/* ------------------------------------------------- 8. Conclusion */
{
  const s = p.addSlide();
  s.background = { color: INK };
  s.addText("WHAT I'D DEFEND", {
    x: M, y: 0.6, w: 11.9, h: 0.3, fontFace: B, fontSize: 12, bold: true,
    color: BLUE, charSpacing: 2, margin: 0,
  });
  s.addText("A negative result I can prove", {
    x: M, y: 0.95, w: 11.9, h: 0.75, fontFace: H, fontSize: 34, bold: true,
    color: PAPER, margin: 0,
  });

  const left = [
    "Test set touched once, after everything was frozen",
    "Chronological splits with an embargo at every seam",
    "No dates or absolute prices in the LLM prompt",
    "Two control arms and bootstrap confidence intervals",
  ];
  const right = [
    "190 automated tests, including a hand-computed P&L check",
    "A script that re-derives every published number and hashes it",
    "Indicators written from scratch, no black-box library",
    "Cannot place a real order: paper trading is enforced in config",
  ];
  [left, right].forEach((col, ci) => {
    col.forEach((t, i) => {
      const x = M + ci * 6.1, y = 2.0 + i * 0.72;
      s.addShape(p.ShapeType.ellipse, { x, y: y + 0.07, w: 0.14, h: 0.14, fill: { color: GREEN } });
      s.addText(t, {
        x: x + 0.32, y, w: 5.4, h: 0.62, fontFace: B, fontSize: 13, color: "D6DAE2",
        margin: 0, lineSpacingMultiple: 1.12,
      });
    });
  });

  s.addText(
    "The machine-learning filter helped modestly. The LLM did not. " +
    "Most of the work went into being able to say that with a straight face.",
    { x: M, y: 5.15, w: 11.9, h: 0.8, fontFace: H, fontSize: 19, bold: true,
      color: PAPER, margin: 0, lineSpacingMultiple: 1.15 }
  );
  s.addText("Live paper trader and dashboard — demo", {
    x: M, y: 6.2, w: 11.9, h: 0.35, fontFace: B, fontSize: 13, color: MUTED,
    italic: true, margin: 0,
  });
  s.addNotes(
    "Close here, then switch to the dashboard. Two minutes maximum on the demo: " +
    "the Live tab, then Results, then one judge decision with its written reason."
  );
}

p.writeFile({ fileName: process.argv[2] || "presentation.pptx" }).then(f => console.log("wrote", f));
