#!/usr/bin/env python3
"""Draws the three diagrams of README.md's Data section as SVG files next to this script. Plain Python, no packages.

    python3 assets/diagrams.py

The PNGs beside them were rendered from the SVGs at their own size (1920 px wide) with headless Chrome:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \\
        --user-data-dir=/tmp/chrome-diagrams --force-device-scale-factor=1 --window-size=1920,<height> \\
        --screenshot=<name>.png <name>.html        # <name>.html: the SVG pasted into a page with zero margins
Any SVG renderer (rsvg-convert, Inkscape, a browser's save-as) gives the same picture.
"""
import pathlib
from xml.sax.saxutils import escape

HERE = pathlib.Path(__file__).resolve().parent
SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "Menlo, Monaco, 'Courier New', monospace"
INK, QUIET, EDGE = "#1f1f1f", "#5a5a5a", "#333333"
BLUE_F, BLUE_S = "#dce8f8", "#2f4f7f"                        # data files
GREY_F, GREY_S = "#f3f3f3", "#333333"                        # processing steps
GREEN_F, GREEN_S, GREEN_T = "#e6f4e7", "#2e7d32", "#2e7d32"  # checks, the trained span
ORANGE_F, ORANGE_S = "#fdebd3", "#9a6b2a"                    # the Spark


def text(x, y, s, size=24, weight=400, fill=INK, family=SANS, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def mono(x, y, s, size=24, weight=400, fill=INK, anchor="start"):
    return text(x, y, s, size, weight, fill, MONO, anchor)


def box(x, y, w, h, fill="#ffffff", stroke=EDGE, sw=2.5, rx=14):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def line(x1, y1, x2, y2, stroke=EDGE, sw=2.5):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"/>'


def arrow(x1, y1, x2, y2, stroke=EDGE, sw=2.5):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" marker-end="url(#arrow)"/>'


def path(d, stroke=EDGE, sw=2.5):
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>'


def svg(width, height, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{escape(title)}">\n<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{EDGE}"/></marker></defs>\n'
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n' + "\n".join(body) + "\n</svg>\n")


def write(name, width, height, body, title):
    out = HERE / name
    out.write_text(svg(width, height, body, title))
    print("wrote", out)


def data_pipeline():
    """01: train.csv -> classify -> split -> build -> three files -> check -> read in place by the containers."""
    W, H = 1920, 1000
    b = [text(48, 66, "How train.csv becomes three JSONL files", 40, 700),
         text(48, 108, "data/make_puzzle_jsonl.py on the Spark host, in the experiment's virtual environment (README steps 3 to 5); both containers then read the files where they are", 24, 400, QUIET)]
    ya, ha = 160, 340
    boxes = [(48, 260), (360, 380), (792, 470), (1314, 558)]   # x, w; the 52 px gaps hold the arrows
    x, w = boxes[0]
    b += [box(x, ya, w, ha, BLUE_F, BLUE_S), mono(x + 20, ya + 48, "train.csv", 30, 700),
          text(x + 20, ya + 100, "9,500 puzzles", 24), text(x + 20, ya + 136, "id, prompt, answer", 24),
          text(x + 20, ya + 172, "six categories,", 24), text(x + 20, ya + 208, "about 1,580 each", 24),
          text(x + 20, ya + 260, "Kaggle, CC BY 4.0", 22, 400, QUIET), text(x + 20, ya + 292, "stays out of git", 22, 400, QUIET)]
    x, w = boxes[1]
    b += [box(x, ya, w, ha, GREY_F, GREY_S), text(x + 20, ya + 46, "1 · classify", 28, 700), text(x + 20, ya + 78, "by keywords in the prompt", 22, 400, QUIET)]
    for i, (c, n) in enumerate([("bit_ops", "1,602"), ("cipher", "1,576"), ("gravity", "1,597"), ("numeral", "1,576"), ("symbol", "1,555"), ("unit_conv", "1,594")]):
        b += [mono(x + 20, ya + 118 + 36 * i, c, 24), text(x + w - 20, ya + 118 + 36 * i, n, 24, 400, INK, SANS, "end")]
    x, w = boxes[2]
    b += [box(x, ya, w, ha, GREY_F, GREY_S), text(x + 20, ya + 46, "2 · split with fixed seeds", 28, 700)]
    for i, (l, r) in enumerate([("val: 10 per category, seed 0", "60"), ("test: 100 per category, seed 1", "600"), ("train: the rest, shuffled, seed 2", "8,840")]):
        b += [text(x + 20, ya + 104 + 44 * i, l, 24), text(x + w - 20, ya + 104 + 44 * i, r, 26, 700, INK, SANS, "end")]
    b += [text(x + 20, ya + 260, "no puzzle lands in two files;", 22, 400, QUIET), text(x + 20, ya + 292, "rerunning gives the same files", 22, 400, QUIET)]
    x, w = boxes[3]
    b += [box(x, ya, w, ha, GREY_F, GREY_S), text(x + 20, ya + 46, "3 · build each row", 28, 700),
          text(x + 20, ya + 100, "prompt", 24, 700), text(x + 118, ya + 100, "= puzzle + boxed-answer instruction", 24),
          text(x + 118, ya + 134, "(+ the vocabulary hint, ciphers only)", 22, 400, QUIET),
          text(x + 20, ya + 184, "completion", 24, 700), mono(x + 40, ya + 218, "<think> trace </think> \\boxed{answer}", 22),
          text(x + 20, ya + 266, "trace computed from the puzzle's examples", 22, 400, QUIET),
          text(x + 20, ya + 296, "where a rule can be read off, else a template", 22, 400, QUIET)]
    for (x1, w1), (x2, w2) in zip(boxes, boxes[1:]):
        b.append(arrow(x1 + w1 + 6, ya + ha / 2, x2 - 8, ya + ha / 2))
    yb, hb = 560, 150
    files = [(48, 560, "train.jsonl", "8,840 rows", "the training set: 8 rows per optimiser step"),
             (660, 560, "val.jsonl", "60 rows", "the [val] loss, every 20 steps"),
             (1272, 600, "test.jsonl", "600 rows", "the scorer's puzzles, never trained on")]
    for x, w, name, rows, what in files:
        b += [box(x, yb, w, hb, BLUE_F, BLUE_S), mono(x + 20, yb + 50, name, 30, 700), text(x + w - 20, yb + 50, rows, 26, 700, INK, SANS, "end"),
              text(x + 20, yb + 104, what, 24, 400, QUIET)]
    xa4 = boxes[3][0] + boxes[3][1] / 2
    centers = [x + w / 2 for x, w, *_ in files]
    b += [line(xa4, ya + ha, xa4, 530), line(centers[0], 530, xa4, 530)]
    for c in centers:
        b.append(arrow(c, 530, c, yb - 4))
    yc, hc = 760, 190
    b += [box(48, yc, 1140, hc, GREEN_F, GREEN_S), text(68, yc + 46, "4 · check:", 28, 700), mono(214, yc + 46, "python data/make_puzzle_jsonl.py --check", 26, 700, GREEN_T),
          text(68, yc + 96, "renders all 9,500 rows with the Lightning chat template, the way AutoModel will, and confirms that each", 23),
          text(68, yc + 130, "training sequence is the served prompt followed by the completion, byte for byte", 23),
          text(68, yc + 168, "longest sequence 492 tokens, mean 240: seq_length 2048 is a cap nothing reaches", 22, 400, QUIET),
          box(1240, yc, 632, hc, ORANGE_F, ORANGE_S), text(1260, yc + 46, "5 · read in place by both containers", 28, 700),
          mono(1260, yc + 96, "~/nemo-shared/nemotron-finetuning/experiments/", 18), mono(1260, yc + 124, "lightning-30b-lora-reasoning-puzzles-dgx-spark/data/", 18),
          text(1260, yc + 168, "= /shared/nemotron-finetuning/… in both containers", 22, 400, QUIET)]
    b += [line(centers[0], yb + hb, centers[0], 735), line(centers[1], yb + hb, centers[1], 735), line(centers[2], yb + hb, centers[2], 735),
          line(centers[0], 735, 1556, 735), arrow(618, 735, 618, yc - 4), arrow(1556, 735, 1556, yc - 4)]
    write("01-data-pipeline.svg", W, H, b, "How train.csv becomes three JSONL files")


def training_sequence():
    """02: one rendered training sequence, the served prompt and the trained span."""
    W, H = 1920, 1070
    b = [text(48, 66, "One puzzle, one training sequence, one served prompt", 40, 700),
         text(48, 108, "Row ca4d10f5 of train.jsonl as the Lightning chat template renders it. The training text begins with exactly what the scorer sends at evaluation.", 24, 400, QUIET)]
    bx, bw, rx = 330, 1070, 1470

    def band(y, h, fill, stroke, lines, label):
        out = [box(bx, y, bw, h, fill, stroke, 2.5, 10), text(bx - 30, y + h / 2 + 8, label, 24, 700, INK, SANS, "end")]
        for i, (s, kind) in enumerate(lines):
            col = {"tok": QUIET, "text": INK, "key": GREEN_T}[kind]
            out.append(mono(bx + 24, y + 38 + 30 * i, s, 22, 700 if kind == "key" else 400, col))
        return out

    def note(y, s):
        return text(bx + bw - 24, y, s, 20, 400, QUIET, SANS, "end")

    y1 = 150
    b += band(y1, 90, "#eef3fb", BLUE_S, [("<|im_start|>system", "tok"), ("<|im_end|>", "tok")], "system turn")
    b += [note(y1 + 38, "empty: the template adds it whenever no system"), note(y1 + 66, "message is given, at training and at inference alike")]
    y2 = y1 + 90 + 12
    user = [("<|im_start|>user", "tok"),
            ("In Alice's Wonderland, the gravitational constant has been secretly changed.", "text"),
            ("Here are some example observations:", "text"),
            ("For t = 1.69s, distance = 9.47 m", "text"), ("For t = 1.58s, distance = 8.28 m", "text"),
            ("For t = 2.26s, distance = 16.93 m", "text"), ("For t = 3.49s, distance = 40.38 m", "text"),
            ("Now, determine the falling distance for t = 2.81s given d = 0.5*g*t^2.", "text"),
            ("Please put your final answer inside `\\boxed{}`.", "text"), ("For example: `\\boxed{your answer}`", "text"),
            ("<|im_end|>", "tok")]
    b += band(y2, 380, "#eef3fb", BLUE_S, user, "user turn")
    b += [note(y2 + 38 + 30 * 4, "the puzzle, as train.csv gives it"),
          note(y2 + 38 + 30 * 9, "the boxed-answer instruction the script adds"),
          note(y2 + 38 + 30 * 10, "(a cipher prompt also gets the vocabulary hint, just before it)")]
    y3 = y2 + 380 + 12
    b += band(y3, 90, "#eef3fb", BLUE_S, [("<|im_start|>assistant", "tok"), ("<think>", "key")], "assistant turn")
    b += [note(y3 + 52, "the served prompt ends here, with an open <think>")]
    y4 = y3 + 90 + 12
    b += band(y4, 200, GREEN_F, GREEN_S, [("Using d = 0.5*g*t^2: g = 2d/t^2 = 2*9.47/1.69^2 = 6.6314.", "text"),
                                          ("Apply to find d for the new t. Answer: 26.18", "text"),
                                          ("</think>", "key"), ("\\boxed{26.18}", "key"), ("<|im_end|>", "tok")], "")
    b += [note(y4 + 38 + 30 * 3, "the completion from train.jsonl, after its opening <think>")]

    def bracket(ya, yb, color):
        return [path(f"M{rx - 40} {ya} h 14 v {yb - ya} h -14", color, 3)]

    b += bracket(y1, y3 + 90, BLUE_S) + bracket(y4, y4 + 200, GREEN_S)
    mid = (y1 + y3 + 90) / 2
    b += [text(rx, mid - 60, "the served prompt", 28, 700, BLUE_S),
          text(rx, mid - 20, "what the scorer sends: the user", 22), text(rx, mid + 10, "message rendered with the generation", 22),
          text(rx, mid + 40, "prompt and thinking on, through", 22), text(rx, mid + 70, "/v1/completions, temperature 0,", 22),
          text(rx, mid + 100, "up to 7,680 new tokens", 22)]
    b += [text(rx, y4 + 44, "the trained span", 28, 700, GREEN_T),
          text(rx, y4 + 84, "the completion's tail plus <|im_end|>:", 22), text(rx, y4 + 114, "the model is scored only on predicting", 22),
          text(rx, y4 + 144, "these tokens, so they are the only", 22), text(rx, y4 + 174, "ones it learns from; the prompt is", 22),
          text(rx, y4 + 204, "read as context but never scored", 22)]
    yn = y4 + 200 + 50
    b += [text(48, yn, "Training text = served prompt + completion tail + <|im_end|>. Step 5 checks it on all 9,500 rows; step 8 repeats the check inside the container,", 23),
          text(48, yn + 34, "against AutoModel's dataset class and its loss mask. With the dataset's default, use_hf_chat_template: false, a space would follow the prompt instead.", 23)]
    write("02-training-sequence.svg", W, H, b, "One puzzle, one training sequence, one served prompt")


def trace_rules():
    """03: which categories get a computed trace, and the rule."""
    W, H = 1920, 820
    b = [text(48, 66, "Which reasoning trace does a puzzle get?", 40, 700),
         text(48, 108, "train.jsonl, 8,840 rows: a trace computed from the puzzle's own examples and answer where a rule can be read off them, otherwise one fixed sentence per category", 24, 400, QUIET)]
    rows = [("cipher", 1466, 1466, ["substitution map read off the example pairs, completed from the answer where a",
                                    "letter is unmapped; the first word decrypted character by character, then the",
                                    "whole text, then the answer"]),
            ("gravity", 1487, 1487, ["g = 2d / t² from the first observation, to four decimals, then the instruction to",
                                     "apply it to the new time, and the answer", ""]),
            ("symbol", 224, 1445, ["two integers joined by one symbol, and the answer equals one of a + b, a − b,",
                                   "b − a, a × b, a // b, the digits of a then b, or the sum of all digits:",
                                   "that operation, named and computed"]),
            ("bit_ops", 92, 1492, ["per output bit, the simplest function of the input bits fitting every pair",
                                   "(a constant, a bit, its NOT, the XOR, AND or OR of two bits), applied to the",
                                   "new input; kept only if the result is the answer"]),
            ("unit_conv", 0, 1484, ["a factor trace exists in the script but expects 'number unit -> number';",
                                    "these prompts read '7.27 m becomes 13.65', so every row gets the template", ""]),
            ("numeral", 0, 1466, ["no computed trace: every row gets the template", "", ""])]
    x0, bar_w, y0, rh, tx = 250, 460, 160, 100, 930
    for i, (name, k, n, lines) in enumerate(rows):
        y = y0 + rh * i
        b += [mono(48, y + 48, name, 26, 700), box(x0, y + 20, bar_w, 44, "#e4e4e4", "#9a9a9a", 1.5, 6)]
        if k:
            b.append(box(x0, y + 20, max(6, bar_w * k / n), 44, GREEN_F, GREEN_S, 2, 6))
        b.append(text(x0 + bar_w + 20, y + 49, f"{k:,} of {n:,}", 24, 700, GREEN_T if k else QUIET))
        for j, s in enumerate(lines):
            if s:
                b.append(text(tx, y + 30 + 27 * j, s, 21))
    yl = y0 + rh * 6 + 16
    b += [box(x0, yl - 4, 36, 24, GREEN_F, GREEN_S, 2, 4), text(x0 + 48, yl + 15, "trace computed from the puzzle's examples and answer", 22),
          box(x0 + 660, yl - 4, 36, 24, "#e4e4e4", "#9a9a9a", 1.5, 4), text(x0 + 708, yl + 15, "the category's one-sentence template", 22)]
    write("03-trace-rules.svg", W, H, b, "Which reasoning trace does a puzzle get?")


if __name__ == "__main__":
    data_pipeline()
    training_sequence()
    trace_rules()
