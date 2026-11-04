#!/usr/bin/env python3
"""Merge compare_puzzles.py runs made at increasing token caps into one table per cap, and measure how often a
generation that hit the cap had already boxed a correct answer.

Pass the per-row CSVs lowest cap first. The first run must cover the whole puzzle set; each later run may cover a
subset (typically the puzzles that hit the previous cap, scored with --ids-from --only-truncated) and replaces those
rows whole. Each CSV is read with its sibling .json (for max_tokens) and -generations.jsonl (for the early-box check).

  python3 cap_sweep.py logs/compare-300.csv                                   # one run: the early-box check only
  python3 cap_sweep.py logs/compare-300.csv logs/compare-300-cap16k.csv logs/compare-300-cap32k.csv --out results/cap-sweep.json

Standard library only; the answer and statistics functions are imported from compare_puzzles.py next to this file.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_puzzles import all_boxed_answers, answers_match, mcnemar_exact, paired_ci  # noqa: E402

CSV_FIELDS = ["id", "category", "answer", "base_pred", "base_ok", "base_tokens", "base_finish",
              "adapter_pred", "adapter_ok", "adapter_tokens", "adapter_finish"]


def parse_args():
    ap = argparse.ArgumentParser(description="Merge puzzle runs at increasing caps into one table per cap.")
    ap.add_argument("csvs", nargs="*", help="per-row CSVs from compare_puzzles.py, lowest cap first")
    ap.add_argument("--base-name", default="lightning-base", help="served base name, for runs whose .json lacks it")
    ap.add_argument("--adapter-name", default="lightning-puzzles")
    ap.add_argument("--out", default=None, help="write the per-cap summaries and early-box counts as JSON")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def sibling(path, suffix):
    return path.rsplit(".", 1)[0] + suffix


def load_run(path, base_name, adapter_name):
    """One run: its cap, its rows by id (in file order), its base generations by id, and the served names."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    try:
        with open(sibling(path, ".json")) as f:
            summary = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{sibling(path, '.json')} is missing; the cap is read from the summary's max_tokens")
    base = summary.get("base_name", base_name)
    gens = {}
    try:
        with open(sibling(path, "-generations.jsonl")) as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    if g.get("model") == base:
                        gens[g["id"]] = g
    except FileNotFoundError:
        pass
    return {"path": path, "cap": int(summary["max_tokens"]), "rows": {r["id"]: r for r in rows},
            "order": [r["id"] for r in rows], "gens": gens, "base_name": base,
            "adapter_name": summary.get("adapter_name", adapter_name)}


def table(order, state):
    """Per-category and overall accuracy, fixed/broke, gain, CI, p and truncation counts for one merged state."""
    cats = sorted({state[i]["category"] for i in order})
    per = {c: {"n": 0, "base": 0, "adapter": 0, "base_truncated": 0} for c in cats}
    fixed = broke = base_ok = ad_ok = 0
    trunc = {"base": 0, "adapter": 0}
    for i in order:
        r = state[i]
        b, l = int(r["base_ok"]), int(r["adapter_ok"])
        c = per[r["category"]]
        c["n"] += 1; c["base"] += b; c["adapter"] += l; c["base_truncated"] += r["base_finish"] == "length"
        base_ok += b; ad_ok += l
        fixed += (not b) and l; broke += b and not l
        trunc["base"] += r["base_finish"] == "length"; trunc["adapter"] += r["adapter_finish"] == "length"
    n = len(order)
    d, lo, hi = paired_ci(fixed, broke, n)
    return {"n": n, "base_acc": base_ok / n, "adapter_acc": ad_ok / n, "fixed": fixed, "broke": broke, "gain": d,
            "ci95": [lo, hi], "mcnemar_p": mcnemar_exact(fixed, broke), "per_category": per, "truncated": trunc}


def early_box(run):
    """Among base generations that hit this run's cap: no box at all, last box correct (what the scorer counted),
    a correct box anywhere, first box correct. Per category and in total."""
    out = {}
    for i in run["order"]:
        r, g = run["rows"][i], run["gens"].get(i)
        if g is None or g.get("finish") != "length":
            continue
        c = out.setdefault(r["category"], {"truncated": 0, "no_box": 0, "last_correct": 0, "any_correct": 0, "first_correct": 0})
        boxes = all_boxed_answers(g.get("text"))
        c["truncated"] += 1
        c["no_box"] += not boxes
        c["last_correct"] += bool(boxes) and answers_match(boxes[-1], r["answer"])
        c["any_correct"] += any(answers_match(b, r["answer"]) for b in boxes)
        c["first_correct"] += bool(boxes) and answers_match(boxes[0], r["answer"])
    if out:
        out["all"] = {k: sum(c[k] for c in out.values()) for k in next(iter(out.values()))}
    return out


def sweep(runs):
    """Overlay the runs in cap order; one table per cap, plus warnings about truncated rows that were not re-run."""
    runs = sorted(runs, key=lambda r: r["cap"])
    order = runs[0]["order"]
    state = dict(runs[0]["rows"])
    caps, warnings = [], []
    caps.append({"cap": runs[0]["cap"], "path": runs[0]["path"], "rows_scored": len(order), **table(order, state)})
    for run in runs[1:]:
        truncated_before = {i for i in order if state[i]["base_finish"] == "length"}
        unknown = [i for i in run["order"] if i not in state]
        if unknown:
            raise SystemExit(f"{run['path']}: {len(unknown)} ids are not in the first run ({unknown[:3]}...)")
        for i in run["order"]:
            state[i] = run["rows"][i]
        missed = sorted(truncated_before - set(run["order"]))
        if missed:
            warnings.append(f"cap {run['cap']}: {len(missed)} puzzles truncated at the previous cap were not re-run "
                            f"(e.g. {', '.join(missed[:3])}); their lower-cap results are kept")
        caps.append({"cap": run["cap"], "path": run["path"], "rows_scored": len(run["order"]), **table(order, state)})
    return caps, warnings


def print_sweep(caps, warnings):
    cats = sorted(caps[0]["per_category"])
    print(f"{'cap':>7}{'scored':>8}{'base':>8}{'adapter':>9}{'gain':>8}{'95% CI':>18}{'p':>8}{'base trunc':>12}")
    for c in caps:
        lo, hi = c["ci95"]
        print(f"{c['cap']:>7}{c['rows_scored']:>8}{c['base_acc']:>8.3f}{c['adapter_acc']:>9.3f}{c['gain']:>+8.3f}"
              f"{lo:>+9.3f} to {hi:+.3f}{c['mcnemar_p']:>8.3f}{c['truncated']['base']:>8}/{c['n']}")
    print(f"\nbase accuracy by category and cap ({', '.join(str(c['cap']) for c in caps)}); adapter in brackets")
    for cat in cats:
        cells = []
        for c in caps:
            p = c["per_category"][cat]
            cells.append(f"{p['base'] / p['n']:.2f} [{p['adapter'] / p['n']:.2f}]")
        print(f"{cat:12}" + "".join(f"{x:>16}" for x in cells))
    for w in warnings:
        print("warning:", w)


def print_early_box(run, eb):
    if not eb:
        print(f"\n{run['path']}: no base generations hit the cap (or the -generations.jsonl file is missing)")
        return
    print(f"\nBase generations that hit the {run['cap']}-token cap in {os.path.basename(run['path'])}:")
    print(f"{'category':12}{'truncated':>10}{'no box':>8}{'last ok':>9}{'any ok':>8}{'first ok':>10}")
    for cat in sorted(k for k in eb if k != "all") + ["all"]:
        c = eb[cat]
        print(f"{cat:12}{c['truncated']:>10}{c['no_box']:>8}{c['last_correct']:>9}{c['any_correct']:>8}{c['first_correct']:>10}")
    print("last ok = the box the scorer used was correct despite the cut; any ok = a correct box appears somewhere; "
          "first ok = the first box was correct (what the base would score if it stopped at its first answer)")


def write_runs(dirpath, name, cap, rows, gens):
    """Test helper: write a CSV, its .json and its -generations.jsonl the way compare_puzzles.py does."""
    path = os.path.join(dirpath, name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(rows)
    with open(sibling(path, ".json"), "w") as f:
        json.dump({"max_tokens": cap, "base_name": "b", "adapter_name": "a"}, f)
    with open(sibling(path, "-generations.jsonl"), "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    return path


def selftest():
    import tempfile
    def row(i, cat, ans, b_pred, b_ok, b_fin, a_pred, a_ok):
        return {"id": i, "category": cat, "answer": ans, "base_pred": b_pred, "base_ok": b_ok, "base_tokens": 0,
                "base_finish": b_fin, "adapter_pred": a_pred, "adapter_ok": a_ok, "adapter_tokens": 0, "adapter_finish": "stop"}
    with tempfile.TemporaryDirectory() as d:
        # cap 100: six puzzles, base truncated on three (p3 boxed the right answer first, then a wrong one; p4 no box;
        # p5 boxed right and kept going), adapter right on four
        low = write_runs(d, "low", 100, [
            row("p1", "x", "1", "1", 1, "stop", "1", 1), row("p2", "x", "2", "3", 0, "stop", "2", 1),
            row("p3", "y", "3", "9", 0, "length", "3", 1), row("p4", "y", "4", None, 0, "length", "0", 0),
            row("p5", "y", "5", "5", 1, "length", "5", 1), row("p6", "x", "6", "6", 1, "stop", "0", 0)], [
            {"id": "p3", "model": "b", "finish": "length", "text": "\\boxed{3} hmm \\boxed{9}"},
            {"id": "p4", "model": "b", "finish": "length", "text": "still thinking"},
            {"id": "p5", "model": "b", "finish": "length", "text": "\\boxed{5} but also"},
            {"id": "p1", "model": "b", "finish": "stop", "text": "\\boxed{1}"},
            {"id": "p3", "model": "a", "finish": "stop", "text": "\\boxed{3}"}])
        # cap 200: only the three truncated puzzles re-run; p3 and p5 now finish and are right, p4 still truncated
        high = write_runs(d, "high", 200, [
            row("p3", "y", "3", "3", 1, "stop", "3", 1), row("p4", "y", "4", None, 0, "length", "0", 0),
            row("p5", "y", "5", "5", 1, "stop", "5", 1)], [])
        runs = [load_run(high, "b", "a"), load_run(low, "b", "a")]  # out of order on purpose
        caps, warnings = sweep(runs)
        assert [c["cap"] for c in caps] == [100, 200] and not warnings
        assert caps[0]["base_acc"] == 3 / 6 and caps[0]["adapter_acc"] == 4 / 6 and caps[0]["truncated"]["base"] == 3
        assert caps[0]["fixed"] == 2 and caps[0]["broke"] == 1
        assert caps[1]["base_acc"] == 4 / 6 and caps[1]["adapter_acc"] == 4 / 6 and caps[1]["truncated"]["base"] == 1
        assert caps[1]["fixed"] == 1 and caps[1]["broke"] == 1 and caps[1]["rows_scored"] == 3
        assert caps[1]["per_category"]["y"] == {"n": 3, "base": 2, "adapter": 2, "base_truncated": 1}
        eb = early_box(runs[1])
        assert eb["y"] == {"truncated": 3, "no_box": 1, "last_correct": 1, "any_correct": 2, "first_correct": 2}, eb
        assert eb["all"] == eb["y"] and "x" not in eb
        # a higher cap that skips a truncated puzzle is warned about, not an error
        partial = write_runs(d, "partial", 300, [row("p4", "y", "4", "4", 1, "stop", "0", 0)], [])
        caps, warnings = sweep([load_run(low, "b", "a"), load_run(partial, "b", "a")])
        assert len(warnings) == 1 and "2 puzzles" in warnings[0] and caps[1]["base_acc"] == 4 / 6
        print_sweep(caps, warnings)
        print_early_box(runs[1], eb)
    print("selftest OK")


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    if not a.csvs:
        raise SystemExit("pass at least one per-row CSV from compare_puzzles.py (or --selftest)")
    runs = [load_run(p, a.base_name, a.adapter_name) for p in a.csvs]
    caps, warnings = sweep(runs)
    print_sweep(caps, warnings)
    early = {}
    for run in sorted(runs, key=lambda r: r["cap"]):
        eb = early_box(run)
        early[str(run["cap"])] = eb
        print_early_box(run, eb)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"caps": caps, "early_box": early, "warnings": warnings}, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
