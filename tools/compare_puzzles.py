#!/usr/bin/env python3
"""Compare the base model and a LoRA adapter on puzzle prompts through a running vLLM server, by generation.

For each row of a JSONL file ({"id", "category", "prompt", "answer"}) the script renders the prompt with the model's
chat template (reasoning on), asks /v1/completions for a greedy continuation from both served names, extracts the last
\\boxed{} answer and compares it with the expected answer (exact string, or 1% relative tolerance for numbers), the way
the Kaggle scorer does. It reports accuracy per category, answers fixed and broken, an exact McNemar test and a paired
95% interval for the accuracy gain.

Run inside the nemo-automodel container, which has transformers, the cached tokenizer and the host network:
  docker exec -it lightning-lora python3 /shared/nemotron-finetuning/tools/compare_puzzles.py --n 12   # defaults: this
  experiment's data/test.jsonl and logs/ under /shared/nemotron-finetuning/experiments/
Options: --url, --base-name, --adapter-name, --n rows to score (0 = all), --max-tokens (7680), --concurrency (16),
--out (CSV per row; a .json summary is written next to it), --selftest (checks the statistics, needs no server).
"""
import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
MAX_MODEL_LEN = 8192


def parse_args():
    ap = argparse.ArgumentParser(description="Score base and adapter on puzzles through a vLLM server.")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--base-name", default="lightning-base")
    ap.add_argument("--adapter-name", default="lightning-puzzles")
    ap.add_argument("--data", default="/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark/data/test.jsonl")
    ap.add_argument("--n", type=int, default=0, help="rows to score, 0 = all")
    ap.add_argument("--max-tokens", type=int, default=7680)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark/logs/compare_puzzles.csv")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def call(url, path, payload=None, timeout=3600):
    req = urllib.request.Request(url + path, method="POST" if payload else "GET",
                                 data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code} from {path}: {e.read().decode()[:500]}") from None


def extract_boxed_answer(text):
    """Last non-empty \\boxed{...}; nested braces handled for the final one."""
    if text is None:
        return None
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                content = text[start:i].strip()
                return content or None
            depth -= 1
    content = text[start:].strip()
    return content or None


def answers_match(pred, gold):
    """Exact string (case-insensitive) or 1% relative tolerance for numbers."""
    if pred is None:
        return False
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass
    return pred.strip().lower() == gold.strip().lower()


def mcnemar_exact(fixed, broke):
    """Two-sided exact McNemar p-value on the discordant pairs (binomial, p = 0.5)."""
    n, k = fixed + broke, min(fixed, broke)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_ci(fixed, broke, n):
    """Wald 95% interval for the paired accuracy difference (adapter minus base)."""
    d = (fixed - broke) / n
    se = math.sqrt(max(fixed + broke - (fixed - broke) ** 2 / n, 0)) / n
    return d, d - 1.96 * se, d + 1.96 * se


def selftest():
    p = mcnemar_exact(80, 53)
    d, lo, hi = paired_ci(80, 53, 1000)
    print(f"HellaSwag run check: 80 fixed / 53 broken of 1000 -> gain {d:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}, exact McNemar p = {p:.3f}")
    assert 0.020 < p < 0.030 and abs(lo - 0.005) < 0.002 and abs(hi - 0.049) < 0.002
    assert extract_boxed_answer("so <think>...</think>\n\\boxed{10010111}") == "10010111"
    assert extract_boxed_answer("\\boxed{}\n then \\boxed{cat imagines book}") == "cat imagines book"
    assert answers_match("154.6", "154.62") and not answers_match("157", "154.62") and answers_match("XXXVIII", "xxxviii")
    assert extract_boxed_answer("no box here") is None and not answers_match(None, "1")
    print("selftest OK")


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    if a.n:  # an evenly spread subset, so every category is represented (test.jsonl is grouped by category)
        rows = rows[::max(1, len(rows) // a.n)][:a.n]
    served = [m["id"] for m in call(a.url, "/models")["data"]]
    for name in (a.base_name, a.adapter_name):
        if name not in served:
            raise SystemExit(f"'{name}' isn't served (have {served}). Check --served-model-name and --lora-modules.")

    def render(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)

    def job(row, model):
        text = render(row["prompt"])
        budget = min(a.max_tokens, MAX_MODEL_LEN - len(tok(text, add_special_tokens=False)["input_ids"]))
        r = call(a.url, "/completions", {"model": model, "prompt": text, "max_tokens": budget, "temperature": 0})
        ch = r["choices"][0]
        pred = extract_boxed_answer(ch["text"])
        return {"id": row["id"], "model": model, "pred": pred, "ok": answers_match(pred, row["answer"]),
                "finish": ch.get("finish_reason"), "tokens": r.get("usage", {}).get("completion_tokens"),
                "text": ch["text"]}

    t0, results = time.time(), {}
    jobs = [(row, m) for row in rows for m in (a.base_name, a.adapter_name)]
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for i, res in enumerate(pool.map(lambda j: job(*j), jobs), 1):
            results[(res["id"], res["model"])] = res
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} generations ({time.time() - t0:.0f} s)", flush=True)

    cats = sorted({r["category"] for r in rows})
    per = {c: {"n": 0, "base": 0, "adapter": 0} for c in cats}
    fixed = broke = base_ok = ad_ok = 0
    out_rows = []
    for row in rows:
        b, l = results[(row["id"], a.base_name)], results[(row["id"], a.adapter_name)]
        c = per[row["category"]]
        c["n"] += 1; c["base"] += b["ok"]; c["adapter"] += l["ok"]
        base_ok += b["ok"]; ad_ok += l["ok"]
        fixed += (not b["ok"]) and l["ok"]; broke += b["ok"] and not l["ok"]
        out_rows.append({"id": row["id"], "category": row["category"], "answer": row["answer"],
                         "base_pred": b["pred"], "base_ok": int(b["ok"]), "base_tokens": b["tokens"], "base_finish": b["finish"],
                         "adapter_pred": l["pred"], "adapter_ok": int(l["ok"]), "adapter_tokens": l["tokens"], "adapter_finish": l["finish"]})
    n = len(rows)
    print(f"\n{'category':12}{'n':>5}{'base':>8}{'adapter':>9}")
    for c in cats:
        print(f"{c:12}{per[c]['n']:>5}{per[c]['base'] / per[c]['n']:>8.3f}{per[c]['adapter'] / per[c]['n']:>9.3f}")
    print(f"{'all':12}{n:>5}{base_ok / n:>8.3f}{ad_ok / n:>9.3f}")
    d, lo, hi = paired_ci(fixed, broke, n)
    p = mcnemar_exact(fixed, broke)
    trunc = {m: sum(1 for r in results.values() if r["model"] == m and r["finish"] == "length") for m in (a.base_name, a.adapter_name)}
    print(f"Adapter fixed {fixed} and broke {broke}: gain {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}), exact McNemar p = {p:.3f}")
    print(f"Hit the token limit: base {trunc[a.base_name]}, adapter {trunc[a.adapter_name]} of {n}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    summary = {"n": n, "base_acc": base_ok / n, "adapter_acc": ad_ok / n, "fixed": fixed, "broke": broke,
               "gain": d, "ci95": [lo, hi], "mcnemar_p": p, "per_category": per, "truncated": trunc,
               "seconds": time.time() - t0, "max_tokens": a.max_tokens}
    with open(a.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(a.out.rsplit(".", 1)[0] + "-generations.jsonl", "w") as f:
        for r in results.values():
            f.write(json.dumps(r) + "\n")
    print(f"Per-row CSV: {a.out}; summary .json and -generations.jsonl beside it")


if __name__ == "__main__":
    main()
