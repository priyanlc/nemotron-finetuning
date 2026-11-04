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
Options: --url, --base-name, --adapter-name, --n rows to score (0 = all), --max-tokens (7680), --max-model-len (0 = ask
the server), --ids-from and --only-truncated (rescore an earlier run's puzzles, or just the ones that hit the cap),
--timeout, --concurrency (16), --out (CSV per row; a .json summary is written next to it), --selftest (checks the
statistics, needs no server).

Rescoring at a higher cap (the server must be started with a larger --max-model-len first):
  compare_puzzles.py --ids-from logs/compare-300.csv --only-truncated --max-tokens 16384 --out logs/compare-300-cap16k.csv
then cap_sweep.py merges the runs into one table per cap.
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
DEFAULT_MAX_MODEL_LEN = 8192  # used only when the server does not report its --max-model-len


def parse_args():
    ap = argparse.ArgumentParser(description="Score base and adapter on puzzles through a vLLM server.")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--base-name", default="lightning-base")
    ap.add_argument("--adapter-name", default="lightning-puzzles")
    ap.add_argument("--data", default="/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark/data/test.jsonl")
    ap.add_argument("--n", type=int, default=0, help="rows to score, 0 = all")
    ap.add_argument("--ids-from", default=None, metavar="CSV",
                    help="score the puzzles of an earlier run's per-row CSV, in its order, instead of --n")
    ap.add_argument("--only-truncated", action="store_true",
                    help="with --ids-from: keep only the puzzles where the base or the adapter hit the token limit")
    ap.add_argument("--max-tokens", type=int, default=7680, help="generation budget per request")
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="the server's context window; 0 reads it from /models (falls back to %d)" % DEFAULT_MAX_MODEL_LEN)
    ap.add_argument("--timeout", type=int, default=0,
                    help="seconds per request; 0 = max(3600, 2 * --max-tokens), enough for one slow stream at the cap")
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


def all_boxed_answers(text):
    """Every non-empty \\boxed{...} in order of appearance (flat braces only); for the early-box analysis."""
    if text is None:
        return []
    return [m.strip() for m in re.findall(r"\\boxed\{([^}]*)\}", text) if m.strip()]


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


def select_rows(rows, a):
    """The puzzles to score: an earlier run's ids (optionally only the truncated ones), or an evenly spread --n."""
    if a.ids_from:
        by_id = {r["id"]: r for r in rows}
        wanted = []
        with open(a.ids_from, newline="") as f:
            for prev in csv.DictReader(f):
                if a.only_truncated and "length" not in (prev.get("base_finish"), prev.get("adapter_finish")):
                    continue
                if prev["id"] not in by_id:
                    raise SystemExit(f"id {prev['id']} from {a.ids_from} is not in {a.data}")
                wanted.append(by_id[prev["id"]])
        if not wanted:
            raise SystemExit(f"nothing to score: no rows selected from {a.ids_from}")
        return wanted
    if a.n:  # an evenly spread subset, so every category is represented (test.jsonl is grouped by category)
        return rows[::max(1, len(rows) // a.n)][:a.n]
    return rows


def served_max_model_len(models, name):
    """The context window vLLM reports for a served name in /v1/models, or None."""
    for m in models:
        if m.get("id") == name and m.get("max_model_len"):
            return int(m["max_model_len"])
    return None


def selftest():
    p = mcnemar_exact(80, 53)
    d, lo, hi = paired_ci(80, 53, 1000)
    print(f"HellaSwag run check: 80 fixed / 53 broken of 1000 -> gain {d:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f}, exact McNemar p = {p:.3f}")
    assert 0.020 < p < 0.030 and abs(lo - 0.005) < 0.002 and abs(hi - 0.049) < 0.002
    assert extract_boxed_answer("so <think>...</think>\n\\boxed{10010111}") == "10010111"
    assert extract_boxed_answer("\\boxed{}\n then \\boxed{cat imagines book}") == "cat imagines book"
    assert all_boxed_answers("\\boxed{}\n \\boxed{a b}, later \\boxed{c}") == ["a b", "c"]
    assert answers_match("154.6", "154.62") and not answers_match("157", "154.62") and answers_match("XXXVIII", "xxxviii")
    assert extract_boxed_answer("no box here") is None and not answers_match(None, "1")
    assert served_max_model_len([{"id": "x", "max_model_len": 32768}, {"id": "y"}], "x") == 32768
    assert served_max_model_len([{"id": "y"}], "y") is None

    class A:  # select_rows on an in-memory CSV
        ids_from, only_truncated, n, data = None, False, 2, "rows"
    rows = [{"id": str(i), "category": "c", "prompt": "p", "answer": "1"} for i in range(6)]
    assert [r["id"] for r in select_rows(rows, A())] == ["0", "3"]
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "base_finish", "adapter_finish"])
        w.writeheader()
        w.writerows([{"id": "4", "base_finish": "length", "adapter_finish": "stop"},
                     {"id": "1", "base_finish": "stop", "adapter_finish": "stop"},
                     {"id": "2", "base_finish": "stop", "adapter_finish": "length"}])
    A.ids_from = f.name
    assert [r["id"] for r in select_rows(rows, A())] == ["4", "1", "2"]
    A.only_truncated = True
    assert [r["id"] for r in select_rows(rows, A())] == ["4", "2"]
    os.unlink(f.name)
    print("selftest OK")


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = select_rows([json.loads(l) for l in open(a.data) if l.strip()], a)
    models = call(a.url, "/models")["data"]
    served = [m["id"] for m in models]
    for name in (a.base_name, a.adapter_name):
        if name not in served:
            raise SystemExit(f"'{name}' isn't served (have {served}). Check --served-model-name and --lora-modules.")
    max_model_len = a.max_model_len or served_max_model_len(models, a.base_name)
    if not max_model_len:
        max_model_len = DEFAULT_MAX_MODEL_LEN
        print(f"the server does not report its context window; assuming --max-model-len {max_model_len}")
    timeout = a.timeout or max(3600, 2 * a.max_tokens)

    def render(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)

    longest_prompt = max(len(tok(render(r["prompt"]), add_special_tokens=False)["input_ids"]) for r in rows)
    if longest_prompt + a.max_tokens > max_model_len:
        print(f"note: --max-tokens {a.max_tokens} plus the longest prompt ({longest_prompt} tokens) exceeds the "
              f"server's window of {max_model_len}; the budget is trimmed per request, so the effective cap is "
              f"{max_model_len - longest_prompt} to {max_model_len - min(len(tok(render(r['prompt']), add_special_tokens=False)['input_ids']) for r in rows)} tokens")
    print(f"scoring {len(rows)} puzzles, cap {a.max_tokens} tokens, server window {max_model_len}, "
          f"timeout {timeout} s per request, concurrency {a.concurrency}", flush=True)

    def job(row, model):
        text = render(row["prompt"])
        budget = min(a.max_tokens, max_model_len - len(tok(text, add_special_tokens=False)["input_ids"]))
        r = call(a.url, "/completions", {"model": model, "prompt": text, "max_tokens": budget, "temperature": 0},
                 timeout=timeout)
        ch = r["choices"][0]
        pred = extract_boxed_answer(ch["text"])
        return {"id": row["id"], "model": model, "pred": pred, "ok": answers_match(pred, row["answer"]),
                "finish": ch.get("finish_reason"), "tokens": r.get("usage", {}).get("completion_tokens"),
                "budget": budget, "text": ch["text"]}

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
               "seconds": time.time() - t0, "max_tokens": a.max_tokens, "max_model_len": max_model_len,
               "ids_from": a.ids_from, "only_truncated": a.only_truncated,
               "base_name": a.base_name, "adapter_name": a.adapter_name}
    with open(a.out.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(a.out.rsplit(".", 1)[0] + "-generations.jsonl", "w") as f:
        for r in results.values():
            f.write(json.dumps(r) + "\n")
    print(f"Per-row CSV: {a.out}; summary .json and -generations.jsonl beside it")


if __name__ == "__main__":
    main()
