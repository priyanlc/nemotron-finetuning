#!/usr/bin/env python3
"""Sample one served model on a puzzle JSONL and record, per puzzle, whether its first boxed answer is correct.

The sampling pass of the STaR recipe (Zelikman et al., 2022): the model solves the training puzzles on its own,
and only the final answers are checked. Each generation is appended to --out the moment it completes, so a run of
many hours survives a dropped connection, and --resume skips the puzzles already in --out. Greedy decoding, the
chat template with thinking on, and the per-request budget rule are the same as compare_puzzles.py.

  docker exec -it lightning-lora python3 /shared/nemotron-finetuning/tools/sample_traces.py --selftest
  docker exec -it lightning-lora python3 /shared/nemotron-finetuning/tools/sample_traces.py \\
      --data $S/data/cipher-train.jsonl --n 12 --out $S/logs/star-smoke.jsonl                  # smoke test
  docker exec -d lightning-lora bash -c "python3 /shared/nemotron-finetuning/tools/sample_traces.py \\
      --data $S/data/cipher-train.jsonl --out $S/logs/star-cipher-7680.jsonl --concurrency 32 \\
      > $S/logs/star-cipher-7680.log 2>&1"                                                     # the full run, hours

Input rows: {"id", "category", "prompt", "answer"}. Output lines: {"id", "category", "model", "finish", "tokens",
"budget", "first_box", "first_box_ok", "last_box_ok", "text"}; a .json summary is written beside --out at the end.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_puzzles import (DEFAULT_MAX_MODEL_LEN, MODEL, all_boxed_answers, answers_match, call,  # noqa: E402
                             extract_boxed_answer, select_rows, served_max_model_len)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="lightning-base", help="the served model name to sample")
    ap.add_argument("--data", required=False, help="JSONL with id, category, prompt, answer")
    ap.add_argument("--n", type=int, default=0, help="rows to sample, spread evenly over the file; 0 = all")
    ap.add_argument("--max-tokens", type=int, default=7680, help="generation cap; the competition's budget")
    ap.add_argument("--max-model-len", type=int, default=0, help="server context window; 0 = ask the server")
    ap.add_argument("--timeout", type=int, default=0, help="seconds per request; 0 = max(3600, 2 * --max-tokens)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", required=False, help="output JSONL, appended to as generations complete")
    ap.add_argument("--resume", action="store_true", help="skip ids already present in --out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest and not (a.data and a.out):
        ap.error("--data and --out are required")
    return a


def classify(text, answer):
    """First and last boxed answers of a generation, checked against the gold answer."""
    boxes = all_boxed_answers(text)
    first = boxes[0] if boxes else None
    return {"first_box": first, "first_box_ok": bool(first) and answers_match(first, answer),
            "last_box_ok": answers_match(extract_boxed_answer(text), answer)}


def done_ids(path):
    """Ids already written to an output file, for --resume."""
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def summarize(results, a, seconds, max_model_len):
    cats = sorted({r["category"] for r in results})
    per = {c: {"n": 0, "finished": 0, "first_box_ok": 0, "last_box_ok": 0, "truncated": 0} for c in cats}
    for r in results:
        p = per[r["category"]]
        p["n"] += 1
        p["finished"] += r["finish"] != "length"
        p["first_box_ok"] += bool(r["first_box_ok"])
        p["last_box_ok"] += bool(r["last_box_ok"])
        p["truncated"] += r["finish"] == "length"
    total = {k: sum(p[k] for p in per.values()) for k in ("n", "finished", "first_box_ok", "last_box_ok", "truncated")}
    return {**total, "per_category": per, "seconds": seconds, "model": a.model, "max_tokens": a.max_tokens,
            "max_model_len": max_model_len, "data": a.data, "out": a.out}


def selftest():
    c = classify("We need \\boxed{}.\nSo \\boxed{dragon reads under forest}.</think>\\boxed{dragon reads under forest}",
                 "dragon reads under forest")
    assert c == {"first_box": "dragon reads under forest", "first_box_ok": True, "last_box_ok": True}, c
    c = classify("first \\boxed{teacher draws the secret} then \\boxed{teacher draws secret}", "teacher draws secret")
    assert c["first_box"] == "teacher draws the secret" and not c["first_box_ok"] and c["last_box_ok"]
    assert classify("no box at all", "1") == {"first_box": None, "first_box_ok": False, "last_box_ok": False}
    assert classify("\\boxed{154.6} ... \\boxed{157}", "154.62") == {"first_box": "154.6", "first_box_ok": True,
                                                                     "last_box_ok": False}
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": "a", "category": "cipher"}) + "\n\n" + json.dumps({"id": "b", "category": "cipher"}) + "\n")
    assert done_ids(f.name) == {"a", "b"}
    os.unlink(f.name)
    assert done_ids(f.name) == set()

    class A:
        model, max_tokens, data, out = "m", 7680, "d", "o"
    s = summarize([{"category": "cipher", "finish": "stop", "first_box_ok": True, "last_box_ok": True},
                   {"category": "cipher", "finish": "length", "first_box_ok": False, "last_box_ok": False},
                   {"category": "gravity", "finish": "stop", "first_box_ok": False, "last_box_ok": True}], A(), 1.0, 8192)
    assert s["n"] == 3 and s["finished"] == 2 and s["first_box_ok"] == 1 and s["last_box_ok"] == 2 and s["truncated"] == 1
    assert s["per_category"]["cipher"] == {"n": 2, "finished": 1, "first_box_ok": 1, "last_box_ok": 1, "truncated": 1}
    print("selftest OK")


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    a.ids_from, a.only_truncated = None, False  # select_rows' other mode
    rows = select_rows([json.loads(l) for l in open(a.data) if l.strip()], a)
    for r in rows:
        if "answer" not in r:
            raise SystemExit(f"row {r.get('id')} has no 'answer' field; --data must carry the gold answers")
    skip = done_ids(a.out) if a.resume else set()
    todo = [r for r in rows if r["id"] not in skip]
    models = call(a.url, "/models")["data"]
    served = [m["id"] for m in models]
    if a.model not in served:
        raise SystemExit(f"'{a.model}' isn't served (have {served}).")
    max_model_len = a.max_model_len or served_max_model_len(models, a.model)
    if not max_model_len:
        max_model_len = DEFAULT_MAX_MODEL_LEN
        print(f"the server does not report its context window; assuming --max-model-len {max_model_len}")
    timeout = a.timeout or max(3600, 2 * a.max_tokens)

    def render(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)

    print(f"sampling {len(todo)} of {len(rows)} puzzles with {a.model} ({len(skip)} already in {a.out}), "
          f"cap {a.max_tokens} tokens, server window {max_model_len}, timeout {timeout} s per request, "
          f"concurrency {a.concurrency}", flush=True)
    if not todo:
        return

    def job(row):
        text = render(row["prompt"])
        budget = min(a.max_tokens, max_model_len - len(tok(text, add_special_tokens=False)["input_ids"]))
        r = call(a.url, "/completions", {"model": a.model, "prompt": text, "max_tokens": budget, "temperature": 0},
                 timeout=timeout)
        ch = r["choices"][0]
        return {"id": row["id"], "category": row.get("category"), "model": a.model, "finish": ch.get("finish_reason"),
                "tokens": r.get("usage", {}).get("completion_tokens"), "budget": budget,
                **classify(ch["text"], row["answer"]), "text": ch["text"]}

    t0, results, lock = time.time(), [], threading.Lock()
    counts = {"finished": 0, "first_box_ok": 0, "truncated": 0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as out, ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = [pool.submit(job, row) for row in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            with lock:
                out.write(json.dumps(res) + "\n")
                out.flush()
                results.append(res)
                counts["finished"] += res["finish"] != "length"
                counts["first_box_ok"] += bool(res["first_box_ok"])
                counts["truncated"] += res["finish"] == "length"
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} generations ({time.time() - t0:.0f} s)   finished {counts['finished']}   "
                      f"first box correct {counts['first_box_ok']}   truncated {counts['truncated']}", flush=True)

    summary = summarize(results, a, time.time() - t0, max_model_len)
    summary["resumed_from"] = len(skip)
    base = os.path.splitext(a.out)[0]
    with open(base + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{summary['n']} sampled this run: finished {summary['finished']}, first box correct "
          f"{summary['first_box_ok']}, truncated {summary['truncated']}; wrote {a.out} and {base}.json")


if __name__ == "__main__":
    main()
