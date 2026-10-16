#!/usr/bin/env python3
"""Compare the base model and the LoRA adapter through a running vLLM server.

The vLLM server serves the base model under one name and the adapter under
another (--lora-modules). This script sends the same requests to both:

  1. Side-by-side text: greedy completions for a few prompts.
  2. HellaSwag accuracy: for each question, asks the server for the
     log-probability of each of the 4 endings (echo + logprobs) and picks the
     most likely, like lm-eval-harness (acc = total, acc_norm = per character).

It needs only the Python standard library plus `datasets` (for HellaSwag),
so run it from the nemo-automodel container, which already has `datasets`
and shares the network with the vLLM container (--network host):

  docker start -ai lightning-lora          # or docker exec -it lightning-lora bash
  python3 /shared/compare_vllm.py

Options: --url, --base-name, --adapter-name (must match the server),
--n questions to score (default 100, 0 to skip), --skip first validation rows
(default 64, used by the training run), --gen side-by-side examples,
--prompts a text file with one prompt per line, --out a JSON results file.
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request


def parse_args():
    ap = argparse.ArgumentParser(description="Compare base and adapter through a vLLM server.")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--base-name", default="lightning-base")
    ap.add_argument("--adapter-name", default="lightning-hellaswag")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--skip", type=int, default=64)
    ap.add_argument("--gen", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--prompts", help="text file, one prompt per line")
    ap.add_argument("--out", default="/shared/logs/compare_vllm.json")
    return ap.parse_args()


def call(url, path, payload=None, timeout=600):
    req = urllib.request.Request(url + path, method="POST" if payload else "GET",
                                 data=json.dumps(payload).encode() if payload else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{e.code} from {path}: {e.read().decode()[:500]}") from None


def clean(text):
    """lm-eval-harness's HellaSwag text cleanup."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def complete(a, model, prompt):
    r = call(a.url, "/completions", {"model": model, "prompt": prompt,
                                     "max_tokens": a.max_tokens, "temperature": 0})
    return r["choices"][0]["text"].strip()


def ending_logprobs(a, model, ctx, endings):
    """Log-probability of each ending after ctx, from the server's prompt logprobs."""
    prompts = [ctx + " " + e for e in endings]
    r = call(a.url, "/completions", {"model": model, "prompt": prompts, "max_tokens": 1,
                                     "temperature": 0, "echo": True, "logprobs": 0})
    totals = [0.0] * len(prompts)
    for ch in r["choices"]:
        i = ch["index"]
        lp = ch["logprobs"]
        start = len(ctx)
        # the echoed prompt tokens come first; the 1 generated token is last
        for tok_lp, off in zip(lp["token_logprobs"][:-1], lp["text_offset"][:-1]):
            if off >= start and tok_lp is not None:
                totals[i] += tok_lp
    per_char = [t / max(len(e), 1) for t, e in zip(totals, endings)]
    return totals, per_char


def hellaswag(a):
    from datasets import load_dataset

    rows = load_dataset("rowan/hellaswag", split="validation").select(range(a.skip, a.skip + a.n))
    t = {k: 0 for k in ("base_acc", "base_norm", "lora_acc", "lora_norm", "fixed", "broke")}
    details, t0 = [], time.time()
    for i, r in enumerate(rows):
        ctx, endings, gold = clean(r["ctx"]), [clean(e) for e in r["endings"]], int(r["label"])
        b_tot, b_norm = ending_logprobs(a, a.base_name, ctx, endings)
        l_tot, l_norm = ending_logprobs(a, a.adapter_name, ctx, endings)
        pick = lambda xs: max(range(4), key=xs.__getitem__)
        t["base_acc"] += pick(b_tot) == gold
        t["lora_acc"] += pick(l_tot) == gold
        t["base_norm"] += pick(b_norm) == gold
        t["lora_norm"] += pick(l_norm) == gold
        t["fixed"] += pick(b_norm) != gold and pick(l_norm) == gold
        t["broke"] += pick(b_norm) == gold and pick(l_norm) != gold
        details.append({"row": a.skip + i, "gold": gold, "base_pick": pick(b_norm),
                        "adapter_pick": pick(l_norm), "base_gold_logprob": b_tot[gold],
                        "adapter_gold_logprob": l_tot[gold]})
        if (i + 1) % 10 == 0 or i + 1 == a.n:
            print(f"  {i + 1}/{a.n}  acc_norm base {t['base_norm'] / (i + 1):.3f}  "
                  f"adapter {t['lora_norm'] / (i + 1):.3f}  ({time.time() - t0:.0f} s)")
    return t, details


def main():
    a = parse_args()
    served = [m["id"] for m in call(a.url, "/models")["data"]]
    print(f"Server models: {served}")
    for name in (a.base_name, a.adapter_name):
        if name not in served:
            raise SystemExit(f"'{name}' isn't served. Check --served-model-name and --lora-modules.")
    results = {"url": a.url, "base": a.base_name, "adapter": a.adapter_name}

    prompts = [l.strip() for l in open(a.prompts) if l.strip()] if a.prompts else []
    if not prompts and a.gen > 0:
        from datasets import load_dataset
        rows = load_dataset("rowan/hellaswag", split="validation")
        rows = rows.select(range(a.skip + a.n, a.skip + a.n + a.gen))
        prompts = [clean(r["ctx"]) for r in rows]
    side = []
    for p in prompts:
        b, l = complete(a, a.base_name, p), complete(a, a.adapter_name, p)
        side.append({"prompt": p, "base": b, "adapter": l})
        print(f"\n--- PROMPT: {p}\n  base:    {b}\n  adapter: {l}")
    results["side_by_side"] = side
    if side and all(s["base"] == s["adapter"] for s in side):
        print("\nNOTE: base and adapter gave identical text for every prompt. "
              "The adapter may not be applied; check the server log for LoRA warnings.")

    if a.n > 0:
        print(f"\nHellaSwag validation rows {a.skip}-{a.skip + a.n - 1}:")
        try:
            t, details = hellaswag(a)
        except (RuntimeError, KeyError, TypeError) as e:
            print(f"HellaSwag scoring failed: {e}\n(The server must return prompt logprobs for echo=True.)")
        else:
            n = a.n
            print(f"\n{'':22}{'base':>8}{'adapter':>10}")
            print(f"{'accuracy (acc)':22}{t['base_acc'] / n:>8.3f}{t['lora_acc'] / n:>10.3f}")
            print(f"{'accuracy (acc_norm)':22}{t['base_norm'] / n:>8.3f}{t['lora_norm'] / n:>10.3f}")
            mb = sum(d["base_gold_logprob"] for d in details) / n
            ml = sum(d["adapter_gold_logprob"] for d in details) / n
            print(f"{'log-prob of answer':22}{mb:>8.2f}{ml:>10.2f}   (higher is better)")
            print(f"Adapter fixed {t['fixed']} answers and broke {t['broke']} (acc_norm).")
            results.update(hellaswag={k: v / n for k, v in t.items() if k not in ("fixed", "broke")},
                           fixed=t["fixed"], broke=t["broke"], details=details)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {a.out}")


if __name__ == "__main__":
    main()
