#!/usr/bin/env python3
"""Build the STaR training set: the base model's own correct cipher traces in place of the scripted ones.

Three subcommands, run from anywhere (paths are relative to this file):

  select   write data/cipher-train.jsonl: the cipher rows of the original experiment's train.jsonl with the gold
           answer joined from its train.csv, for tools/sample_traces.py.
  build    read the sampler's output, keep the generations that finished with a correct first boxed answer, cut
           each at the start of the line that holds that box (or at the model's own </think>), rewrap it as
           <think>...</think>\\boxed{}, drop rows longer than --seq-length, hold out --val-holdout traces for val.jsonl, and mix the rest with the original
           train.jsonl minus its cipher rows (repeated so cipher is --share of the mix). Writes data/train.jsonl,
           data/val.jsonl and results/star-dataset.json.
  check    render every row with the chat template and confirm it extends the inference prefix, the completion
           layout is right, and no row exceeds --seq-length.

  python3 make_star_jsonl.py select
  python3 make_star_jsonl.py build --samples ../logs/star-cipher-7680.jsonl
  python3 make_star_jsonl.py check
  python3 make_star_jsonl.py --selftest         # the cut-and-rewrap rule and the mixing arithmetic, no tokenizer
"""
import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIG = HERE.parent.parent / "lightning-30b-lora-reasoning-puzzles-dgx-spark" / "data"
RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE.parent.parent.parent / "tools"))
from compare_puzzles import answers_match  # noqa: E402

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
SEQ_LENGTH = 8192  # the new dataset block's seq_length; rows longer than this are dropped, never truncated
BOX = re.compile(r"\\boxed\{([^}]*)\}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["select", "build", "check"])
    ap.add_argument("--samples", help="build: the sampler's output JSONL")
    ap.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    ap.add_argument("--share", type=float, default=1 / 6, help="build: cipher's share of the training mix")
    ap.add_argument("--val-holdout", type=int, default=10, help="build: kept traces held out for val.jsonl")
    ap.add_argument("--seed", type=int, default=0, help="build: seed for the val holdout")
    ap.add_argument("--shuffle-seed", type=int, default=2, help="build: seed for the final shuffle")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest and not a.command:
        ap.error("a command is required")
    if a.command == "build" and not a.samples:
        ap.error("build needs --samples")
    return a


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ----- the STaR rule ------------------------------------------------------------------------------------------------

def star_completion(gen, answer):
    """(completion, None) for a usable generation, else (None, reason).

    The generation is the text after the prompt's open <think>. Keep it if it finished and its first non-empty box
    matches the answer. Cut at the start of the line holding the box so the trace ends on a complete sentence, or at
    the model's own </think> if it closed the block before the box, then wrap in the layout the recipe trains on."""
    if gen.get("finish") == "length":
        return None, "truncated"
    text = gen["text"]
    first = next((m for m in BOX.finditer(text) if m.group(1).strip()), None)
    if first is None:
        return None, "no_box"
    box = first.group(1).strip()
    if not answers_match(box, answer):
        return None, "first_box_wrong"
    close = text.find("</think>", 0, first.start())
    if close != -1:  # the model closed its think block and boxed outside it: keep what it thought
        trace = text[:close].rstrip()
    else:
        trace = text[:text.rfind("\n", 0, first.start()) + 1].rstrip()
    if not trace:
        return None, "empty_trace"
    return f"<think>\n{trace}\n</think>\n\\boxed{{{box}}}", None


def repeats_for_share(n_cipher, n_other, share):
    """How many times to repeat the cipher rows so they are at least `share` of the mix."""
    target = share * n_other / (1 - share)
    return max(1, math.ceil(target / n_cipher)) if n_cipher else 0


# ----- subcommands --------------------------------------------------------------------------------------------------

def cmd_select():
    train = read_jsonl(ORIG / "train.jsonl")
    with open(ORIG / "train.csv", newline="") as f:
        answers = {r["id"]: r["answer"] for r in csv.DictReader(f)}
    cipher = [r for r in train if r["category"] == "cipher"]
    missing = [r["id"] for r in cipher if r["id"] not in answers]
    if missing:
        raise SystemExit(f"FAILED: {len(missing)} cipher ids have no answer in train.csv (e.g. {missing[:3]})")
    other = {r["id"] for name in ("test", "val") for r in read_jsonl(ORIG / f"{name}.jsonl")}
    overlap = [r["id"] for r in cipher if r["id"] in other]
    if overlap:
        raise SystemExit(f"FAILED: {len(overlap)} cipher training ids are also in test or val (e.g. {overlap[:3]})")
    out = [{"id": r["id"], "category": r["category"], "prompt": r["prompt"], "answer": answers[r["id"]]} for r in cipher]
    write_jsonl(HERE / "cipher-train.jsonl", out)
    print(f"read {len(train)} rows from {ORIG / 'train.jsonl'}, {len(cipher)} cipher; answers joined from train.csv on id")
    print(f"no overlap with test.jsonl or val.jsonl ({len(other)} ids)")
    print(f"wrote {HERE / 'cipher-train.jsonl'}: {len(out)} rows {{id, category, prompt, answer}}")


def cmd_build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    answers = {r["id"]: r["answer"] for r in read_jsonl(HERE / "cipher-train.jsonl")}
    prompts = {r["id"]: r["prompt"] for r in read_jsonl(HERE / "cipher-train.jsonl")}
    gens = read_jsonl(a.samples)
    seen = set()
    reasons, kept = Counter(), []
    for g in gens:
        if g["id"] in seen:  # a resumed run can repeat an id; the first generation wins
            reasons["duplicate"] += 1
            continue
        seen.add(g["id"])
        if g["id"] not in answers:
            raise SystemExit(f"FAILED: sampled id {g['id']} is not in cipher-train.jsonl")
        comp, why = star_completion(g, answers[g["id"]])
        if why:
            reasons[why] += 1
            continue
        kept.append({"id": g["id"], "category": "cipher", "prompt": prompts[g["id"]], "completion": comp})

    def ntok(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def total_tokens(r):
        pre = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=True)
        return ntok(pre) + ntok(r["completion"][len("<think>\n"):] + tok.eos_token + "\n")

    lengths = {r["id"]: total_tokens(r) for r in kept}
    too_long = [r for r in kept if lengths[r["id"]] > a.seq_length]
    reasons[f"over_{a.seq_length}_tokens"] = len(too_long)
    kept = [r for r in kept if lengths[r["id"]] <= a.seq_length]
    print(f"read {len(gens)} generations from {a.samples}")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {why:32s} {n:5d}   dropped")
    print(f"  {'kept':32s} {len(kept):5d}")
    if len(kept) <= a.val_holdout:
        raise SystemExit(f"FAILED: only {len(kept)} usable traces; nothing left after holding out {a.val_holdout}")

    rng = random.Random(a.seed)
    val_ids = set(rng.sample(sorted(r["id"] for r in kept), a.val_holdout))
    val_star = [r for r in kept if r["id"] in val_ids]
    train_star = [r for r in kept if r["id"] not in val_ids]

    orig_train = [r for r in read_jsonl(ORIG / "train.jsonl") if r["category"] != "cipher"]
    orig_val = [r for r in read_jsonl(ORIG / "val.jsonl") if r["category"] != "cipher"]
    reps = repeats_for_share(len(train_star), len(orig_train), a.share)
    train = orig_train + train_star * reps
    random.Random(a.shuffle_seed).shuffle(train)
    val = orig_val + val_star
    write_jsonl(HERE / "train.jsonl", train)
    write_jsonl(HERE / "val.jsonl", val)

    cats = Counter(r["category"] for r in train)
    star_lens = sorted(lengths[r["id"]] for r in kept)
    comp_lens = [ntok(r["completion"]) for r in kept]
    buckets = {"<=2048": len(orig_train)}
    for lo in range(2048, a.seq_length, 2048):
        buckets[f"{lo + 1}-{lo + 2048}"] = sum(lo < t <= lo + 2048 for t in star_lens) * reps
    summary = {"samples": a.samples, "generations": len(gens), "dropped": dict(reasons), "kept": len(kept),
               "val_holdout": len(val_star), "train_traces": len(train_star), "repeats": reps,
               "cipher_rows": len(train_star) * reps, "train_rows": len(train), "val_rows": len(val),
               "per_category": dict(sorted(cats.items())), "seq_length": a.seq_length, "share": a.share,
               "completion_tokens": {"mean": round(statistics.fmean(comp_lens)), "max": max(comp_lens)},
               "total_tokens": {"mean": round(statistics.fmean(star_lens)), "max": max(star_lens)},
               "rows_by_total_length": buckets, "seed": a.seed, "shuffle_seed": a.shuffle_seed}
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "star-dataset.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"held out {len(val_star)} kept traces for val (seed {a.seed}); {len(train_star)} for train")
    print(f"cipher share target {a.share:.3f} of the mix: {len(train_star)} x {reps} repeats = {len(train_star) * reps} cipher rows")
    print(f"train.jsonl: {len(train)} rows  {dict(sorted(cats.items()))}")
    print(f"val.jsonl:   {len(val)} rows  ({len(orig_val)} copied from the original, {len(val_star)} STaR cipher)")
    print(f"STaR completion tokens: mean {summary['completion_tokens']['mean']}, max {summary['completion_tokens']['max']}; "
          f"total with prompt: mean {summary['total_tokens']['mean']}, max {summary['total_tokens']['max']}")
    print("rows by total length: " + "   ".join(f"{k}: {v}" for k, v in buckets.items()))
    print(f"wrote {HERE / 'train.jsonl'}, {HERE / 'val.jsonl'}, {RESULTS / 'star-dataset.json'}")


def fail(msg):
    raise SystemExit(f"FAILED: {msg}")


def cmd_check(a):
    """The original experiment's check, plus the completion layout and the sequence-length ceiling."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.eos_token != "<|im_end|>":
        fail(f"unexpected end-of-turn token {tok.eos_token!r}")

    def prefix(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)

    def ntok(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    for name in ("train", "val"):
        rows = read_jsonl(HERE / f"{name}.jsonl")
        totals, star = [], 0
        for r in rows:
            pre = prefix(r["prompt"])
            if not pre.startswith("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n") or \
                    not pre.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n"):
                fail(f"{r['id']}: unexpected prefix {pre!r}")
            comp = r["completion"]
            if not comp.startswith("<think>\n") or not re.search(r"\n</think>\n\\boxed\{.+\}$", comp, re.S):
                fail(f"{r['id']}: completion layout is not <think>\\n...\\n</think>\\n\\boxed{{...}}: {comp[-80:]!r}")
            if comp.count("</think>") != 1:
                fail(f"{r['id']}: {comp.count('</think>')} closing think tags")
            text = tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": r["prompt"]},
                                            {"role": "assistant", "content": comp}], tokenize=False)
            tail = comp[len("<think>\n"):] + tok.eos_token + "\n"
            if text != pre + tail:
                fail(f"{r['id']}: the training text does not extend the inference prefix:\n{text!r}")
            totals.append(ntok(pre) + ntok(tail))
            star += r["category"] == "cipher"
        over = sum(t > a.seq_length for t in totals)
        print(f"{name}.jsonl: {len(rows)} rows OK ({star} cipher); every row renders as prefix + completion + "
              f"<|im_end|>; longest row {max(totals)} tokens, mean {statistics.fmean(totals):.0f}, "
              f"rows over {a.seq_length}: {over}")
        if over:
            fail(f"{name}.jsonl has {over} rows over --seq-length {a.seq_length}; rebuild with a lower ceiling")
    print("check OK")


# ----- selftest -----------------------------------------------------------------------------------------------------

def selftest():
    gold = "dragon reads under forest"
    text = ("We need to find the mapping.\nFourth word: rlnehm\n\nSo the decrypted text is: \"dragon reads under forest\"\n\n"
            "We need to put the final answer inside \\boxed{}.\n\nSo \\boxed{dragon reads under forest}.</think>"
            "\\boxed{dragon reads under forest}")
    comp, why = star_completion({"finish": "stop", "text": text}, gold)
    assert why is None, why
    assert comp == ("<think>\nWe need to find the mapping.\nFourth word: rlnehm\n\nSo the decrypted text is: \"dragon reads "
                    "under forest\"\n\nWe need to put the final answer inside \\boxed{}.\n</think>\n\\boxed{dragon reads under forest}"), comp
    assert star_completion({"finish": "length", "text": text}, gold) == (None, "truncated")
    assert star_completion({"finish": "stop", "text": "thinking... no answer"}, gold) == (None, "no_box")
    assert star_completion({"finish": "stop", "text": "so \\boxed{dragon reads under the forest}"}, gold) == (None, "first_box_wrong")
    comp, why = star_completion({"finish": "stop", "text": "done\n</think>\nThe answer: \\boxed{dragon reads under forest}"}, gold)
    assert comp == "<think>\ndone\n</think>\n\\boxed{dragon reads under forest}", comp
    assert star_completion({"finish": "stop", "text": "\\boxed{dragon reads under forest}"}, gold) == (None, "empty_trace")
    # the model's own box text is kept (case differs from gold but matches under the metric)
    comp, why = star_completion({"finish": "stop", "text": "map found\nSo \\boxed{Dragon Reads Under Forest}"}, gold)
    assert comp == "<think>\nmap found\n</think>\n\\boxed{Dragon Reads Under Forest}", comp
    # a numeric first box inside tolerance, with a wrong later box, is still kept
    comp, why = star_completion({"finish": "stop", "text": "g = 6.63\n\\boxed{26.2}\nhmm \\boxed{30}"}, "26.18")
    assert comp == "<think>\ng = 6.63\n</think>\n\\boxed{26.2}", comp
    assert repeats_for_share(385, 7374, 1 / 6) == 4      # 385 x 4 = 1540 >= 7374 / 5
    assert repeats_for_share(1500, 7374, 1 / 6) == 1     # already a sixth; no repeats
    assert repeats_for_share(0, 7374, 1 / 6) == 0
    print("selftest OK")


def main():
    a = parse_args()
    if a.selftest:
        return selftest()
    if a.command == "select":
        cmd_select()
    elif a.command == "build":
        cmd_build(a)
    else:
        cmd_check(a)


if __name__ == "__main__":
    main()
