#!/usr/bin/env python3
"""Build the train, validation and test JSONL files for AutoModel's column-mapped dataset from the Kaggle puzzle data.

Input:  train.csv from the NVIDIA Nemotron Model Reasoning Challenge (columns id, prompt, answer; 9,500 rows).
Output, next to this script by default:
  train.jsonl, val.jsonl   {"id", "category", "prompt", "completion"}: prompt is the user message, completion the
                           assistant message, a <think> reasoning trace followed by the \\boxed{} answer
  test.jsonl               {"id", "category", "prompt", "answer"}: held-out puzzles for tools/compare_puzzles.py

Splits: 10 puzzles per category go to val.jsonl (seed 0), 100 per category to test.jsonl (seed 1) and the rest to
train.jsonl, shuffled (seed 2). No id appears in two files.

The prompt is the puzzle plus the boxed-answer instruction (a cipher puzzle also gets the vocabulary hint). The
completion's trace is computed from the puzzle's own examples and answer where a rule can be read off them (cipher,
gravity, some bit_ops and symbol puzzles) and is a one-sentence template otherwise; the README's Data section
states the rules.

    python make_puzzle_jsonl.py            # needs polars
    python make_puzzle_jsonl.py --check    # also renders every row with the Lightning chat template (transformers,
                                           # jinja2, a Hugging Face login) and checks it against the inference prefix
"""
import argparse
import json
import pathlib
import re
import statistics

import polars as pl

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
SEQ_LENGTH = 2048  # the dataset block's seq_length: AutoModel would truncate longer rows

# ----- puzzle classification, the boxed-answer instruction and the seeded sampling ----------------------------------

METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def classify_type(prompt_text):
    """Classify a puzzle prompt into one of 6 categories."""
    p = prompt_text.lower()
    if 'bit manipulation' in p or '8-bit binary' in p: return 'bit_ops'
    elif 'encrypt' in p or 'decrypt' in p: return 'cipher'
    elif 'gravitational' in p or 'falling distance' in p: return 'gravity'
    elif 'numeral system' in p: return 'numeral'
    elif 'transformation rules' in p: return 'symbol'
    elif 'unit conversion' in p or 'convert the following measurement' in p: return 'unit_conv'
    return 'unknown'


def stratified_sample(df, n_per_type, seed):
    """Sample n_per_type from each category, categories in order of first appearance so the output is reproducible."""
    dfs = []
    for qtype in df['qtype'].unique(maintain_order=True).to_list():
        subset = df.filter(pl.col('qtype') == qtype)
        n = min(n_per_type, len(subset))
        dfs.append(subset.sample(n=n, seed=seed))
    return pl.concat(dfs)


# ----- prompt and reasoning-trace builders --------------------------------------------------------------------------

COT_BY_TYPE = {
    'bit_ops': "Let me analyze each input-output pair to identify the bit transformation rule. I'll compare the binary strings to find the pattern, then apply it to the new input.",
    'gravity': "I need to determine the gravitational constant g from the given time-distance pairs using d = 0.5*g*t^2. I'll solve for g using a data point, then compute the distance for the new time.",
    'unit_conv': "I need to find the conversion factor from the given examples. I'll divide the output by the input to get the ratio, then apply it to the new measurement.",
    'cipher': "I need to build the substitution mapping from the ciphertext-plaintext pairs. I'll map each character, then use the mapping to decrypt the new ciphertext.",
    'numeral': "I need to convert the given number to the appropriate numeral system by breaking it down into components.",
    'symbol': "I need to identify the transformation rules from the examples and apply them to the new equation.",
}
COT_DEFAULT = "Let me analyze the pattern in the given examples and apply it to solve the problem."


def build_cipher_vocab(df):
    """The vocabulary of the cipher answers (77 words in train.csv): lowercase, punctuation stripped, 2+ letters, sorted."""
    words = set()
    for ans in df.filter(pl.col('qtype') == 'cipher')['answer'].to_list():
        for w in str(ans).lower().split():
            w = w.strip('.,;:!?\'"()[]{}')
            if w.isalpha() and len(w) >= 2:
                words.add(w)
    return sorted(words)


def augment_cipher_prompt(prompt, vocab):
    """Append the vocabulary hint to a cipher prompt. Other prompts are returned unchanged."""
    if classify_type(prompt) != 'cipher':
        return prompt
    hint = (f'\n\n(The decoded plaintext uses words from this fixed vocabulary, '
            f'encrypted via the same substitution as the examples above.)\n'
            f'Vocabulary: {", ".join(vocab)}')
    return prompt + hint


def find_bit_function(out_bit_idx, examples):
    """The simplest boolean function of the 8 input bits that gives output bit out_bit_idx in every example, or None."""
    targets = [int(out[out_bit_idx]) for _, out in examples]
    if all(t == 0 for t in targets):
        return ('= 0', lambda inp: 0)
    if all(t == 1 for t in targets):
        return ('= 1', lambda inp: 1)
    for k in range(8):
        if all(t == int(inp[k]) for t, (inp, _) in zip(targets, examples)):
            return (f'= input[{k}]', lambda inp, k=k: int(inp[k]))
        if all(t == 1 - int(inp[k]) for t, (inp, _) in zip(targets, examples)):
            return (f'= NOT input[{k}]', lambda inp, k=k: 1 - int(inp[k]))
    for i in range(8):
        for j in range(i + 1, 8):
            if all(t == int(inp[i]) ^ int(inp[j]) for t, (inp, _) in zip(targets, examples)):
                return (f'= input[{i}] XOR input[{j}]', lambda inp, i=i, j=j: int(inp[i]) ^ int(inp[j]))
            if all(t == int(inp[i]) & int(inp[j]) for t, (inp, _) in zip(targets, examples)):
                return (f'= input[{i}] AND input[{j}]', lambda inp, i=i, j=j: int(inp[i]) & int(inp[j]))
            if all(t == int(inp[i]) | int(inp[j]) for t, (inp, _) in zip(targets, examples)):
                return (f'= input[{i}] OR input[{j}]', lambda inp, i=i, j=j: int(inp[i]) | int(inp[j]))
    return None


def build_bit_ops_dynamic_cot(prompt, answer):
    """Per-bit function walk for bit_ops puzzles; None when no simple function fits or the result is not the answer."""
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    if len(pairs) < 2:
        return None
    all_example_inputs = {p[0] for p in pairs}
    new_input = None
    for cand in re.findall(r'\b([01]{8})\b', prompt):
        if cand not in all_example_inputs:
            new_input = cand
            break
    if new_input is None:
        return None
    descriptions, evaluators = [], []
    for k in range(8):
        result = find_bit_function(k, pairs)
        if result is None:
            return None
        desc, evaluator = result
        descriptions.append(f'  bit[{k}] {desc}')
        evaluators.append(evaluator)
    predicted = ''.join(str(ev(new_input)) for ev in evaluators)
    if predicted != str(answer).strip():
        return None
    return (f"Examining each output bit position against the example pairs:\n"
            + '\n'.join(descriptions)
            + f"\nApplying to input {new_input}, computing each output bit:\n"
            + f"output = {predicted}. Answer: {answer}")


def build_dynamic_cot(qtype, prompt, answer):
    """A reasoning trace computed from the puzzle's own examples and answer, or None for the static template."""
    if qtype == 'cipher':
        pairs = re.findall(r'^([^\n]+?)\s*->\s*([^\n]+?)\s*$', prompt.lower(), re.MULTILINE)
        mapping = {}
        for ct, pt in pairs:
            ct, pt = ct.strip(), pt.strip()
            if len(ct) == len(pt):
                for c, p in zip(ct, pt):
                    if c.isalpha() and p.isalpha():
                        mapping[c] = p
        mt = re.search(r'decrypt the following text:\s*(.+)', prompt, re.IGNORECASE)
        target = mt.group(1).strip() if mt else None
        if mapping and target:
            if len(target) == len(answer):
                for c, p in zip(target.lower(), answer.lower()):
                    if c.isalpha() and p.isalpha() and c not in mapping:
                        mapping[c] = p
            decoded = ''.join(mapping.get(ch, ch) if ch.isalpha() else ch for ch in target.lower())
            first = target.split()[0] if target.split() else target
            steps = ', '.join(f'{ch}->{mapping.get(ch, "?")}' for ch in first.lower())
            return (f"I'll derive the substitution map from the examples and decrypt "
                    f"char by char. '{first}': {steps} = "
                    f"'{''.join(mapping.get(c, c) for c in first.lower())}'. "
                    f"Full decryption of '{target}': {decoded}. Answer: {answer}")
    elif qtype == 'gravity':
        m = re.search(r't\s*=\s*([\d.]+)s?\s*,?\s*distance\s*=\s*([\d.]+)\s*m', prompt, re.IGNORECASE)
        if m:
            t, d = float(m.group(1)), float(m.group(2))
            g = 2 * d / (t * t) if t > 0 else 0
            return (f"Using d = 0.5*g*t^2: g = 2d/t^2 = 2*{d}/{t}^2 = {g:.4f}. "
                    f"Apply to find d for the new t. Answer: {answer}")
    elif qtype == 'unit_conv':
        m = re.search(r'([\d.]+)\s*[a-zA-Z]+\s*->\s*([\d.]+)', prompt)
        if m:
            inp, out = float(m.group(1)), float(m.group(2))
            factor = out / inp if inp > 0 else 1
            return (f"Conversion factor = output/input = {out}/{inp} = {factor:.4f}. "
                    f"Apply factor to new input. Answer: {answer}")
    elif qtype == 'bit_ops':
        return build_bit_ops_dynamic_cot(prompt, answer)
    elif qtype == 'symbol':
        mt = re.search(r'result for:\s*(.+)', prompt, re.IGNORECASE)
        if mt:
            m = re.fullmatch(r'\s*(-?\d+)\s*(\D)\s*(\d+)\s*', mt.group(1).strip())
            ans_clean = answer.strip().strip('\'"').strip()
            if m:
                try:
                    a, sym, b = int(m.group(1)), m.group(2), int(m.group(3))
                    gold = int(ans_clean)
                    cands = {
                        'addition (a+b)': a + b,
                        'subtraction (a-b)': a - b,
                        'reverse subtraction (b-a)': b - a,
                        'multiplication (a*b)': a * b,
                        'integer division (a//b)': (a // b if b else None),
                        'concatenation (digits of a then b)': int(f'{a}{b}'),
                        'sum of digits': sum(int(d) for d in f'{a}{b}'),
                    }
                    match = [name for name, v in cands.items() if v == gold]
                    if match:
                        return (f"The symbol '{sym}' encodes an operation on {a} and {b}. "
                                f"Matching the examples, '{sym}' is {match[0]}. "
                                f"Computing: {a} {sym} {b} = {gold}. Answer: {gold}")
                except ValueError:
                    pass
    return None


# ----- this experiment ----------------------------------------------------------------------------------------------

def build_example(row, vocab):
    """One puzzle -> (user message, assistant message)."""
    qtype, answer = row["qtype"], row["answer"]
    user = augment_cipher_prompt(row["prompt"], vocab) + METRIC_SUFFIX
    cot = build_dynamic_cot(qtype, row["prompt"], answer) or COT_BY_TYPE.get(qtype, COT_DEFAULT)
    return user, f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"


def fail(msg):
    raise SystemExit(f"FAILED: {msg}")


def check(out, model):
    """Render every row as AutoModel's chat-template path does and confirm it extends the inference prefix."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    if tok.eos_token != "<|im_end|>":
        fail(f"unexpected end-of-turn token {tok.eos_token!r}")

    def prefix(prompt):  # what the scorer and the Kaggle metric send: the user turn plus an open <think>
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)

    def ntok(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    for name in ("train", "val", "test"):
        rows = [json.loads(line) for line in open(out / f"{name}.jsonl")]
        p_lens, c_lens = [], []
        for r in rows:
            pre = prefix(r["prompt"])
            if not pre.startswith("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n") or \
                    not pre.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n"):
                fail(f"{r['id']}: unexpected prefix {pre!r}")
            p_lens.append(ntok(pre))
            if "completion" in r:
                # AutoModel renders [system (empty), user, assistant] with the tokenizer's chat template
                text = tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": r["prompt"]},
                                                {"role": "assistant", "content": r["completion"]}], tokenize=False)
                tail = r["completion"][len("<think>\n"):] + tok.eos_token + "\n"
                if not r["completion"].startswith("<think>\n") or text != pre + tail:
                    fail(f"{r['id']}: the training text does not extend the inference prefix:\n{text!r}")
                c_lens.append(ntok(tail))
        line = f"{name}.jsonl: {len(rows)} rows render as prefix + completion; prompt tokens max {max(p_lens)}, mean {statistics.fmean(p_lens):.0f}"
        if c_lens:
            totals = [p + c for p, c in zip(p_lens, c_lens)]
            line += (f"; completion tokens max {max(c_lens)}, mean {statistics.fmean(c_lens):.0f}; "
                     f"total max {max(totals)}, mean {statistics.fmean(totals):.0f}; rows over {SEQ_LENGTH}: {sum(t > SEQ_LENGTH for t in totals)}")
        print(line)
    print("check OK")


def main():
    here = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Build train/val/test JSONL files from the Kaggle puzzle train.csv.")
    ap.add_argument("--train-csv", default=here / "train.csv")
    ap.add_argument("--out", default=here, help="folder for train.jsonl, val.jsonl and test.jsonl")
    ap.add_argument("--val-per-type", type=int, default=10)
    ap.add_argument("--test-per-type", type=int, default=100)
    ap.add_argument("--check", action="store_true", help="render every row with the chat template and check it")
    ap.add_argument("--model", default=MODEL, help="tokenizer for --check")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(a.train_csv, infer_schema=False).with_columns(  # every column as text: answers keep leading zeros
        pl.col("prompt").map_elements(classify_type, return_dtype=pl.Utf8).alias("qtype"))
    unknown = df.filter(pl.col("qtype") == "unknown").height
    if unknown:
        fail(f"{unknown} puzzles could not be classified")
    vocab = build_cipher_vocab(df)

    val = stratified_sample(df, a.val_per_type, seed=0)
    rest = df.filter(~pl.col("id").is_in(val["id"].to_list()))
    test = stratified_sample(rest, a.test_per_type, seed=1)
    train = rest.filter(~pl.col("id").is_in(test["id"].to_list())).sample(fraction=1.0, shuffle=True, seed=2)
    ids = [set(part["id"].to_list()) for part in (train, val, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2] or sum(map(len, ids)) != df.height:
        fail("the splits overlap or do not cover train.csv")

    for name, part in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w") as f:
            for row in part.iter_rows(named=True):
                user, completion = build_example(row, vocab)
                f.write(json.dumps({"id": row["id"], "category": row["qtype"], "prompt": user, "completion": completion}) + "\n")
    with open(out / "test.jsonl", "w") as f:
        for row in test.iter_rows(named=True):
            user, _ = build_example(row, vocab)
            f.write(json.dumps({"id": row["id"], "category": row["qtype"], "prompt": user, "answer": row["answer"]}) + "\n")

    for name, part in (("train", train), ("val", val), ("test", test)):
        counts = part.group_by("qtype").len().sort("qtype")
        print(f"{name}.jsonl: {part.height} rows", dict(zip(counts["qtype"].to_list(), counts["len"].to_list())))
    print(f"cipher vocabulary hint: {len(vocab)} words")
    if a.check:
        check(out, a.model)


if __name__ == "__main__":
    main()
