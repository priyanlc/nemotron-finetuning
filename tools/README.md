# Shared tools

## compare_vllm.py

`compare_vllm.py` compares a base model and a LoRA adapter served by one vLLM server, sending identical requests to both model names. It first prints a few greedy continuations side by side, then scores HellaSwag questions and counts the answers the adapter fixed or broke. It needs only Python's standard library and `datasets`, so it runs inside NVIDIA's AutoModel container, which already has both and shares the host network with the vLLM container. The [Lightning guide](../experiments/lightning-30b-lora-hellaswag-dgx-spark/README.md#step-11-run-the-full-comparison) shows it in use; its `results/README.md` records the run it produced.

```bash
cp ~/nemo-shared/nemotron-finetuning/tools/compare_vllm.py ~/nemo-shared/
docker exec -it lightning-lora python3 /shared/compare_vllm.py --n 20 --gen 2   # quick check
docker exec -it lightning-lora python3 /shared/compare_vllm.py --n 1000         # the full run, about 10 minutes
```

### How the scoring works

For each HellaSwag question the script sends the context followed by each of the four endings, as `ctx + " " + ending`, to vLLM's `/v1/completions` endpoint with `echo: true` and `logprobs: 0`, so the server returns the **log-probability of every prompt token**. It adds up the log-probabilities of the ending's tokens; the highest-scoring ending is the model's answer, as in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The scoring function:

```python
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
```

It reports two accuracy measures:

- **`acc`** selects the ending with the highest total log-probability.
- **`acc_norm`** normalises each total by the ending's character length to reduce the bias against longer options. This is the commonly reported HellaSwag metric in lm-eval.

Two things to know before comparing numbers with anything else. The space in `ctx + " " + ending` is not what AutoModel's HellaSwag dataset trains on (it joins context and ending with nothing in between), and the prompts carry no activity-label prefix, unlike lm-eval's HellaSwag task. The script is therefore for **base versus adapter under one procedure**, not for leaderboard scores.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--url` | `http://localhost:8000/v1` | The vLLM server. |
| `--base-name`, `--adapter-name` | `lightning-base`, `lightning-hellaswag` | The two served model names; they must match the server's `--served-model-name` and `--lora-modules`. |
| `--n` | `100` | HellaSwag validation questions to score; `0` skips scoring. |
| `--skip` | `64` | Validation rows to skip first, so questions used for validation during training are not scored. |
| `--gen`, `--max-tokens` | `3`, `60` | Side-by-side continuations to print, and their length. |
| `--prompts` | | A text file with one prompt per line for the side-by-side examples; otherwise HellaSwag contexts after the scored rows are used. |
| `--out` | `/shared/logs/compare_vllm.json` | Where the per-question results are written. |

`compare_puzzles.py` below scores the Kaggle reasoning puzzles by generation. The two scorers may be folded into one script behind a `--task` option later.

## compare_puzzles.py

`compare_puzzles.py` compares a base model and a LoRA adapter served by one vLLM server on the Kaggle reasoning puzzles, by generation. For each row of a JSONL file (`{"id", "category", "prompt", "answer"}`, as `experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark/data/make_puzzle_jsonl.py` writes `test.jsonl`) it renders the prompt with the model's chat template, asks `/v1/completions` for a greedy continuation from both model names, extracts the last `\boxed{}` and compares it with the expected answer the way the Kaggle metric does. It reports accuracy per category, the answers the adapter fixed and broke, an exact McNemar test and a paired 95% interval for the gain. It needs the standard library plus `transformers` for the tokenizer, both in NVIDIA's AutoModel container. The [reasoning-puzzles guide](../experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark/README.md#step-11-time-the-scorer-on-12-puzzles) shows it in use.

```bash
cp ~/nemo-shared/nemotron-finetuning/tools/compare_puzzles.py ~/nemo-shared/
docker exec -it lightning-lora python3 /shared/compare_puzzles.py --selftest     # checks the statistics; no server needed
docker exec -it lightning-lora python3 /shared/compare_puzzles.py --n 12 --out /shared/<experiment>/logs/compare-12.csv
docker exec -it lightning-lora python3 /shared/compare_puzzles.py --n 0 --out /shared/<experiment>/logs/compare-600.csv   # all rows; hours
```

### How the scoring works

**The prompt** is the user message rendered with `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`, which for Lightning ends in `<|im_start|>assistant\n<think>\n`. It goes to the completions endpoint, not the chat endpoint, so the text the server sees is exactly the prefix that training extended. The request asks for `min(7680, 8192 - prompt tokens)` tokens at temperature 0, the competition's budget and decoding.

**The answer** is the last non-empty `\boxed{...}` in the generation (nested braces handled for the final one). It matches the expected answer if both parse as numbers within 1% relative tolerance, or if the strings are equal ignoring case and surrounding whitespace. A missing box, or a generation cut off by the token limit, counts as wrong. The matching function:

```python
def answers_match(pred, gold):
    if pred is None:
        return False
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass
    return pred.strip().lower() == gold.strip().lower()
```

**The statistics** use only the puzzles the two models disagree on. With b answers fixed by the adapter, c broken, and n puzzles scored: the gain is (b − c) / n; its Wald standard error is sqrt(b + c − (b − c)² / n) / n, and the 95% interval is the gain ± 1.96 standard errors; the exact McNemar p-value is the two-sided binomial probability of a split at least as uneven as b against c among the b + c discordant pairs when fixes and breaks are equally likely. `--selftest` checks them on the HellaSwag run's 80 fixed and 53 broken of 1,000: gain +0.027, interval +0.004 to +0.050, p = 0.024.

**The subset** chosen by `--n` is every k-th row of the file, so that each category is represented when the file is grouped by category: `--n 12` scores 2 per category, `--n 300` 50. `--n 0` scores every row.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--url` | `http://localhost:8000/v1` | The vLLM server. |
| `--base-name`, `--adapter-name` | `lightning-base`, `lightning-puzzles` | The two served model names; they must match the server's `--served-model-name` and `--lora-modules`. |
| `--data` | `/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/data/test.jsonl` | The puzzles to score. |
| `--n` | `0` | Rows to score, spread over the file; `0` scores all of them. |
| `--max-tokens` | `7680` | The generation budget, capped per request at 8,192 minus the prompt's tokens. |
| `--concurrency` | `16` | Requests in flight at once; vLLM batches them. |
| `--out` | `/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/logs/compare_puzzles.csv` | The per-row CSV (prediction, correctness, token count and finish reason for both models). A `.json` summary and a `-generations.jsonl` file with every generation are written beside it. |
| `--selftest` | | Checks the statistics and the answer functions without a server. |

