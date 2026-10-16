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

HellaSwag is the only scorer today. Other tasks will add their own, selected by a `--task` option, when an experiment needs them.
