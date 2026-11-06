# Nemotron 3.5 Lightning 30B-A3B: STaR traces for the cipher puzzles, one DGX Spark

*Part of [nemotron-finetuning](../../README.md). A follow-up to the [reasoning-puzzles experiment](../lightning-30b-lora-reasoning-puzzles-dgx-spark/README.md), which it leaves untouched; it reuses that experiment's data files, containers and served base model.*

**Replace the scripted cipher training traces with the base model's own correct derivations, retrain the LoRA with everything else unchanged, and measure whether the cipher regression goes away without giving back the other categories.** The published adapter scores 0.550 against the base's 0.290 on 300 held-out puzzles, but regressed on cipher from 0.36 to 0.04, and the [higher-cap rescore](../lightning-30b-lora-reasoning-puzzles-dgx-spark/higher-cap-rescore.md) showed the regression widens when the base has room (base 0.54, adapter 0.04 at 16,384 tokens). Every one of the 1,466 scripted cipher traces completes the letter map from the known answer and states the decryption in one sentence, so the adapter learned the shape of a decryption without the work. The STaR recipe (Self-Taught Reasoner, [Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)) builds traces the other way round: the model solves the training puzzles itself, only the final answers are checked, and the correct solutions become the training data. The traces are real work, in the model's own distribution, and never saw the answer. A reader's comment on the write-up suggested this.

> **Setup:** one DGX Spark (GB10, 128 GB) · `nvcr.io/nvidia/nemo-automodel:26.08` to train, `vllm/vllm-openai:v0.27.1` to serve · `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` · NVIDIA's single-GPU LoRA recipe with the dataset block swapped and `seq_length` raised to 8,192 · the original experiment's `data/train.jsonl`, `train.csv`, `val.jsonl` and `test.jsonl` already on the Spark.

## Contents

| Step | Task | Run from | Time |
|---|---|---|---|
| 1 | [Serve the base model](#step-1-serve-the-base-model) | `vllm-lightning` | Minutes |
| 2 | [Select the cipher puzzles and smoke-test the sampler](#step-2-select-the-cipher-puzzles-and-smoke-test-the-sampler) | Spark host, executing inside `lightning-lora` | 17 minutes |
| 3 | [Sample the base on all 1,466 cipher training puzzles](#step-3-sample-the-base-on-all-1466-cipher-training-puzzles) | Same, detached | 16 h 40 min |
| 4 | [Build and check the STaR training set](#step-4-build-and-check-the-star-training-set) | Same | Minutes |
| 5 | [Write the recipe config](#step-5-write-the-recipe-config) | Spark host | Minutes |
| 6 | [Decode check and 20-step smoke run](#step-6-decode-check-and-20-step-smoke-run) | `lightning-lora`, GPU freed | 13 minutes |
| 7 | [Train 200 steps](#step-7-train-200-steps) | `lightning-lora` | 56 minutes |
| 8 | [Serve the base and both adapters](#step-8-serve-the-base-and-both-adapters) | `vllm-lightning` | 10 minutes |
| 9 | [Score the same 300 puzzles](#step-9-score-the-same-300-puzzles) | Spark host, executing inside `lightning-lora` | 59 minutes, then 3 h 55 min |
| 10 | [Write up](#step-10-write-up) | Anywhere | 2 hours |

Then [Results](#results) and [Files](#files). All times are measured, 2026-11-05/06.

Steps 4 to 9 can run unattended: [`star_chain.sh`](star_chain.sh) waits for the sampler's summary file, then runs them in order with a gate after each (kept traces, `check`, the config's two `seq_length` lines, the decode check, the smoke run's peak memory under `MEM_LIMIT` (85 GiB by default), a `LOWEST_VAL` adapter, the server listing all three model names, both `.json` summaries) and appends one line per stage to `logs/chain.status`. Start it on the Spark host with `nohup setsid bash $S/star_chain.sh > $S/logs/chain.log 2>&1 &`; the first failed gate stops it with a `FAILED` line naming the log to read. On this run the smoke gate tripped at 88.4 GiB, under the box's real limit but over the default; the chain was restarted with `SKIP_TO=train MEM_LIMIT=90` and ran through to `DONE`. The full status log is in [`results/chain.status`](results/chain.status).

## What the data looks like

The sampler's input is the cipher rows of the original `train.jsonl` with the gold answer joined from `train.csv`: the prompt already carries the boxed-answer instruction and the 77-word vocabulary hint, so the base model sees exactly the text the adapter was trained on and is scored on. The test and val puzzles are never sampled.

**The rule** (`star_completion` in [`data/make_star_jsonl.py`](data/make_star_jsonl.py)): keep a generation if it finished under the cap and its first non-empty `\boxed{}` matches the answer under the Kaggle metric (exact string or 1% numeric tolerance). Cut the text at the start of the line that holds the box, so the trace ends on a complete sentence, or at the model's own `</think>` when it closed the block before boxing, and wrap it as `<think>\n{trace}\n</think>\n\boxed{answer}`, the layout the original experiment trains on. The box keeps the model's own wording. Rows whose prompt plus completion exceed 8,192 tokens are dropped, never truncated, because a cut trace loses its box.

Three facts from the earlier runs shaped this. At the 7,680-token cap the base finishes 16 of 50 test cipher puzzles and 14 of those have a correct first box, so about 28% of the training puzzles should yield a trace. The first box sits at the very end of the trace, so cutting at it removes under 1% of the text; the kept traces are 4,000 to 7,500 tokens where the scripted completions were about 50, which is why `seq_length` goes to 8,192. And the base writes its box inside the think block in 22 of 23 cases, hence the rewrap.

**The mix.** The new `train.jsonl` is the original minus its 1,466 cipher rows (7,374 rows of the other five categories, scripted traces unchanged) plus the kept STaR traces, repeated so that cipher is about a sixth of the mix, shuffled with seed 2. Ten kept traces are held out, with seed 0, to replace the ten scripted cipher rows of `val.jsonl`; without that, `LOWEST_VAL` would be chosen on the loss of traces the adapter is no longer meant to produce. The row schema is the original's, `{id, category, prompt, completion}`.

**What is not done.** STaR's rationalization step, which shows the answer as a hint to the puzzles the model got wrong and keeps those traces too, is skipped: it would bring answer-aware traces back in. No second STaR iteration, no GRPO, and the other categories' scripted traces are left as they are.

## Step 1: Serve the base model

The original experiment's [step 10](../lightning-30b-lora-reasoning-puzzles-dgx-spark/README.md#step-10-free-the-gpu-and-serve-base-and-adapter) command serves `lightning-base`; the adapter module is not needed for sampling but does no harm. A server left running from the higher-cap follow-up, with `--max-model-len 32768`, works as is: the sampler reads the window from the server and caps each request at 7,680 new tokens regardless.

## Step 2: Select the cipher puzzles and smoke-test the sampler

```bash
S=/shared/nemotron-finetuning/experiments/lightning-30b-lora-puzzles-star-dgx-spark
T=/shared/nemotron-finetuning/tools
docker exec -it lightning-lora python3 $T/sample_traces.py --selftest
docker exec -it lightning-lora python3 $S/data/make_star_jsonl.py --selftest
docker exec -it lightning-lora python3 $S/data/make_star_jsonl.py select        # data/cipher-train.jsonl, 1466 rows
docker exec -it lightning-lora python3 $T/sample_traces.py --data $S/data/cipher-train.jsonl --n 12 --concurrency 12 \
  --out $S/logs/star-smoke.jsonl
```

The smoke run prints the cap (7,680), the server window and a progress line with running counts of finished, first-box-correct and truncated generations. Expect about a third kept, lines appearing in `star-smoke.jsonl` while the run is going (the file is appended per generation), and a second invocation with `--resume` reporting `12 already in` and sampling nothing. Measured 2026-11-05: 12 puzzles in 1,010 s, 2 finished, both with a correct first box, 10 truncated.

## Step 3: Sample the base on all 1,466 cipher training puzzles

```bash
docker exec -d lightning-lora bash -c "python3 $T/sample_traces.py --data $S/data/cipher-train.jsonl \
  --out $S/logs/star-cipher-7680.jsonl --concurrency 32 --resume > $S/logs/star-cipher-7680.log 2>&1"
tail -n 3 $S/logs/star-cipher-7680.log; wc -l $S/logs/star-cipher-7680.jsonl       # from the Spark host, any time
```

Detached, so a lost connection does not stop it; `--resume` makes the same command safe to run again after a container restart. Estimate: 1,466 puzzles at about 7,100 tokens each (two thirds reach the cap) at the server's ~175 tokens per second aggregate is about 17 hours. The `.json` summary beside the output has the per-category counts when it finishes. Measured 2026-11-05/06: the remaining 1,454 puzzles in 59,995 s (16 h 40 min), 465 finished, 397 with a correct first box, 989 truncated; with the smoke run, 1,466 sampled.

## Step 4: Build and check the STaR training set

```bash
docker exec -it lightning-lora python3 $S/data/make_star_jsonl.py build --samples $S/logs/star-cipher-7680.jsonl
docker exec -it lightning-lora python3 $S/data/make_star_jsonl.py check
```

`build` prints what it dropped and why (truncated, no box, first box wrong, over 8,192 tokens), the holdout, the repeat factor, the per-category counts of the new `train.jsonl`, and the length histogram; the same goes to `results/star-dataset.json`, which is committed. `check` fails on any completion that does not start with `<think>\n` and end with `</think>\n\boxed{...}`, any row that does not render to the inference prefix plus the completion, and any row over 8,192 tokens.

## Step 5: Write the recipe config

The same three commands as the original experiment's step 7, with this folder's block:

```bash
E=/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark
cp $E/configs/nemotron_nano_v3_5_lightning_singlegpu_lora.yaml $S/configs/star_lora.yaml
sed -i '/^dataset:$/,/^  split: train$/d' $S/configs/star_lora.yaml
grep -c hellaswag $S/configs/star_lora.yaml          # expect 0
cat $S/configs/star-dataset-block.yaml >> $S/configs/star_lora.yaml
```

Everything in the recipe other than the dataset paths and `seq_length` stays as the published adapter had it (LoRA r8/α32, learning rate 1e-4 cosine, global batch 8, 200 steps), so the comparison is only about the data.

## Step 6: Decode check and 20-step smoke run

Stop the vLLM server first (`Ctrl-C` in its terminal); the GPU is shared. Then the original experiment's [step 8](../lightning-30b-lora-reasoning-puzzles-dgx-spark/README.md#step-8-decode-check-and-20-step-smoke-run) decode check with this folder's `val.jsonl` and `seq_length=8192`, and its 20-step run with `$S/configs/star_lora.yaml` and a checkpoint folder under `$S/checkpoints/`. Watch `mem` in the step records: the original ran at about 62 GiB on rows of at most 492 tokens, and the first batches that contain a 7,000-token cipher row show the new peak. If it approaches 90 GiB, rebuild with `--seq-length 6144` and run again. Measured 2026-11-06: decode check `60 rows OK`, longest row 7,572 tokens; smoke run 20 steps at 15.2 s/step, loss 0.403 → 0.430, peak 88.4 GiB. Steps whose batch holds an 8k row sit at 80 to 88 GiB, steps of short rows only at 62, so the peak is set by the longest single row, not the batch.

## Step 7: Train 200 steps

The original's [step 9](../lightning-30b-lora-reasoning-puzzles-dgx-spark/README.md#step-9-train-200-steps-with-validation) command with `$S/configs/star_lora.yaml`, `--checkpoint.checkpoint_dir $S/checkpoints/$RUN`, validation and a checkpoint every 20 steps. Then `ln -sfn $RUN $S/checkpoints/star-200` and note which step `LOWEST_VAL` points at. Record the validation losses in `results/validation-loss.csv`. Measured 2026-11-06, run `star-200-1106-0809`: 200 steps in 56 min at 15.1 s/step, loss 0.404 → 0.080, peak 88.5 GiB; validation loss 0.109 at step 19 → 0.057 at step 179 (0.058 at 199), so `LOWEST_VAL` is step 179. The STaR rows have very low loss from the first step (0.14 to 0.17 on steps that hold one, against 1.4 to 3.5 on steps of scripted rows only): they are the base's own greedy output, which it already predicts well, so the run teaches the adapter to keep that behaviour rather than something new.

## Step 8: Serve the base and both adapters

The original's step 10 command with two LoRA modules, so the published adapter and the new one are scored in the same server session:

```bash
--lora-modules lightning-puzzles=$E/checkpoints/puzzles-200/LOWEST_VAL/model lightning-star=$S/checkpoints/star-200/LOWEST_VAL/model
```

`--max-model-len 8192` is enough for the competition cap; `32768` allows a 16,384-cap run afterwards. Measured: the server listed all three names 10 minutes after `docker run`, at 32,768.

## Step 9: Score the same 300 puzzles

Both runs score the 300 puzzles of the original step 12 (`--n 300` selects the same rows) at the competition cap:

```bash
docker exec -it lightning-lora python3 $T/compare_puzzles.py --n 300 --concurrency 32 \
  --base-name lightning-puzzles --adapter-name lightning-star --out $S/logs/compare-300-v1-vs-star.csv
docker exec -it lightning-lora python3 $T/compare_puzzles.py --n 300 --concurrency 32 \
  --base-name lightning-base --adapter-name lightning-star --out $S/logs/compare-300-star.csv
```

The first pairs the published adapter against the new one, puzzle by puzzle, and is quick because only the cipher traces are long; its `lightning-puzzles` column should reproduce the published result within run-to-run churn (the higher-cap rescore measured the adapter's churn at 3 gained and 2 lost over 300 puzzles). The second is the headline, paired against the base, about 3.5 hours because the base's generations are long. Optionally repeat the second at `--max-tokens 16384` and merge with `cap_sweep.py`. Measured 2026-11-06: 59 min and 3 h 55 min.

## Step 10: Write up

Commit the two `.json` summaries and `results/star-dataset.json`, fill in [Results](#results), and add the new row to the top-level README. The sentence the write-up needs is whether cipher is back at the base's level under the competition cap (about 0.36) and what the other five categories did.

## Results

Both scoring runs finished 2026-11-06. **Cipher is back, and above the base's level under the cap:** the STaR adapter finishes 22 of the 50 test cipher puzzles and every one it finishes is correct, where the published adapter had 2 and the base had 18 in October and 13 in this run. The other five categories did not move beyond run-to-run churn.

| category | base, October | base, this run | published adapter | STaR adapter |
|---|---|---|---|---|
| cipher | 0.36 | 0.26 | 0.04 | **0.44** |
| bit_ops | 0.02 | 0.04 | 0.24 (0.26) | 0.30 |
| gravity | 0.22 | 0.30 | 0.92 (0.94) | 0.92 |
| numeral | 0.98 | 1.00 | 1.00 | 1.00 |
| symbol | 0.10 | 0.06 | 0.10 | 0.08 / 0.10 |
| unit_conv | 0.06 | 0.12 | 1.00 | 1.00 |
| **all 300** | **0.290** | **0.297** | **0.550 (0.557)** | **0.623 / 0.627** |

All at the 7,680 cap on the same 300 puzzles. The published-adapter column is the October value with this session's reproduction in brackets (`compare-300-v1-vs-star.json`). The STaR column is from the two paired runs; where they differ it shows both, one symbol puzzle flipping between runs. The base's cipher score moved from 18 to 13 of 50 between October and this run with nothing changed but the server session: its cipher traces average 7,068 tokens against a 7,680 cap, so a few tokens decide whether it finishes, and the greedy drift the [higher-cap rescore](../lightning-30b-lora-reasoning-puzzles-dgx-spark/higher-cap-rescore.md) describes lands on that edge.

The two paired tests (`results/compare-300-v1-vs-star.json`, `results/compare-300-star.json`):

| pairing | accuracies | gain, 95% CI, McNemar p | fixed / broke | truncated | time |
|---|---|---|---|---|---|
| published adapter → STaR adapter | 0.557 → 0.623 | +0.067, +0.027 to +0.106, p = 0.0017 | 29 / 9 | 0 / 28 | 59 min |
| base → STaR adapter | 0.297 → 0.627 | +0.330, +0.272 to +0.388, p = 1.8e-24 | 105 / 6 | 216 / 28 | 3 h 55 min |

The published adapter's own paired test against the base was +0.260 (CI +0.197 to +0.323).

**Why it beats the base at cipher.** Not by reasoning better: whenever either model finishes a cipher puzzle it is right (base 13 of 13, STaR adapter 22 of 22). The STaR adapter's cipher traces average 6,745 tokens where the base's average 7,068, and under a 7,680 cap that is enough to finish nine more puzzles. The rescore at 16,384 showed the base reaches 0.54 with room, so the honest claim is "finishes more often under the competition cap", not "better at cipher". The 28 truncated cipher attempts are the cost: this is the first of the adapters that runs past the cap at all, and the 48 of 50 the published adapter got wrong in 86 tokens are now 6,700-token attempts of which 22 land.

**What it cost elsewhere.** Nothing measurable. Against the published adapter in the same session, bit_ops 13 → 15, gravity 47 → 46, numeral 50 → 50, symbol 5 → 4, unit_conv 50 → 50, and the mean output length per category is identical to the token on four of the five. The 7,374 non-cipher rows were unchanged, and the 1,524 cipher rows did not pull the short-answer format off the other categories.

**On the recipe.** The STaR rows cost almost nothing to fit (loss 0.14 at step 0) because they are the base's own greedy text; the run's work was keeping that text intact while the scripted rows taught the other five formats. 381 traces at 4 repeats, one iteration, no rationalization, was enough. The ceiling under the cap is set by trace length, so the next step for cipher is shorter traces (a second STaR iteration sampled from the STaR adapter, or keeping the shortest correct trace per puzzle), not more of the same.

## Files

| File | What it is |
|---|---|
| [`data/make_star_jsonl.py`](data/make_star_jsonl.py) | `select`, `build`, `check` and `--selftest`; the STaR rule and the mix live here. |
| [`configs/star-dataset-block.yaml`](configs/star-dataset-block.yaml) | The dataset block appended to the recipe copy: new paths, `seq_length: 8192`. |
| [`star_chain.sh`](star_chain.sh), [`data/decode_check.py`](data/decode_check.py) | Steps 4 to 9 unattended, with gates; the decode check the chain runs before training. |
| [`../../tools/sample_traces.py`](../../tools/sample_traces.py) | The sampler; shared with any later STaR pass on other categories. |
| `data/cipher-train.jsonl`, `data/train.jsonl`, `data/val.jsonl` | Generated, git-ignored. |
| `logs/star-cipher-7680.{jsonl,json,log}`, `logs/compare-300-*.{csv,json,-generations.jsonl}` | Generated on the Spark, git-ignored. |
| `results/star-dataset.json` | What `build` kept and dropped; committed. |
