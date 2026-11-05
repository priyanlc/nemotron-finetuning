# Nemotron 3.5 Lightning 30B-A3B: STaR traces for the cipher puzzles, one DGX Spark

*Part of [nemotron-finetuning](../../README.md). A follow-up to the [reasoning-puzzles experiment](../lightning-30b-lora-reasoning-puzzles-dgx-spark/README.md), which it leaves untouched; it reuses that experiment's data files, containers and served base model.*

**Replace the scripted cipher training traces with the base model's own correct derivations, retrain the LoRA with everything else unchanged, and measure whether the cipher regression goes away without giving back the other categories.** The published adapter scores 0.550 against the base's 0.290 on 300 held-out puzzles, but regressed on cipher from 0.36 to 0.04, and the [higher-cap rescore](../lightning-30b-lora-reasoning-puzzles-dgx-spark/higher-cap-rescore.md) showed the regression widens when the base has room (base 0.54, adapter 0.04 at 16,384 tokens). Every one of the 1,466 scripted cipher traces completes the letter map from the known answer and states the decryption in one sentence, so the adapter learned the shape of a decryption without the work. The STaR recipe (Self-Taught Reasoner, [Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)) builds traces the other way round: the model solves the training puzzles itself, only the final answers are checked, and the correct solutions become the training data. The traces are real work, in the model's own distribution, and never saw the answer. A reader's comment on the write-up suggested this.

> **Setup:** one DGX Spark (GB10, 128 GB) · `nvcr.io/nvidia/nemo-automodel:26.08` to train, `vllm/vllm-openai:v0.27.1` to serve · `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` · NVIDIA's single-GPU LoRA recipe with the dataset block swapped and `seq_length` raised to 8,192 · the original experiment's `data/train.jsonl`, `train.csv`, `val.jsonl` and `test.jsonl` already on the Spark.

## Contents

| Step | Task | Run from | Time |
|---|---|---|---|
| 1 | [Serve the base model](#step-1-serve-the-base-model) | `vllm-lightning` | Minutes |
| 2 | [Select the cipher puzzles and smoke-test the sampler](#step-2-select-the-cipher-puzzles-and-smoke-test-the-sampler) | Spark host, executing inside `lightning-lora` | About 20 minutes |
| 3 | [Sample the base on all 1,466 cipher training puzzles](#step-3-sample-the-base-on-all-1466-cipher-training-puzzles) | Same, detached | About 17 hours (estimate) |
| 4 | [Build and check the STaR training set](#step-4-build-and-check-the-star-training-set) | Same | Minutes |
| 5 | [Write the recipe config](#step-5-write-the-recipe-config) | Spark host | Minutes |
| 6 | [Decode check and 20-step smoke run](#step-6-decode-check-and-20-step-smoke-run) | `lightning-lora`, GPU freed | About 40 minutes |
| 7 | [Train 200 steps](#step-7-train-200-steps) | `lightning-lora` | 1.5 to 3.5 hours (estimate) |
| 8 | [Serve the base and both adapters](#step-8-serve-the-base-and-both-adapters) | `vllm-lightning` | Minutes |
| 9 | [Score the same 300 puzzles](#step-9-score-the-same-300-puzzles) | Spark host, executing inside `lightning-lora` | About 5 hours |
| 10 | [Write up](#step-10-write-up) | Anywhere | 2 hours |

Then [Results](#results) and [Files](#files). Times marked estimate are replaced with measured values as the steps are run.

Steps 4 to 9 can run unattended: [`star_chain.sh`](star_chain.sh) waits for the sampler's summary file, then runs them in order with a gate after each (kept traces, `check`, the config's two `seq_length` lines, the decode check, the smoke run's peak memory under 85 GiB, a `LOWEST_VAL` adapter, the server listing all three model names, both `.json` summaries) and appends one line per stage to `logs/chain.status`. Start it on the Spark host with `nohup setsid bash $S/star_chain.sh > $S/logs/chain.log 2>&1 &`; the first failed gate stops it with a `FAILED` line naming the log to read.

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

The smoke run prints the cap (7,680), the server window and a progress line with running counts of finished, first-box-correct and truncated generations. Expect about a third kept, lines appearing in `star-smoke.jsonl` while the run is going (the file is appended per generation), and a second invocation with `--resume` reporting `12 already in` and sampling nothing. Measured 2026-11-05: *pending*.

## Step 3: Sample the base on all 1,466 cipher training puzzles

```bash
docker exec -d lightning-lora bash -c "python3 $T/sample_traces.py --data $S/data/cipher-train.jsonl \
  --out $S/logs/star-cipher-7680.jsonl --concurrency 32 --resume > $S/logs/star-cipher-7680.log 2>&1"
tail -n 3 $S/logs/star-cipher-7680.log; wc -l $S/logs/star-cipher-7680.jsonl       # from the Spark host, any time
```

Detached, so a lost connection does not stop it; `--resume` makes the same command safe to run again after a container restart. Estimate: 1,466 puzzles at about 7,100 tokens each (two thirds reach the cap) at the server's ~175 tokens per second aggregate is about 17 hours. The `.json` summary beside the output has the per-category counts when it finishes.

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

Stop the vLLM server first (`Ctrl-C` in its terminal); the GPU is shared. Then the original experiment's [step 8](../lightning-30b-lora-reasoning-puzzles-dgx-spark/README.md#step-8-decode-check-and-20-step-smoke-run) decode check with this folder's `val.jsonl` and `seq_length=8192`, and its 20-step run with `$S/configs/star_lora.yaml` and a checkpoint folder under `$S/checkpoints/`. Watch `mem` in the step records: the original ran at about 62 GiB on rows of at most 492 tokens, and the first batches that contain a 7,000-token cipher row show the new peak. If it approaches 90 GiB, rebuild with `--seq-length 6144` and run again. Measured: *pending*.

## Step 7: Train 200 steps

The original's [step 9](../lightning-30b-lora-reasoning-puzzles-dgx-spark/README.md#step-9-train-200-steps-with-validation) command with `$S/configs/star_lora.yaml`, `--checkpoint.checkpoint_dir $S/checkpoints/$RUN`, validation and a checkpoint every 20 steps. Then `ln -sfn $RUN $S/checkpoints/star-200` and note which step `LOWEST_VAL` points at. Record the validation losses in `results/validation-loss.csv`. Measured: *pending*.

## Step 8: Serve the base and both adapters

The original's step 10 command with two LoRA modules, so the published adapter and the new one are scored in the same server session:

```bash
--lora-modules lightning-puzzles=$E/checkpoints/puzzles-200/LOWEST_VAL/model lightning-star=$S/checkpoints/star-200/LOWEST_VAL/model
```

`--max-model-len 8192` is enough for the competition cap; `32768` allows a 16,384-cap run afterwards.

## Step 9: Score the same 300 puzzles

Both runs score the 300 puzzles of the original step 12 (`--n 300` selects the same rows) at the competition cap:

```bash
docker exec -it lightning-lora python3 $T/compare_puzzles.py --n 300 --concurrency 32 \
  --base-name lightning-puzzles --adapter-name lightning-star --out $S/logs/compare-300-v1-vs-star.csv
docker exec -it lightning-lora python3 $T/compare_puzzles.py --n 300 --concurrency 32 \
  --base-name lightning-base --adapter-name lightning-star --out $S/logs/compare-300-star.csv
```

The first pairs the published adapter against the new one, puzzle by puzzle, and is quick because only the cipher traces are long; its `lightning-puzzles` column should reproduce the published result within run-to-run churn (the higher-cap rescore measured the adapter's churn at 3 gained and 2 lost over 300 puzzles). The second is the headline, paired against the base, about 3.5 hours because the base's generations are long. Optionally repeat the second at `--max-tokens 16384` and merge with `cap_sweep.py`.

## Step 10: Write up

Commit the two `.json` summaries and `results/star-dataset.json`, fill in [Results](#results), and add the new row to the top-level README. The sentence the write-up needs is whether cipher is back at the base's level under the competition cap (about 0.36) and what the other five categories did.

## Results

*Pending. The sampling run started 2026-11-05.*

| category | base | published adapter | STaR adapter |
|---|---|---|---|
| cipher | 0.36 | 0.04 | |
| bit_ops | 0.02 | 0.24 | |
| gravity | 0.22 | 0.92 | |
| numeral | 0.98 | 1.00 | |
| symbol | 0.10 | 0.10 | |
| unit_conv | 0.06 | 1.00 | |
| **all 300** | **0.29** | **0.55** | |

Base and published-adapter columns are the original experiment's October run at the 7,680 cap; the STaR column will come from the paired runs of step 9.

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
