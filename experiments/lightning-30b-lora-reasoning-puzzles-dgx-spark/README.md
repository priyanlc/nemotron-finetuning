# Nemotron 3.5 Lightning 30B-A3B: LoRA on reasoning puzzles, one DGX Spark

*Part of [nemotron-finetuning](../../README.md), a series of fine-tuning experiments on NVIDIA's Nemotron models.*

**Fine-tune Nemotron 3.5 Lightning with LoRA on the Kaggle reasoning puzzles, on the same Spark, containers and recipe as the [HellaSwag experiment](../lightning-30b-lora-hellaswag-dgx-spark/README.md), and compare the adapter with the base model by generating answers to 600 held-out puzzles.** Status: in progress. Steps 1 to 5 ran on a Mac on 18 October 2026; steps 6 to 13 have not been run on this data yet, and their times are estimates.

> **Setup:** one DGX Spark (GB10, 128 GB) · `nvcr.io/nvidia/nemo-automodel:26.08` for training, `vllm/vllm-openai:v0.27.1` for serving · `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` · NVIDIA's single-GPU LoRA recipe, unchanged except for the dataset · a laptop with Python 3.12 for the data steps.

## Contents

Follow the steps in order. The model, the hardware and the two containers are explained in the [HellaSwag guide](../lightning-30b-lora-hellaswag-dgx-spark/README.md#how-the-pieces-fit-together).

| Step | Task | Run from | Time |
|---|---|---|---|
| 1 | [Create the virtual environment](#step-1-create-the-virtual-environment) | Laptop | 2 minutes |
| 2 | [Get the competition data](#step-2-get-the-competition-data) | Laptop, in the venv | 1 minute |
| 3 | [Build the JSONL files](#step-3-build-the-jsonl-files) | Laptop, in the venv | 2 seconds |
| 4 | [Check the training text against the inference prefix](#step-4-check-the-training-text-against-the-inference-prefix) | Laptop, in the venv | 10 seconds, plus the tokenizer download once |
| 5 | [Copy the data to the Spark](#step-5-copy-the-data-to-the-spark) | Laptop | 1 minute |
| 6 | [Set up the Spark and the experiment folder](#step-6-set-up-the-spark-and-the-experiment-folder) | Spark host, then `lightning-lora` | Minutes if the HellaSwag setup exists; otherwise the image pull and the 60 GB download |
| 7 | [Put the puzzle data into the recipe](#step-7-put-the-puzzle-data-into-the-recipe) | `lightning-lora` | 15 minutes |
| 8 | [Decode check and 20-step smoke run](#step-8-decode-check-and-20-step-smoke-run) | `lightning-lora` | About 30 minutes (estimate) |
| 9 | [Train 200 steps with validation](#step-9-train-200-steps-with-validation) | `lightning-lora` | 45 to 90 minutes (estimate) |
| 10 | [Free the GPU and serve base and adapter](#step-10-free-the-gpu-and-serve-base-and-adapter) | Second Spark terminal, then `vllm-lightning` | About 10 minutes |
| 11 | [Time the scorer on 12 puzzles](#step-11-time-the-scorer-on-12-puzzles) | Spark host, executing inside `lightning-lora` | Minutes |
| 12 | [Run the full 600-puzzle comparison](#step-12-run-the-full-600-puzzle-comparison) | Spark host, executing inside `lightning-lora` | 2 to 5 hours (estimate) |
| 13 | [Inspect the generations and write up](#step-13-inspect-the-generations-and-write-up) | Spark host, then laptop | 3 to 4 hours |

**After the walkthrough:** [Results](#results) · [If it fails](#if-it-fails) · [Files](#files-in-this-experiment) · [Licences](#third-party-material-and-licences) · [References](#references)

## Data

`train.csv` from the [NVIDIA Nemotron Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge): 9,500 few-shot rule-induction puzzles (`id,prompt,answer`) in six balanced categories, each answered inside `\boxed{}`. NVIDIA, CC BY 4.0; not committed here. One script turns it into three files:

![How train.csv becomes three JSONL files](assets/01-data-pipeline.png)

*The data steps at a glance: classify, split with fixed seeds, build a prompt and a completion per row, write three files, check them against the chat template, copy them to the Spark.*

Each row of `train.jsonl` holds the user message (`prompt`: the puzzle plus the boxed-answer instruction; cipher prompts also get the 77-word vocabulary their plaintexts come from) and the assistant message (`completion`: a short reasoning trace in `<think>` tags, then the boxed answer). Row `ca4d10f5`:

```json
{"id": "ca4d10f5", "category": "gravity",
 "prompt": "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:\nFor t = 1.69s, distance = 9.47 m\nFor t = 1.58s, distance = 8.28 m\nFor t = 2.26s, distance = 16.93 m\nFor t = 3.49s, distance = 40.38 m\nNow, determine the falling distance for t = 2.81s given d = 0.5*g*t^2.\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`",
 "completion": "<think>\nUsing d = 0.5*g*t^2: g = 2d/t^2 = 2*9.47/1.69^2 = 6.6314. Apply to find d for the new t. Answer: 26.18\n</think>\n\\boxed{26.18}"}
```

The trace is computed from the puzzle's own examples and answer where a rule can be read off them; otherwise it is one fixed sentence per category:

![Which reasoning trace does a puzzle get?](assets/03-trace-rules.png)

*How much of each category gets a computed trace, and the rule that computes it.*

AutoModel renders each pair through the Lightning chat template (`use_hf_chat_template: true`) with the loss on the assistant turn only. At evaluation the scorer sends the same template up to the open `<think>`, so every training sequence is the served prompt followed by the completion. Step 4 checks this on all 9,500 rows; step 8 repeats it inside the container.

![One puzzle, one training sequence, one served prompt](assets/02-training-sequence.png)

*The same row as the template renders it. Blue: the served prompt, which the scorer sends unchanged. Green: the only tokens that carry the loss.*

Splits: 10 puzzles per category with seed 0 to `val.jsonl` (the `[val]` loss during training), 100 per category with seed 1 to `test.jsonl` (the final comparison, never trained on), the remaining 8,840 shuffled with seed 2 to `train.jsonl`. The longest sequence is 492 tokens (mean 240), so the recipe's `seq_length: 2048` is never reached; 200 steps of 8 rows see 1,600 of them. Both models get identical prompts at evaluation, so the result is adapter versus base under one procedure.

## Prepare the data: on your laptop, in a virtual environment

### Step 1: Create the virtual environment

**Run on your laptop.** Needs Python 3.12 and [uv](https://docs.astral.sh/uv/) (or `python3 -m venv` and `pip`); no GPU, no PyTorch. `.venv/` is git-ignored.

```bash
git clone https://github.com/priyanlc/nemotron-finetuning ~/codebase/nemotron-finetuning   # skip if you have a clone
cd ~/codebase/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python -c "import polars, transformers, jinja2; print(polars.__version__, transformers.__version__, jinja2.__version__)"
```

Expected: `2.0.0 5.20.1 3.1.6`, after a warning that PyTorch was not found (only the tokenizer is used). Every later laptop command assumes the environment is active and this folder is the working directory.

### Step 2: Get the competition data

**Run on your laptop, in the venv.** `train.csv` goes into `data/`, which git ignores. Download it from the [competition's data page](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/data) with an account that has accepted the competition rules, in the browser (Download All, unzip into `data/`) or with the Kaggle CLI:

```bash
kaggle auth login                       # opens a browser; an API token in ~/.kaggle/kaggle.json also works
kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p data
unzip -o data/nvidia-nemotron-model-reasoning-challenge.zip -d data
shasum -a 256 data/train.csv
```

Expected: 3,069,304 bytes, SHA-256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`. The CLI route was not exercised here: the file used came from the competition page in March 2026, and the checksum is the check that you have the same data.

### Step 3: Build the JSONL files

**Run on your laptop, in the venv:**

```bash
python data/make_puzzle_jsonl.py
```

Expected output, in about two seconds:

```text
train.jsonl: 8840 rows {'bit_ops': 1492, 'cipher': 1466, 'gravity': 1487, 'numeral': 1466, 'symbol': 1445, 'unit_conv': 1484}
val.jsonl: 60 rows {'bit_ops': 10, 'cipher': 10, 'gravity': 10, 'numeral': 10, 'symbol': 10, 'unit_conv': 10}
test.jsonl: 600 rows {'bit_ops': 100, 'cipher': 100, 'gravity': 100, 'numeral': 100, 'symbol': 100, 'unit_conv': 100}
cipher vocabulary hint: 77 words
```

Running it again gives the same files. `--val-per-type`, `--test-per-type`, `--train-csv` and `--out` change the defaults.

### Step 4: Check the training text against the inference prefix

**Run on your laptop, in the venv.** The check needs the Lightning tokenizer: if it is not in your Hugging Face cache, accept the model's terms on its Hugging Face page and log in once (only the tokenizer files are downloaded).

```bash
hf auth login                              # once, if the tokenizer is not cached
python data/make_puzzle_jsonl.py --check
```

`--check` rebuilds the files, renders every row the way AutoModel will (system, user and assistant turns through the chat template) and the way the scorer will (the user turn with the generation prompt), and requires the first to be the second followed by the completion and `<|im_end|>`. Expected after the four lines of step 3, in about ten seconds:

```text
train.jsonl: 8840 rows render as prefix + completion; prompt tokens max 392, mean 184; completion tokens max 191, mean 56; total max 492, mean 240; rows over 2048: 0
val.jsonl: 60 rows render as prefix + completion; prompt tokens max 353, mean 184; completion tokens max 108, mean 55; total max 461, mean 239; rows over 2048: 0
test.jsonl: 600 rows render as prefix + completion; prompt tokens max 382, mean 185
check OK
```

A `FAILED:` line means the training text differs from what the model will see when served: do not continue until it passes.

### Step 5: Copy the data to the Spark

**Run on your laptop.** This experiment keeps its files under `/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/{data,configs,checkpoints,logs}` (`~/nemo-shared` on the Spark is `/shared` inside both containers), so it never shares a checkpoint folder with the HellaSwag run:

```bash
SPARK=<user>@<spark-host>
ssh $SPARK 'mkdir -p ~/nemo-shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/{data,configs,checkpoints,logs}'
scp data/train.jsonl data/val.jsonl data/test.jsonl $SPARK:~/nemo-shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/data/
```

## Train: on the Spark

### Step 6: Set up the Spark and the experiment folder

**Run on the Spark host.** If the HellaSwag experiment has run on this Spark, the container, the clone and the model exist: `docker start -ai lightning-lora` reopens the container (`Ctrl-p`, `Ctrl-q` detaches; `docker attach lightning-lora` returns) and you can skip to the last block. Otherwise create them (the HellaSwag guide's steps 1 to 4 explain each command):

```bash
nvidia-smi -L && free -g && df -h ~            # GB10; about 120 GB in total; about 200 GB free
mkdir -p ~/nemo-shared
git clone https://github.com/priyanlc/nemotron-finetuning ~/nemo-shared/nemotron-finetuning
docker run -it \
  --name lightning-lora \
  --gpus all --network host \
  --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HOME=/shared/.cache/huggingface \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v ~/nemo-shared:/shared \
  nvcr.io/nvidia/nemo-automodel:26.08
```

**Inside `lightning-lora`**, download the model once, after accepting its OpenMDW-1.1 terms on Hugging Face:

```bash
hf auth login                 # token from huggingface.co/settings/tokens
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```

Then, inside `lightning-lora`, make the experiment's folders and update the clone:

```bash
E=/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark
mkdir -p $E/configs $E/checkpoints $E/logs
cd /shared/nemotron-finetuning && git pull
ls $E/data                                     # train.jsonl  val.jsonl  test.jsonl
```

Set `E` again in every new shell; the commands below use it.

### Step 7: Put the puzzle data into the recipe

**Run inside `lightning-lora`.** Copy NVIDIA's recipe from the HellaSwag experiment, delete its HellaSwag `dataset:` block and append [`configs/custom-dataset-block.yaml`](configs/custom-dataset-block.yaml): the training and validation datasets on the JSONL files, plus the validation dataloader. Nothing else changes: LoRA rank 8, alpha 32, AdamW 1e-4 to 1e-5, batch 8 × 1.

```bash
R=/shared/nemotron-finetuning/experiments
cp $R/lightning-30b-lora-hellaswag-dgx-spark/configs/nemotron_nano_v3_5_lightning_singlegpu_lora.yaml $E/configs/puzzles_lora.yaml
sed -i '/^dataset:$/,/^  split: train$/d' $E/configs/puzzles_lora.yaml
grep -c hellaswag $E/configs/puzzles_lora.yaml                                       # expect 0
cat $R/lightning-30b-lora-reasoning-puzzles-dgx-spark/configs/custom-dataset-block.yaml >> $E/configs/puzzles_lora.yaml
grep -n '^[a-z_]*:' $E/configs/puzzles_lora.yaml      # dataset, validation_dataset, validation_dataloader once each
```

Run the `sed` before the `cat` (the appended block also contains `split: train`), and append only once. In the block, `use_hf_chat_template: true` renders the rows through the chat template and `answer_only_loss_mask: true` keeps the loss on the assistant turn; `seq_length: 2048` is a cap no row reaches.

### Step 8: Decode check and 20-step smoke run

**Run inside `lightning-lora`.** First the decode check: it loads only the tokenizer and the dataset class, and every row of `val.jsonl` must decode to the template's native render with the loss on the assistant turn only.

```bash
python3 - <<'EOF'
import json
from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import ColumnMappedTextInstructionDataset
E = "/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark"
tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
ds = ColumnMappedTextInstructionDataset(
    path_or_dataset_id=f"{E}/data/val.jsonl", column_mapping={"question": "prompt", "answer": "completion"},
    tokenizer=tok, use_hf_chat_template=True, answer_only_loss_mask=True,
    seq_length=2048, padding="do_not_pad", truncation="longest_first")
rows = [json.loads(l) for l in open(f"{E}/data/val.jsonl")]
for i, r in enumerate(rows):
    s = ds[i]
    text = tok.decode(s["input_ids"])
    expected = tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}], tokenize=False)
    assert text.rstrip("\n") == expected.rstrip("\n"), (i, text[-200:])
    target = tok.decode([l for l in s["labels"] if l != -100])
    assert target.rstrip("\n").endswith("<|im_end|>"), (i, target[-80:])
    assert "<think>\n" in target and "\\boxed{" in target and "<think>\n " not in text, (i, target[:120])
    assert r["prompt"][:60] not in target, (i, "prompt tokens are in the loss")
s0 = ds[0]
print(len(rows), "rows OK; supervised tokens in row 0:", sum(l != -100 for l in s0["labels"]), "of", len(s0["input_ids"]))
print("loss span of row 0 starts:", repr(tok.decode([l for l in s0["labels"] if l != -100])[:70]))
EOF
```

Pass: `60 rows OK`, a supervised count well below the sequence length, and a loss span that starts at or just after `<|im_start|>assistant` and ends at `<|im_end|>`. An assertion failure means the training text differs from what the model will see at inference: do not train ([If it fails](#if-it-fails) has the fallback).

Then NVIDIA's unchanged 20-step schedule on the puzzle data, with one validation pass at step 10. `tmux` keeps the run alive if the connection drops (`Ctrl-b`, `d` detaches; `tmux attach -t train` returns):

```bash
apt-get install -y tmux 2>/dev/null; tmux new -s train     # or tmux attach -t train
E=/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark
cd /opt/Automodel
RUN=puzzles-test
test -e $E/checkpoints/$RUN && echo "exists, pick another name" || \
CUDA_VISIBLE_DEVICES=0 automodel $E/configs/puzzles_lora.yaml \
  --nproc-per-node 1 \
  --step_scheduler.val_every_steps 10 \
  --checkpoint.checkpoint_dir $E/checkpoints/$RUN \
  2>&1 | tee $E/logs/$RUN.log
```

Model loading takes several minutes; then 20 step records print, two `[val]` lines appear, and the adapter lands in `$E/checkpoints/puzzles-test/epoch_0_step_19/model/`. Read the numbers from the run folder:

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('$E/checkpoints/puzzles-test/training.jsonl')]
print(sorted(rows[0]))
for r in rows[::4] + rows[-1:]: print({k: r.get(k) for k in ('step', 'loss', 'grad_norm', 'lr', 'mem', 'tps')})"
grep '\[val\]' $E/logs/puzzles-test.log
ls $E/checkpoints/puzzles-test/epoch_0_step_19/model/
```

Pass before step 9: `mem` near the HellaSwag run's 62 GiB (above 90 GiB is a stop); `loss` falls over the 20 steps; `grad_norm` finite; two `[val]` lines; `adapter_model.safetensors` and `adapter_config.json` present; seconds per step noted from the log timestamps.

### Step 9: Train 200 steps with validation

**Run inside `lightning-lora`, in `tmux`.** The HellaSwag schedule on the puzzle data: 200 steps of 8 rows, validation every 20 steps and a checkpoint at every validation, so `LOWEST_VAL` is a real choice. The run name is timestamped and an existing folder is refused, because AutoModel resumes any run whose checkpoint folder exists:

```bash
E=/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark
cd /opt/Automodel
RUN=puzzles-200-$(date +%m%d-%H%M)
test -e $E/checkpoints/$RUN && echo "exists, pick another name" || \
CUDA_VISIBLE_DEVICES=0 automodel $E/configs/puzzles_lora.yaml \
  --nproc-per-node 1 \
  --step_scheduler.max_steps 200 --step_scheduler.val_every_steps 20 --step_scheduler.ckpt_every_steps 20 \
  --checkpoint.checkpoint_dir $E/checkpoints/$RUN \
  2>&1 | tee $E/logs/$RUN.log
echo "run: $RUN"; grep '\[val\]' $E/logs/$RUN.log
```

Expect 10 to 20 s per step (the HellaSwag run took 7 s at a third of the tokens), so 35 to 70 minutes after loading. When it finishes, give the run a stable name and pick the checkpoint:

```bash
ln -sfn $RUN $E/checkpoints/puzzles-200
ln -sf $RUN.log $E/logs/puzzles-200.log
readlink $E/checkpoints/puzzles-200/LOWEST_VAL        # the step with the lowest validation loss
python3 -c "
import json
for r in map(json.loads, open('$E/checkpoints/puzzles-200/validation.jsonl')): print({k: r.get(k) for k in ('step', 'val_loss', 'lr', 'mem')})"
ls $E/checkpoints/puzzles-200/LOWEST_VAL/model/
```

Serve `LOWEST_VAL` in step 10 and record its step (validation loss on 60 rows is noisy: if it beats step 199 by a hair, say so). If validation loss is still falling clearly at step 199, start a fresh 400-step run in a new folder rather than resuming, which would keep the old learning-rate schedule.

## Evaluate: serve the base model and the adapter with vLLM

### Step 10: Free the GPU and serve base and adapter

**Run on the Spark host, in a second terminal**, once training has finished and `free -g` shows the memory back:

```bash
docker run -it --rm --name vllm-lightning \
  --gpus all --ipc=host --network host \
  -v ~/nemo-shared:/shared \
  -e HF_HOME=/shared/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.27.1
```

**Inside `vllm-lightning`**, serve the base model and the adapter under two names:

```bash
E=/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --served-model-name lightning-base \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --mamba-backend flashinfer \
  --enable-lora --max-lora-rank 8 \
  --lora-modules lightning-puzzles=$E/checkpoints/puzzles-200/LOWEST_VAL/model
```

`--max-model-len 8192` with the 7,680-token generation budget mirrors the competition; `--max-lora-rank` must equal the adapter's rank. Wait for `Application startup complete` before sending requests.

### Step 11: Time the scorer on 12 puzzles

[`tools/compare_puzzles.py`](../../tools/compare_puzzles.py) renders each puzzle of `test.jsonl` as step 4 did, asks both model names for a greedy completion of up to 7,680 tokens, takes the last `\boxed{…}` as the answer and scores it as the competition does (exact string ignoring case, or numbers within 1%; no box or a truncated generation counts as wrong). It reports accuracy per category, the answers the adapter fixed and broke, the gain with a paired 95% interval and an exact McNemar p-value; [tools/README.md](../../tools/README.md#compare_puzzlespy) has the formulas and options.

**Run on the Spark host, in a terminal separate from the server.** Copy the script to the shared directory, run its self-test, then time 12 puzzles, two per category:

```bash
cp ~/nemo-shared/nemotron-finetuning/tools/compare_puzzles.py ~/nemo-shared/
docker start lightning-lora
docker exec -it lightning-lora python3 /shared/compare_puzzles.py --selftest
docker exec -it lightning-lora python3 /shared/compare_puzzles.py --n 12 \
  --out /shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/logs/compare-12.csv
```

Read one generation from each model before trusting the numbers:

```bash
docker exec -it lightning-lora python3 -c "
import json
rows = [json.loads(l) for l in open('/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/logs/compare-12-generations.jsonl')]
for r in rows[:2]: print('==', r['model'], r['finish'], r['tokens'], 'tokens, pred', r['pred']); print(r['text'][:1200])"
```

Multiply the 12-puzzle time by 50 for the full set; expect hours, since the base model's traces can run to thousands of tokens. If that does not fit, run `--n 300` (50 per category) and say so in the write-up.

### Step 12: Run the full 600-puzzle comparison

**Run on the Spark host, in the second terminal:**

```bash
docker exec -it lightning-lora python3 /shared/compare_puzzles.py --n 0 --concurrency 16 \
  --out /shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/logs/compare-600.csv \
  2>&1 | tee ~/nemo-shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/logs/compare-600.log
```

It prints accuracy per category for both models, the answers fixed and broken, the gain with its interval and p-value, and how many generations of each model hit the token limit. Three files land in `logs/`: the per-row CSV, a `.json` summary and `-generations.jsonl` with every generation. Both models get the same prompts, so the comparison is base versus adapter under one procedure; a result near p = 0.05 deserves a second run, since vLLM's batching can flip a few greedy answers. Stop the server with `Ctrl-C` when done.

### Step 13: Inspect the generations and write up

**Run on the Spark host.** Profile the generations (boxed-answer rate, truncation, trace length per model) and the fixed and broken counts per category:

```bash
L=/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/logs
docker exec -it lightning-lora python3 -c "
import json, statistics as st
rows = [json.loads(l) for l in open('$L/compare-600-generations.jsonl')]
for m in sorted({r['model'] for r in rows}):
    rs = [r for r in rows if r['model'] == m]; toks = sorted(r['tokens'] or 0 for r in rs)
    print(m, '| boxed answer', round(sum(r['pred'] is not None for r in rs) / len(rs), 3), '| hit the limit', sum(r['finish'] == 'length' for r in rs),
          '| tokens median', st.median(toks), 'p90', toks[int(0.9 * len(toks)) - 1])"
docker exec -it lightning-lora python3 -c "
import csv, collections
c = collections.Counter()
for r in csv.DictReader(open('$L/compare-600.csv')): c[(r['category'], r['base_ok'], r['adapter_ok'])] += 1
for cat in sorted({k[0] for k in c}):
    print(f'{cat:10} fixed {c[(cat, \"0\", \"1\")]:3}  broke {c[(cat, \"1\", \"0\")]:3}  both right {c[(cat, \"1\", \"1\")]:3}  both wrong {c[(cat, \"0\", \"0\")]:3}')"
```

Then read a few adapter generations per category next to the base's (does the think block copy the template verbatim, is the box always there, does the base run out of tokens?), and update this folder as the [template](../_template/README.md) asks: this README's status and results, `configs/puzzles_lora.yaml` as run, `results/validation-loss.csv` from `validation.jsonl`, `results/puzzles-600.csv` from the per-row CSV, and a `results/README.md`. The JSONL data and the generations stay out of git.

## Results

Not run yet. This section will report the validation-loss curve and the `LOWEST_VAL` step, then accuracy per category for base and adapter on the 600 puzzles, the answers fixed and broken, the gain with its 95% interval and p-value, the truncation counts, and the times and memory, all from files committed under `results/`.

## If it fails

The HellaSwag guide's [If it fails](../lightning-30b-lora-hellaswag-dgx-spark/README.md#if-it-fails) covers the shared setup. Specific to this experiment:

| You see | Likely cause | What to do |
|---|---|---|
| `ModuleNotFoundError` for `polars` or `transformers` | The venv is not active | `source .venv/bin/activate` in the experiment folder. |
| `401` or `GatedRepoError` in step 4 | Tokenizer download refused | Accept the model's terms on Hugging Face, then `hf auth login`. |
| `403 Forbidden` from the Kaggle CLI | Competition rules not accepted, or not logged in | Accept the rules on the competition page, `kaggle auth login`, or download the zip in the browser. |
| `FAILED: …` in step 4 | A different tokenizer or chat template | Check the model id in `--model` and that the cached template is Lightning's. |
| The step 8 decode check fails (a space after `<think>`, no `<\|im_end\|>`, prompt tokens in the loss) | The chat-template path does not reproduce the native render on this container | Do not train. Switch to NVIDIA's `make_agent_chat_dataset` (the DGX Station Lightning LoRA recipe): rows are `{"messages": [user, assistant]}` with the trace in `reasoning_content` and `\boxed{}` in `content`; set `mask_reasoning_content: false` and `seq_length: 2048`; rerun the decode check. |
| `num_label_tokens` of 0 in the log | A row with an empty loss mask was skipped | Fix the mask before training. |
| `mem` far above 62 GiB | Longer sequences than expected (run 1 is 492 tokens at most) | Keep local batch 1; drop rather than truncate long rows (a cut trace loses its `\boxed{}`). |
| Training finishes instantly: `Loading checkpoint from …` | The checkpoint folder holds a finished run and was resumed | One folder per run; keep the `test -e` guard. |
| Server error naming LoRA modules | vLLM rejects the adapter (not expected: the same layout loaded on 14 October) | Merge with `/opt/Automodel/tools/merge_lora.py --automodel-native-export --dtype bfloat16` and serve the merged model as `lightning-puzzles`, as in the HellaSwag guide. |
| The 12-puzzle timing predicts more hours than you have | Long base-model traces | `--n 300`; raise `--concurrency` to 32. Do not lower `--max-tokens`: it changes the procedure. |

## Files in this experiment

| Path | What it is |
|---|---|
| `README.md` | This guide. |
| `requirements.txt` | The laptop environment for steps 1 to 4, pinned to the versions tested. |
| `data/make_puzzle_jsonl.py` | Builds and checks the three JSONL files (steps 3 and 4). |
| `data/README.md` | The data files, their licence and checksum. The data itself is git-ignored. |
| `configs/custom-dataset-block.yaml` | The dataset blocks appended to the recipe in step 7. |
| `../lightning-30b-lora-hellaswag-dgx-spark/configs/nemotron_nano_v3_5_lightning_singlegpu_lora.yaml` | NVIDIA's recipe, verbatim; step 7 copies it. |
| `../../tools/compare_puzzles.py` | The scorer (steps 11 and 12); `../../tools/README.md` explains it. |
| `assets/` | The three diagrams (PNG and SVG) and `diagrams.py`, which draws them. |
| `results/` | To be added after the run. |

## Third-party material and licences

- Model `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (OpenMDW-1.1); containers `nvcr.io/nvidia/nemo-automodel:26.08` (NVIDIA Software License Agreement) and `vllm/vllm-openai:v0.27.1` (Apache-2.0). None are redistributed here.
- NVIDIA's recipe `nemotron_nano_v3_5_lightning_singlegpu_lora.yaml` (Apache-2.0), kept in the HellaSwag experiment's `configs/`.
- The puzzle data: NVIDIA, via Kaggle, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); not committed, derived files described in [`data/README.md`](data/README.md).

## References

- The HellaSwag experiment this guide builds on: [lightning-30b-lora-hellaswag-dgx-spark](../lightning-30b-lora-hellaswag-dgx-spark/README.md)
- NVIDIA Nemotron Reasoning Challenge on Kaggle: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- AutoModel's `ColumnMappedTextInstructionDataset`: https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/llm/column_mapped_text_instruction_dataset.py
- NVIDIA, Lightning LoRA on one DGX Station (the `make_agent_chat_dataset` fallback): https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/dgx-station-recipes/lora.md
- Nemotron 3.5 Lightning model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16

*Licences for the model, containers and dataset: see [LICENSES.md](../../LICENSES.md). The data steps ran on an Apple-silicon Mac on 18 October 2026; the Spark steps have not been run on this data yet.*
