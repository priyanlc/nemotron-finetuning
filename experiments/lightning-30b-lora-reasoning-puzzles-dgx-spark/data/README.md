# Data

Only `make_puzzle_jsonl.py` and this note are committed. `train.csv` and the three JSONL files it produces are git-ignored (the repository's `.gitignore` keeps `experiments/*/data/` to scripts and notes), so each clone rebuilds them with steps 3 to 5 of the [experiment README](../README.md), on the Spark host. Both containers read this folder as `/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark/data/`.

| File | Rows | Made by | One row |
|---|---|---|---|
| `train.csv` | 9,500 | Kaggle (step 3) | `id,prompt,answer` |
| `train.jsonl` | 8,840 | `make_puzzle_jsonl.py` (step 4) | `{"id", "category", "prompt", "completion"}`: the user message and the assistant message (a `<think>` trace, then the `\boxed{}` answer). Shuffled with seed 2. |
| `val.jsonl` | 60 | the same run | 10 per category, drawn with seed 0; the `[val]` loss during training. |
| `test.jsonl` | 600 | the same run | 100 per category, drawn with seed 1 from the rest; `{"id", "category", "prompt", "answer"}` for `tools/compare_puzzles.py`. Never trained on. |

No id appears in two files. The counts per category in `train.jsonl` are bit_ops 1,492, cipher 1,466, gravity 1,487, numeral 1,466, symbol 1,445 and unit_conv 1,484.

**Source and licence.** `train.csv` is the training file of the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle, provided by NVIDIA under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): 3,069,304 bytes, SHA-256 `d204af160633b638448723a437aa51c0db70fd0b64ff92f6ad6f52e5ac6377fa`. The JSONL files are derived from it: the prompt gains the boxed-answer instruction and, for ciphers, a vocabulary hint; the completion is a reasoning trace built from the puzzle's own examples and answer, or a one-sentence template. The README's [Data](../README.md#data) section states the rules. If you redistribute any of these files, credit NVIDIA, link the competition and the licence, and say that the files were changed.

**What the script contains.** The puzzle classifier (keywords in the prompt), the boxed-answer instruction and the cipher vocabulary hint, the trace builders for each category and the template sentences, the seeded splits, and the `--check` renderer that compares every row with the inference prefix. It needs polars; `--check` also needs transformers and jinja2.
