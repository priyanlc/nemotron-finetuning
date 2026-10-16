# Nemotron 3.5 Lightning 30B-A3B: LoRA on reasoning puzzles, one DGX Spark

*Part of [nemotron-finetuning](../../README.md), a series of fine-tuning experiments on NVIDIA's Nemotron models.*

**Planned: fine-tune Lightning with LoRA on puzzle-style reasoning data from the Kaggle Nemotron reasoning challenge, on the same Spark, containers and recipe as the [HellaSwag experiment](../lightning-30b-lora-hellaswag-dgx-spark/README.md), and measure the adapter against the base model on held-out puzzles.** Status: planned. Nothing below has been run; the config block is untested against the Lightning recipe.

> **Intended setup:** one DGX Spark (GB10, 128 GB) · `nvcr.io/nvidia/nemo-automodel:26.08` for training and `vllm/vllm-openai:v0.27.1` for serving · `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` · NVIDIA's single-GPU LoRA recipe as used in the HellaSwag experiment.

## The question

HellaSwag was a manageable end-to-end test, not a target application: the base model already answers 64% of its questions and the adapter gained 2.7 points. The puzzles should be a more demanding task for the base model; whether LoRA fine-tuning improves performance on them is what this run will test. The evaluation principle stays the same: **train, serve, and compare on unseen examples**.

The data comes from the [NVIDIA Nemotron Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) on Kaggle: about 9,500 few-shot rule-induction puzzles in six categories (bit operations, ciphers, gravity, numerals, symbols and unit conversion), each answered in `\boxed{}` form. The competition is built around Nemotron 3 Nano and caps the LoRA rank at 32; here the puzzles serve as a harder dataset for the Lightning recipe, not as a competition entry. Data preparation and the answer scorer start from [autoresearch-sft](https://github.com/priyanlc/autoresearch-sft), which already works with this data.

## Data

Each puzzle becomes one JSONL line, `{"prompt": "...", "completion": "..."}`, and the Lightning recipe's HellaSwag dataset is replaced by NVIDIA's column-mapped JSONL instruction dataset, which reads such files. What goes into the prompt (the few-shot examples and the question) and whether the completion is the bare `\boxed{}` answer or a reasoning trace ending in it is the first decision to make, and it has to match how the adapter is evaluated. Replace the recipe's `dataset` block with:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /shared/data/train.jsonl
  split: train
  column_mapping:
    question: prompt
    answer: completion
  seq_length: 2048
  answer_only_loss_mask: true
  padding: do_not_pad
  truncation: longest_first
```

`answer_only_loss_mask: true` excludes prompt tokens from the loss, as the HellaSwag dataset does with its context tokens. For validation, repeat the block as `validation_dataset` with `path_or_dataset_id: /shared/data/val.jsonl`, a separate held-out file; keep `split: train`, which is ignored for local files because the file loads as one split. Both blocks are written out in `configs/custom-dataset-block.yaml`.

**Formatting matters.** With the default `use_hf_chat_template: false`, the dataset joins `prompt + completion` without inserting a separator. Include any intended separator, such as a newline, at the end of each prompt. Alternatively, use `use_hf_chat_template: true` to format examples using the tokenizer's chat template, then evaluate through a chat endpoint rather than `/v1/completions`.

As in the HellaSwag experiment, a small validation file gives the `[val]` loss during training and a separate held-out file is kept for the final base-versus-adapter comparison; their sizes are to be set once the data is prepared. The 62 GiB that run used leaves some room for longer sequences, but memory must be measured again with this data.

## Steps (draft)

1. **Steps 1–4 of the HellaSwag guide, unchanged:** check the Spark, start `lightning-lora`, copy the recipe and download the model ([Train](../lightning-30b-lora-hellaswag-dgx-spark/README.md#train-from-a-fresh-spark-to-an-adapter-step-by-step)).
2. **Prepare the data** on the Spark host: `train.jsonl`, `val.jsonl` and a held-out test file under `~/nemo-shared` (`/shared/` inside the containers). The preparation scripts will live in this folder's `data/`; the files themselves stay out of git.
3. **Swap the dataset:** append the two blocks in [`configs/custom-dataset-block.yaml`](configs/custom-dataset-block.yaml) in place of the recipe's `dataset` block, with the paths of step 2. Use a new checkpoint folder and log name; the repository's convention is `/shared/lightning-30b-lora-reasoning-puzzles-dgx-spark/{configs,checkpoints,logs,data}`.
4. **Test before scaling up.** This dataset substitution has not yet been tested against the Lightning recipe used here. Begin with 20 steps and monitor memory; 2,048-token sequences can require substantially more memory than the short HellaSwag examples.
5. **Train with validation** as in step 6 of the guide; the run length follows from the 20-step test and the validation curve.
6. **Serve and compare** as in steps 8–11 of the guide. With `use_hf_chat_template: true`, evaluate through the chat endpoint instead of `/v1/completions`. `compare_vllm.py` scores only HellaSwag today; the puzzles need a scorer that compares the `\boxed{}` answer with the expected one, to be added behind a `--task` option, with the autoresearch scorer as the reference.

## Results

Not run yet.

## If it fails

The HellaSwag guide's [If it fails](../lightning-30b-lora-hellaswag-dgx-spark/README.md#if-it-fails) covers the shared setup: containers, model download, memory, checkpoints and serving. Failures specific to this data will be added here.

## Files in this experiment

| Path | What it is |
|---|---|
| `README.md` | This plan. |
| `configs/custom-dataset-block.yaml` | The `dataset` and `validation_dataset` blocks for NVIDIA's column-mapped JSONL dataset class; untested against the Lightning recipe. |

## Third-party material and licences

- Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, OpenMDW-1.1. Containers: `nvcr.io/nvidia/nemo-automodel:26.08` (NVIDIA Software License Agreement and Product-Specific Terms for NVIDIA AI Products) and `vllm/vllm-openai:v0.27.1` (Apache License 2.0). None are redistributed here.
- The recipe is NVIDIA's `nemotron_nano_v3_5_lightning_singlegpu_lora.yaml` (Apache License 2.0), kept in the [HellaSwag experiment's `configs/`](../lightning-30b-lora-hellaswag-dgx-spark/configs/).
- The puzzle data is subject to the Kaggle competition's rules; none of it is committed here.

## References

- The HellaSwag experiment this plan builds on: [lightning-30b-lora-hellaswag-dgx-spark](../lightning-30b-lora-hellaswag-dgx-spark/README.md)
- NVIDIA Nemotron Reasoning Challenge on Kaggle: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- autoresearch-sft and autoresearch-sft-grpo (data preparation and scorer): https://github.com/priyanlc/autoresearch-sft, https://github.com/priyanlc/autoresearch-sft-grpo
- AutoModel's `ColumnMappedTextInstructionDataset`: https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/llm/column_mapped_text_instruction_dataset.py
- Nemotron 3.5 Lightning model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
