# nemotron-finetuning

Fine-tuning experiments on NVIDIA's Nemotron models, run end to end on real hardware: so far one DGX Spark and RunPod GPUs. Each experiment uses NVIDIA's NeMo AutoModel recipes and containers to fine-tune a model, then measures it against the base model, so far through a vLLM server. Every number in this repository comes from a log, and each experiment folder is a self-contained, reproducible guide: the commands in order, the configs, the results, and what they do and don't show.

## Experiments

| Experiment | Model | Method | Data | Hardware | Headline result | Date | Status |
|---|---|---|---|---|---|---|---|
| [lightning-30b-lora-hellaswag-dgx-spark](experiments/lightning-30b-lora-hellaswag-dgx-spark/) | Nemotron 3.5 Lightning 30B-A3B | LoRA r8/α32, 200 steps | HellaSwag | DGX Spark (GB10, 128 GB) | acc_norm 0.644 → 0.671 (+2.7 points, p ≈ 0.02) on 1,000 held-out questions | 2026-10-14 | measured |
| nano-30b-lora-hellaswag-runpod-2xa100 | Nemotron 3 Nano 30B-A3B | LoRA r8/α32, experts split over 2 GPUs | HellaSwag | RunPod, 2 × A100 80 GB | validation loss 2.517 | 2026-10-09 | measured, write-up pending |
| nano-8b-lora-squad-runpod-a6000 | Llama-3.1-Nemotron-Nano-8B-v1 | LoRA r8/α32, 100 steps | SQuAD | RunPod, 1 × RTX A6000 48 GB | validation loss 4.66 → 0.189 | 2026-10-07 | measured, write-up pending |
| lightning-30b-lora-reasoning-puzzles-dgx-spark | Nemotron 3.5 Lightning 30B-A3B | LoRA on JSONL prompt/completion data | Kaggle Nemotron reasoning puzzles | DGX Spark | — | — | planned |

Status labels: `planned` (designed, not run) · `in progress` · `measured` (ran end to end, results committed) · `published` (a post about it is live) · `superseded` (replaced by a later experiment, which the row then links to). "Write-up pending" means the run is measured but its folder has not been written yet.

**Start with** [lightning-30b-lora-hellaswag-dgx-spark](experiments/lightning-30b-lora-hellaswag-dgx-spark/): it walks from a fresh Spark to a measured adapter in 11 steps and explains what happens inside a training step.

## Shared tooling

| Path | What it is |
|---|---|
| [`tools/compare_vllm.py`](tools/compare_vllm.py) | Compares a base model and a LoRA adapter served by one vLLM server: greedy completions side by side, then HellaSwag accuracy by log-likelihood of the four endings (acc and acc_norm, as in lm-evaluation-harness). Standard library plus `datasets`. Other tasks will add their own scorers. |

## Conventions

**Layout.** One folder per experiment under `experiments/`, named `<model>-<method>-<data>-<hardware>`, with the hardware shorthand `dgx-spark`, `dgx-station` or `runpod-<gpu>` (for example `runpod-2xa100`). Inside:

```
experiments/<slug>/
├── README.md      the guide: setup, numbered steps that say where each one runs, results, what can go wrong
├── configs/       the recipe and any blocks appended to it
├── results/       CSVs plus a README that says how they were produced
├── assets/        diagrams and photos (keep images small; one cover photo at most)
└── data/          fetch scripts and notes only; the data itself is git-ignored
```

**Configs.** Either NVIDIA's recipe verbatim (its licence header kept) plus small overlay blocks and command-line overrides, or overrides only, when the recipe ships inside the container image. The README says which.

**Environments.** Each experiment pins its own container tags or requirements. There is no repository-wide environment, because the stacks differ: the AutoModel 26.08 image ships transformers 5.12.1, vLLM is at v0.27.1, NeMo-RL at v0.7.0, and the autoresearch repositories pin transformers 4.51.3.

**On the machine.** New experiments keep their files under `/shared/<slug>/{configs,checkpoints,logs,data}` so that two experiments never share a checkpoint folder (reusing one resumes the old run). The Lightning guide keeps the flat `/shared/{configs,checkpoints,logs}` layout it was measured with.

**Numbers.** Every figure in a README comes from a log, from `training.jsonl` or `validation.jsonl`, or from an evaluation output that is committed or described in `results/README.md`. Results are reported against the base model, with their uncertainty. Absolute benchmark scores are not compared with leaderboards unless the prompt format matches.

## Adding an experiment

```bash
cp -r experiments/_template experiments/<slug>
```

Fill in the README (its comments say what each section is for), add a row to the Experiments table above, and list any redistributed third-party file in [`LICENSES.md`](LICENSES.md).

## Related repositories

- [autoresearch-sft](https://github.com/priyanlc/autoresearch-sft) and [autoresearch-sft-grpo](https://github.com/priyanlc/autoresearch-sft-grpo): an autonomous SFT (and SFT + GRPO) optimisation loop for the Kaggle Nemotron reasoning challenge. They stay separate because the loop depends on their own git history; the planned reasoning-puzzles experiment here will reuse their data preparation and scorer.

## Licences

Code is under the Apache License 2.0 ([`LICENSE`](LICENSE)); text, data and figures are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party material, such as NVIDIA's recipes (Apache-2.0) and the cover photo (CC BY-SA 4.0), is listed in [`LICENSES.md`](LICENSES.md). Models, container images and datasets are not redistributed; each experiment names their licences.
