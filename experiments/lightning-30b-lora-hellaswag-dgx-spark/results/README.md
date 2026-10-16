# Measured results

One DGX Spark (GB10), 2026-10-14. Recipe `nemotron_nano_v3_5_lightning_singlegpu_lora.yaml` in `nvcr.io/nvidia/nemo-automodel:26.08`, with `--step_scheduler.max_steps 200 --step_scheduler.val_every_steps 20` and the validation block in `configs/validation-block.yaml`.

- `validation-loss.csv`: validation loss on 64 held-out HellaSwag examples and the learning rate at each of the 10 checks (from the run's `validation.jsonl`). 200 steps took about 24 minutes (09:03:28 to 09:27:15), about 7 s per step including validation, at about 62 GiB of memory and about 107 tokens/s.
- `hellaswag-1000.csv`: HellaSwag validation rows 64–1063 (1,000 questions used for neither training nor validation), scored through `vllm/vllm-openai:v0.27.1` serving the BF16 base model and the step-199 adapter, with `tools/compare_vllm.py --n 1000` (566 s). Paired McNemar test on 80 fixed against 53 broken: p ≈ 0.02; paired 95% range for the acc_norm gain about +0.5 to +4.9 points. An earlier 100-question run (+2 points, 8 fixed, 6 broken) was not significant.

To add the raw outputs from your own run, copy `~/nemo-shared/checkpoints/<run>/training.jsonl`, `validation.jsonl` and `~/nemo-shared/logs/compare_vllm.json` here.
