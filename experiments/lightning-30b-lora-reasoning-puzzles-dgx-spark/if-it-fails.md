# If it fails: LoRA on reasoning puzzles, one DGX Spark

*Part of the [experiment README](README.md); the step numbers are its steps. Failures of the shared setup, the Spark, the containers and the model, are in the [HellaSwag experiment's troubleshooting](../lightning-30b-lora-hellaswag-dgx-spark/README.md#if-it-fails).*

| You see | Likely cause | What to do |
|---|---|---|
| `python3 -m venv` fails on `ensurepip` | `python3-venv` not installed | `sudo apt-get install -y python3-venv`; repeat step 2. |
| `ModuleNotFoundError` for `polars` or `transformers` | venv not active | `source .venv/bin/activate` in the experiment folder. |
| `403 Forbidden` from the Kaggle CLI | Rules not accepted, or not logged in | Accept the rules, `kaggle auth login --no-launch-browser`, or `scp` a browser download into `data/`. |
| `401`/`GatedRepoError` in step 5 | Rare: the repo is public and ungated, so this points to a proxy or an HF outage | Retry; if it persists, `hf auth login` with a token from huggingface.co/settings/tokens and retry. |
| `PermissionError` under `~/nemo-shared/.cache` in step 5 | The root-owned container cache was written to | Drop `HF_HOME` and `HF_HUB_OFFLINE` so the download goes to `~/.cache/huggingface`. |
| `PermissionError` under `~/.cache/huggingface` in step 5 | The host cache belongs to root, created by a container that mounted it or by `sudo`, so the download cannot be written | `sudo chown -R "$USER:$USER" ~/.cache/huggingface`; rerun the check. |
| `FAILED: …` in step 5 | Wrong tokenizer or chat template | Check `--model` and that the cached template is Lightning's. |
| Step 8 decode check fails (space after `<think>`, no `<\|im_end\|>`, prompt tokens in the loss) | The chat-template path does not match the native render on this container | Do not train. Use NVIDIA's `make_agent_chat_dataset` (DGX Station Lightning LoRA recipe): rows `{"messages": [user, assistant]}`, trace in `reasoning_content`, `\boxed{}` in `content`, `mask_reasoning_content: false`, `seq_length: 2048`; rerun the check. |
| `num_label_tokens` of 0 in the log | A row with an empty loss mask was skipped | Fix the mask before training. |
| `mem` far above 62 GiB | Sequences longer than expected (run 1 peaks at 492 tokens) | Keep local batch 1; drop long rows rather than truncate (a cut trace loses its `\boxed{}`). |
| Training finishes instantly: `Loading checkpoint from …` | The checkpoint folder holds a finished run and was resumed | One folder per run; keep the `test -e` guard. |
| Server error naming LoRA modules | vLLM rejects the adapter (unexpected: the same layout loaded on 14 October) | Merge with `/opt/Automodel/tools/merge_lora.py --automodel-native-export --dtype bfloat16` and serve the merge as `lightning-puzzles`. |
| The 12-puzzle timing predicts more hours than you have | Long base-model traces | `--n 300`; `--concurrency 32`. Never lower `--max-tokens`: it changes the procedure. |
