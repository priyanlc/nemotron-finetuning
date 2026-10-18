# Licences

`nemotron-finetuning` combines work under several licences. The defaults: **code is under the Apache License 2.0** (see `LICENSE`); **text, data and figures are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**. Third-party material and exceptions are listed below; each experiment's README names the licences of the model, dataset and upstream files it uses.

| Part | Licence |
|---|---|
| `tools/compare_vllm.py`, `tools/compare_puzzles.py`, `experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark/data/make_puzzle_jsonl.py`, `assets/diagrams.py` and `requirements.txt`; the config blocks written for this repository (`experiments/*/configs/validation-block.yaml`, `custom-dataset-block.yaml`) | Copyright 2026 Priyan Chandrapala, Apache License 2.0. |
| All Markdown files (`README.md`, `experiments/*/*.md`, `tools/README.md`), `experiments/*/results/`, `experiments/*/assets/*.png` and `*.svg` | Copyright 2026 Priyan Chandrapala, CC BY 4.0. |
| `experiments/lightning-30b-lora-hellaswag-dgx-spark/configs/nemotron_nano_v3_5_lightning_singlegpu_lora.yaml` | Copyright NVIDIA Corporation, Apache License 2.0 (the header is kept in the file). |
| `experiments/lightning-30b-lora-hellaswag-dgx-spark/assets/dgx-spark-oblique-2400.jpg` | "Oblique view of an NVIDIA DGX Spark" by Daniel Lu, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), resized. See `assets/cover-credit.md` beside it. |

Not redistributed here: the model `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (OpenMDW-1.1), the container image `nvcr.io/nvidia/nemo-automodel:26.08` (NVIDIA Software License Agreement and Product-Specific Terms for NVIDIA AI Products), the image `vllm/vllm-openai:v0.27.1` (Apache License 2.0), the HellaSwag dataset (MIT), and the Kaggle reasoning-puzzle data (NVIDIA, CC BY 4.0; what is derived from it is described in `experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark/data/README.md`).
