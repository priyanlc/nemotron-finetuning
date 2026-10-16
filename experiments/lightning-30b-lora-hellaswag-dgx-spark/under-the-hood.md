# Under the hood

*Part of the [Lightning guide](README.md): what the recipe's settings do and what happens inside one optimiser step. You can complete the walkthrough without it.*

## The recipe's settings, explained

The full recipe is `configs/nemotron_nano_v3_5_lightning_singlegpu_lora.yaml`. Its key settings are:

| Setting | Value | What it does |
|---|---|---|
| `step_scheduler` | global 8, local 1, 20 steps | 8 examples per update, processed one at a time and accumulated (8 passes per step on one GPU). |
| `num_nextn_predict_layers`, `mtp_use_repeated_layer`, `mtp_loss_scaling_factor` | 2, true, 0.1 | Keeps Lightning's multi-token prediction training on: the checkpoint's one MTP layer is reused to predict 2 extra tokens ahead, adding 0.1 × its loss. |
| `backend` | attn `te`, linear `torch`, rms_norm `torch_fp32`, experts `torch_mm`, dispatcher `torch` | Kernel choices: Transformer Engine attention, PyTorch grouped matrix multiplies for the experts, no DeepEP. All experts stay on the one GPU. |
| `peft` | dim 8, alpha 32, exclude `*.out_proj`, Triton, memory-efficient LoRA | See [Why LoRA](README.md#why-lora-not-full-parameter-sft). Memory-efficient LoRA avoids storing one intermediate tensor. |
| `distributed` | fsdp2, every size 1, activation checkpointing on | One process, no model splitting. Activations are recomputed in the backward pass to save memory. |
| `dataset` | HellaSwag train | See [The dataset](README.md#the-dataset-and-the-sft-prompts). |
| `optimizer` / `lr_scheduler` | AdamW 1e-4, betas 0.9/0.95, weight decay 0.1; 2 warm-up steps, cosine to 1e-5 | A LoRA-sized learning rate with a short warm-up. |
| `checkpoint` | safetensors, `save_consolidated: final` | Writes the adapter at the end of the run. |

## Inside a training step: precision, FSDP2 and memory

**Precision.** The recipe does not override the model dtype, so the base weights load in **BF16**. The recipe uses FP32 for numerically sensitive normalisation (`rms_norm: torch_fp32`), and the log identifies small Mamba-2 parameters stored in FP32 (`NemotronV3MambaFP32Params`).

**Adapter precision.** The recipe does not set `lora_dtype`, so AutoModel creates each LoRA matrix in the dtype of the weight it attaches to (`dtype = lora_dtype or obj.weight.dtype` in [its LoRA code](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/_peft/lora.py)): BF16 here, like the base weights. `torch.optim.AdamW` allocates its moment buffers in the parameters' dtype, so the optimiser state is BF16 too. The loss is still computed in FP32, because `MaskedCrossEntropy` upcasts the logits first. To train the adapter in FP32 instead, add `lora_dtype: torch.float32` to the recipe's `peft` block; we have not measured whether it changes the result. To check a saved adapter, run **inside `lightning-lora`**:

```bash
python3 -c "
from safetensors import safe_open; import collections
f = safe_open('/shared/checkpoints/lightning-200/LOWEST_VAL/model/adapter_model.safetensors', 'pt')
print(collections.Counter(str(f.get_tensor(k).dtype) for k in f.keys()))"
```

Expected output: `Counter({'torch.bfloat16': N})`, N being the number of adapter tensors.

**FSDP2 on one GPU.** Fully Sharded Data Parallel normally divides model weights and optimiser state across GPUs. With a world size of one, AutoModel skips sharding. The recipe retains `strategy: fsdp2` for compatibility with larger configurations, while `ep_size: 1` keeps all experts on the local GPU. The log confirms it (`World size or FSDP mesh size is 1, skipping parallelization`); gradient clipping defaults to a max norm of 1.0.

**One optimiser step, in order.**

1. Each of the 8 micro-batches (one example each) runs forward in BF16 through the 52 layers with **activation checkpointing**: a layer keeps only its input and recomputes the rest in the backward pass, trading compute for memory.
2. The loss is computed on the ending tokens, plus 0.1 × the multi-token-prediction loss from the MTP layer predicting two extra tokens ahead.
3. The backward pass gives gradients to the LoRA weights only; the 8 passes' gradients are accumulated.
4. The combined gradient is clipped to norm 1.0 and AdamW updates the adapter at the scheduled learning rate (table above).

**Memory use.** The 60 GB base weights account for most of the **62 GiB** measured; the adapter is small, and short inputs plus activation checkpointing keep activation memory low. Longer sequences need more (see [Next step](README.md#next-step-sft-for-a-custom-use-case)).
