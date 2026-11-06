"""Decode check for the STaR val set: every row must render to the chat template's native text with the
loss on the assistant turn only, and nothing may exceed the sequence length. Run inside lightning-lora:

    python3 /shared/nemotron-finetuning/experiments/lightning-30b-lora-puzzles-star-dgx-spark/data/decode_check.py

The original experiment's step 8 script, with this folder's val.jsonl and seq_length 8192."""
import json
import sys

from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import ColumnMappedTextInstructionDataset

S = "/shared/nemotron-finetuning/experiments/lightning-30b-lora-puzzles-star-dgx-spark"
SEQ_LENGTH = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
ds = ColumnMappedTextInstructionDataset(
    path_or_dataset_id=f"{S}/data/val.jsonl", column_mapping={"question": "prompt", "answer": "completion"},
    tokenizer=tok, use_hf_chat_template=True, answer_only_loss_mask=True,
    seq_length=SEQ_LENGTH, padding="do_not_pad", truncation="longest_first")
rows = [json.loads(l) for l in open(f"{S}/data/val.jsonl")]
longest = 0
for i, r in enumerate(rows):
    s = ds[i]
    longest = max(longest, len(s["input_ids"]))
    text = tok.decode(s["input_ids"])
    expected = tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}], tokenize=False)
    assert text.rstrip("\n") == expected.rstrip("\n"), (i, text[-200:])
    target = tok.decode([l for l in s["labels"] if l != -100])
    assert target.rstrip("\n").endswith("<|im_end|>"), (i, target[-80:])
    assert "<think>\n" in target and "\\boxed{" in target and "<think>\n " not in text, (i, target[:120])
    assert r["prompt"][:60] not in target, (i, "prompt tokens are in the loss")
assert longest < SEQ_LENGTH, ("a row reaches the sequence length and would be truncated", longest)
s0 = ds[0]
print(len(rows), "rows OK; longest row", longest, "tokens; supervised tokens in row 0:",
      sum(l != -100 for l in s0["labels"]), "of", len(s0["input_ids"]))
print("loss span of row 0 starts:", repr(tok.decode([l for l in s0["labels"] if l != -100])[:70]))
