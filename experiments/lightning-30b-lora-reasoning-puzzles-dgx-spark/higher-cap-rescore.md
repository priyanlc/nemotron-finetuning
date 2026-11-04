# Rescoring at a higher cap: does the adapter's gain survive when the base can finish?

*A follow-up to the [experiment README](README.md); it starts from that guide's step 12 output and leaves steps 1 to 13 and their files untouched. New files only: `logs/compare-300-cap*.{csv,json,-generations.jsonl}` and `results/cap-sweep.json`.*

The base model hit the 7,680-token cap on 218 of the 300 puzzles and a generation with no box scores zero, so part of the adapter's gain is the base running out of room rather than reasoning worse. This follow-up measures how much of the gain survives when the base can finish, the check a reader of the write-up asked for. It re-runs only the puzzles that hit the cap: under greedy decoding a puzzle that finished at 7,680 tokens gives the same output at a larger cap, and the adapter never truncated (its longest generation was 105 tokens). The adapter is still re-run on the same puzzles, at little cost, so the statistics stay paired. A higher-cap generation is a fresh request, not a continuation of the earlier text, because vLLM's batching can change a greedy prefix; [`tools/cap_sweep.py`](../../tools/cap_sweep.py) therefore replaces each re-run row whole. Method and options: [tools/README.md](../../tools/README.md#cap_sweeppy).

**First, the offline check, before any GPU time.** It reads the 7,680 run's generations and reports, per category, how many truncated base generations had no box, a correct last box, a correct box anywhere, or a correct first box. If most truncated cipher generations already contain the right answer, the cap hid it; if most have no box, the base was not getting there.

```bash
E=/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark
docker exec -it lightning-lora python3 /shared/nemotron-finetuning/tools/cap_sweep.py $E/logs/compare-300.csv
```

The table's single row must reproduce the README's [Results](README.md#results) (0.290 / 0.550, 218 truncated) before you go on.

**Then restart the server with a larger window.** Stop [step 10](README.md#step-10-free-the-gpu-and-serve-base-and-adapter)'s `vllm serve` with `Ctrl-C` and start it again with `--max-model-len 32768` and everything else unchanged. The model supports far longer contexts; the window only sizes the KV cache. In the startup log, find the line that reports the maximum concurrency at 32,768 tokens per request; if it is below 32, use that number for `--concurrency` below.

**Smoke test, then the two runs.** In the second terminal, inside `lightning-lora`:

```bash
E=/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark
T=/shared/nemotron-finetuning/tools
docker exec -it lightning-lora python3 $T/compare_puzzles.py --n 12 --max-tokens 16384 --out $E/logs/compare-12-cap16k.csv
docker exec -it lightning-lora bash -c "python3 $T/compare_puzzles.py --ids-from $E/logs/compare-300.csv --only-truncated \
  --max-tokens 16384 --concurrency 32 --out $E/logs/compare-300-cap16k.csv 2>&1 | tee $E/logs/compare-300-cap16k.log"
docker exec -it lightning-lora bash -c "python3 $T/compare_puzzles.py --ids-from $E/logs/compare-300-cap16k.csv --only-truncated \
  --max-tokens 32768 --concurrency 32 --out $E/logs/compare-300-cap32k.csv 2>&1 | tee $E/logs/compare-300-cap32k.log"
```

The smoke run should print a server window of 32,768 and a cap of 16,384, and its generations file should show `"budget": 16384` with no `length` finish for the adapter. The 16k run scores the 218 truncated puzzles; the 32k run scores whatever is still truncated after that. Time: the 7,680 run moved about 190 tokens per second in aggregate, so each run is at most about 5 hours if every puzzle goes to its cap, and the second is usually much shorter. If a meaningful share is still truncated at 32k, one more round at 65,536 on that remainder follows the same pattern; otherwise stop.

**Merge and record:**

```bash
docker exec -it lightning-lora python3 $T/cap_sweep.py $E/logs/compare-300.csv $E/logs/compare-300-cap16k.csv \
  $E/logs/compare-300-cap32k.csv --out $E/results/cap-sweep.json
```

The sweep table has one row per cap: base and adapter accuracy, the gain with its interval and p-value, and how many base generations are still truncated. The 7,680 row equals the README's Results, and base accuracy cannot fall as the cap rises except by batching noise, which the warnings surface. Commit `results/cap-sweep.json`, say in `results/README.md` how it was produced, and fill the numbers into the README's Results. The sentence the write-up needs is how many of the +0.260 survive once the base has room to finish, with cipher and bit manipulation reported separately, since cipher is where the cap most plausibly hid correct answers.
