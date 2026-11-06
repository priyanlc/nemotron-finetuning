# Rescoring at a higher cap: does the adapter's gain survive when the base can finish?

*A follow-up to the [experiment README](README.md); it starts from that guide's step 12 output and leaves steps 1 to 13 and their files untouched. New files only: `logs/compare-300-cap*.{csv,json,-generations.jsonl}` and `results/cap-sweep.json`.*

The base model hit the 7,680-token cap on 218 of the 300 puzzles and a generation with no box scores zero, so part of the adapter's gain is the base running out of room rather than reasoning worse. This follow-up measures how much of the gain survives when the base can finish, the check a reader of the write-up asked for. It re-runs all 300 puzzles for both models at each larger cap and merges the runs with [`tools/cap_sweep.py`](../../tools/cap_sweep.py). Method and options: [tools/README.md](../../tools/README.md#cap_sweeppy).

**Why all 300 and not just the 218 that hit the cap.** The first draft of this follow-up re-ran only the truncated puzzles, on the assumption that under greedy decoding a puzzle that finished at 7,680 tokens gives the same output at a larger cap. The 12-puzzle smoke test on 2026-11-04 showed that it does not: two puzzles the base had solved in 4,080 and 6,911 tokens in October ran past 16,384 tokens, one it had solved in 7,058 finished in 4,621, and even short numeral answers shifted by dozens of tokens. vLLM batches requests differently from run to run, bf16 arithmetic shifts slightly with the batch, and on a mixture-of-experts model a small shift can flip an expert choice and send the trace down another path. Re-running only the base's failures would re-roll its losses and never its wins, which biases the result in its favour. Re-running everything costs little more, because the truncated puzzles dominate the time anyway. The same drift means the October run's base-versus-adapter comparison carries run-to-run noise its p-value does not include; a repeat of the original 300 at 7,680 (same command as step 12, new output name) measures that noise floor in about 3.5 hours and is worth doing if there is time.

**First, the offline check, before any GPU time.** It reads the 7,680 run's generations and reports, per category, how many truncated base generations had no box, a correct last box, a correct box anywhere, or a correct first box. If most truncated cipher generations already contain the right answer, the cap hid it; if most have no box, the base was not getting there.

```bash
E=/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark
docker exec -it lightning-lora python3 /shared/nemotron-finetuning/tools/cap_sweep.py $E/logs/compare-300.csv
```

The table's single row must reproduce the README's [Results](README.md#results) (0.290 / 0.550, 218 truncated) before you go on.

**Then restart the server with a larger window.** Stop [step 10](README.md#step-10-free-the-gpu-and-serve-base-and-adapter)'s `vllm serve` with `Ctrl-C` and start it again with `--max-model-len 32768` and everything else unchanged. The model supports far longer contexts; the window only sizes the KV cache. In the startup log, find the line that reports the maximum concurrency at 32,768 tokens per request; if it is below 32, use that number for `--concurrency` below.

**Smoke test, then the two runs.** In the second terminal, inside `lightning-lora`. The runs take hours, so start them detached (`docker exec -d`) and follow the `.log` file; `tee` and the output files are written inside the container because `logs/` belongs to root after the earlier docker runs.

```bash
E=/shared/nemotron-finetuning/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark
T=/shared/nemotron-finetuning/tools
docker exec -it lightning-lora python3 $T/compare_puzzles.py --n 12 --max-tokens 16384 --out $E/logs/compare-12-cap16k.csv
docker exec -d lightning-lora bash -c "python3 $T/compare_puzzles.py --n 300 --max-tokens 16384 --concurrency 32 \
  --out $E/logs/compare-300-cap16k.csv > $E/logs/compare-300-cap16k.log 2>&1"
docker exec -d lightning-lora bash -c "python3 $T/compare_puzzles.py --n 300 --max-tokens 32768 --concurrency 32 \
  --out $E/logs/compare-300-cap32k.csv > $E/logs/compare-300-cap32k.log 2>&1"
```

The smoke run should print a server window of 32,768 and a cap of 16,384, and its generations file should show `"budget": 16384` with no `length` finish for the adapter (measured 2026-11-04: 32 minutes, 6 of 12 base generations still at the cap). `--n 300` selects the same 300 puzzles as step 12, so each run is a full re-score of that set. Time: at `--concurrency 32` the server generates about 170 tokens per second in aggregate, so the 16k run is about 6 hours if most puzzles go to the cap, and the 32k run up to twice that; start the second only after reading the first's truncation count. If a meaningful share is still truncated at 32k, one more round at 65,536 follows the same pattern; otherwise stop.

**Merge and record:**

```bash
docker exec -it lightning-lora python3 $T/cap_sweep.py $E/logs/compare-300.csv $E/logs/compare-300-cap16k.csv \
  $E/logs/compare-300-cap32k.csv --out $E/results/cap-sweep.json
```

The sweep table has one row per cap: base and adapter accuracy, the gain with its interval and p-value, how many base generations are still truncated, and how many answers each model gained and lost against the previous cap. The 7,680 row equals the README's Results. Because of the run-to-run drift described above, base accuracy can fall as well as rise between caps; the gained and lost columns show how much of each movement is churn rather than a trend, and the 7,680 repeat, if made, gives the churn at a constant cap. Commit `results/cap-sweep.json`, say in `results/README.md` how it was produced, and fill the numbers into the README's Results. The sentence the write-up needs is how many of the +0.260 survive once the base has room to finish, with cipher and bit manipulation reported separately, since cipher is where the cap most plausibly hid correct answers.
