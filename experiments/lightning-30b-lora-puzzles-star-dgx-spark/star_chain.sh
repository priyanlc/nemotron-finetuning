#!/usr/bin/env bash
# star_chain.sh: steps 4 to 9 of the README, unattended. Waits for the sampler's summary file, builds and
# checks the STaR set, writes the recipe config, swaps vLLM for AutoModel, runs the decode check, the
# 20-step smoke run and the 200-step run, serves the base with both adapters, and scores the 300 puzzles
# twice. Every stage appends one line to logs/chain.status; the first gate that fails stops the chain.
#
# Run on the Spark host, detached:
#   S=~/nemo-shared/nemotron-finetuning/experiments/lightning-30b-lora-puzzles-star-dgx-spark
#   nohup setsid bash $S/star_chain.sh > $S/logs/chain.log 2>&1 &
#   tail -f $S/logs/chain.status
# SKIP_TO=train restarts at step 7 once a smoke run has been judged by hand (the built set, config and
# decode check are reused); MEM_LIMIT (GiB) and MAX_MODEL_LEN can be overridden the same way.
#
# Gates: kept traces >= MIN_KEPT; check passes; config has two seq_length lines and no HellaSwag; decode
# check passes; smoke run finishes 20 steps with finite loss and peak mem under MEM_LIMIT GiB; the 200-step
# run leaves a LOWEST_VAL adapter; the server lists all three model names; both scoring runs write a .json.
set -u

H=$HOME/nemo-shared/nemotron-finetuning                       # host path of the tree
C=/shared/nemotron-finetuning                                 # the same tree inside the containers
S=$H/experiments/lightning-30b-lora-puzzles-star-dgx-spark
SC=$C/experiments/lightning-30b-lora-puzzles-star-dgx-spark
EC=$C/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark
TC=$C/tools
STATUS=$S/logs/chain.status
MIN_KEPT=${MIN_KEPT:-200}
MEM_LIMIT=${MEM_LIMIT:-85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
SAMPLER_STALL_S=7200
SKIP_TO=${SKIP_TO:-}

say()  { echo "$(date +'%Y-%m-%d %H:%M:%S') $*" | tee -a "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
inlora() { docker exec lightning-lora bash -c "$*"; }
wait_memory() {                                                # wait until the training/serving process has let go
  for _ in $(seq 1 60); do
    avail=$(free -g | awk '/^Mem:/ {print $7}')
    [ "$avail" -ge 80 ] && return 0
    sleep 10
  done
  fail "$1: host memory still in use (available ${avail} GB)"
}

mkdir -p "$S/logs" "$S/checkpoints" "$S/results"
say "START chain pid $$${SKIP_TO:+ (skip to $SKIP_TO, mem limit $MEM_LIMIT)}"

if [ "$SKIP_TO" != "train" ]; then
# ---- step 3, wait for the sampler ---------------------------------------------------------------------
JSON=$S/logs/star-cipher-7680.json
JSONL=$S/logs/star-cipher-7680.jsonl
until [ -f "$JSON" ]; do
  if [ -f "$JSONL" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$JSONL") ))
    [ "$age" -gt "$SAMPLER_STALL_S" ] && fail "sampler stalled: no new generation for ${age}s and no summary file"
  fi
  sleep 300
done
say "SAMPLED $(python3 -c "
import json; d = json.load(open('$JSON'))
print('n', d['n'], 'finished', d['finished'], 'first_box_ok', d['first_box_ok'], 'truncated', d['truncated'], 'seconds', int(d['seconds']))")"

# ---- step 4, build and check ------------------------------------------------------------------------------
inlora "python3 $SC/data/make_star_jsonl.py build --samples $SC/logs/star-cipher-7680.jsonl" > "$S/logs/chain-build.log" 2>&1 \
  || fail "build (see logs/chain-build.log)"
kept=$(python3 -c "import json; print(json.load(open('$S/results/star-dataset.json'))['kept'])")
[ "$kept" -ge "$MIN_KEPT" ] || fail "build kept only $kept traces (minimum $MIN_KEPT)"
inlora "python3 $SC/data/make_star_jsonl.py check" > "$S/logs/chain-check.log" 2>&1 || fail "check (see logs/chain-check.log)"
say "BUILT $(python3 -c "
import json; d = json.load(open('$S/results/star-dataset.json'))
print('kept', d['kept'], 'train_traces', d['train_traces'], 'repeats', d['repeats'], 'cipher_rows', d['cipher_rows'], 'train_rows', d['train_rows'], 'val_rows', d['val_rows'], 'dropped', d['dropped'])")"

# ---- step 5, config ---------------------------------------------------------------------------------------
E=$H/experiments/lightning-30b-lora-reasoning-puzzles-dgx-spark
cp "$E/configs/nemotron_nano_v3_5_lightning_singlegpu_lora.yaml" "$S/configs/star_lora.yaml"
sed -i '/^dataset:$/,/^  split: train$/d' "$S/configs/star_lora.yaml"
cat "$S/configs/star-dataset-block.yaml" >> "$S/configs/star_lora.yaml"
[ "$(grep -c hellaswag "$S/configs/star_lora.yaml")" -eq 0 ] || fail "config still mentions hellaswag"
[ "$(grep -c '^  seq_length: 8192$' "$S/configs/star_lora.yaml")" -eq 2 ] || fail "config does not have two seq_length: 8192 lines"
[ "$(grep -c "puzzles-star-dgx-spark/data/" "$S/configs/star_lora.yaml")" -eq 2 ] || fail "config does not point at this folder's data"
say "CONFIG configs/star_lora.yaml written"

# ---- step 6, free the GPU, decode check, smoke run -------------------------------------------------------
if docker ps --format '{{.Names}}' | grep -qx vllm-lightning; then
  docker stop vllm-lightning > /dev/null || fail "docker stop vllm-lightning"
fi
docker rm vllm-lightning > /dev/null 2>&1 || true                  # the container was started without --rm
wait_memory "after stopping vLLM"
say "GPU_FREE vllm-lightning stopped and removed"

inlora "python3 $SC/data/decode_check.py 8192" > "$S/logs/chain-decode.log" 2>&1 || fail "decode check (see logs/chain-decode.log)"
say "DECODE $(tail -n 2 "$S/logs/chain-decode.log" | head -n 1)"

SMOKE=star-test-$(date +%m%d-%H%M)
run_automodel "$SMOKE" --step_scheduler.val_every_steps 10 || fail "smoke run exited non-zero (see logs/$SMOKE.log)"
summary=$(inspect_run "$SMOKE" 20) || fail "smoke run: $summary"
[ -f "$S/checkpoints/$SMOKE/epoch_0_step_19/model/adapter_model.safetensors" ] || fail "smoke run left no adapter"
say "SMOKE $SMOKE $summary"
fi

run_automodel() {                                                # $1 run name, $2.. extra automodel args
  local run=$1; shift
  [ -e "$S/checkpoints/$run" ] && fail "checkpoint folder $run exists"
  inlora "cd /opt/Automodel && CUDA_VISIBLE_DEVICES=0 automodel $SC/configs/star_lora.yaml --nproc-per-node 1 $* \
    --checkpoint.checkpoint_dir $SC/checkpoints/$run" > "$S/logs/$run.log" 2>&1
  return $?
}
inspect_run() {                                                  # $1 run name, $2 expected steps: prints a summary or exits 1
  python3 - "$S/checkpoints/$1/training.jsonl" "$2" "$MEM_LIMIT" "$S/logs/$1.log" <<'EOF'
import json, math, sys
from datetime import datetime
rows = [json.loads(l) for l in open(sys.argv[1])]
steps, limit = int(sys.argv[2]), float(sys.argv[3])
problems = []
if len(rows) < steps: problems.append(f"only {len(rows)} of {steps} step records")
if any(not math.isfinite(r["loss"]) for r in rows): problems.append("non-finite loss")
if any(not math.isfinite(r.get("grad_norm", 0.0)) for r in rows): problems.append("non-finite grad_norm")
peak = max(r["mem"] for r in rows)
if peak > limit: problems.append(f"peak mem {peak:.1f} GiB over {limit}")
t = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) for r in rows]
sps = (t[-1] - t[0]).total_seconds() / max(1, len(rows) - 1)
first, last = rows[0]["loss"], sum(r["loss"] for r in rows[-5:]) / len(rows[-5:])
print(f"steps {len(rows)} loss {first:.3f}->{last:.3f} peak_mem {peak:.1f}GiB sec/step {sps:.1f}")
if problems:
    print(" ; ".join(problems)); sys.exit(1)
EOF
}

# ---- step 7, 200 steps ------------------------------------------------------------------------------------
RUN=star-200-$(date +%m%d-%H%M)
run_automodel "$RUN" --step_scheduler.max_steps 200 --step_scheduler.val_every_steps 20 --step_scheduler.ckpt_every_steps 20 \
  || fail "200-step run exited non-zero (see logs/$RUN.log)"
summary=$(inspect_run "$RUN" 200) || fail "200-step run: $summary"
ln -sfn "$RUN" "$S/checkpoints/star-200"
ln -sf "$RUN.log" "$S/logs/star-200.log"
LOWEST=$(readlink "$S/checkpoints/star-200/LOWEST_VAL") || fail "no LOWEST_VAL link"
[ -f "$S/checkpoints/star-200/LOWEST_VAL/model/adapter_model.safetensors" ] || fail "LOWEST_VAL has no adapter"
python3 -c "
import json, csv, sys
rows = [json.loads(l) for l in open('$S/checkpoints/star-200/validation.jsonl')]
w = csv.writer(open('$S/results/validation-loss.csv', 'w')); w.writerow(['step', 'val_loss', 'mem'])
for r in rows: w.writerow([r['step'], round(r['val_loss'], 4), round(r['mem'], 1)])
print(' '.join(f\"{r['step']}:{r['val_loss']:.3f}\" for r in rows))" > "$S/logs/chain-val.txt"
say "TRAINED $RUN lowest_val $LOWEST $summary val $(cat "$S/logs/chain-val.txt")"
wait_memory "after training"

# ---- step 8, serve base + both adapters -------------------------------------------------------------------
docker run -d --name vllm-lightning \
  --gpus all --ipc=host --network host \
  -v "$HOME/nemo-shared:/shared" \
  -e HF_HOME=/shared/.cache/huggingface \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.27.1 -c "vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --served-model-name lightning-base --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.80 \
    --mamba-backend flashinfer --enable-lora --max-lora-rank 8 \
    --lora-modules lightning-puzzles=$EC/checkpoints/puzzles-200/LOWEST_VAL/model \
                   lightning-star=$SC/checkpoints/star-200/LOWEST_VAL/model \
    2>&1 | tee $SC/logs/vllm-lightning-star.log" > /dev/null || fail "docker run vllm-lightning"
for i in $(seq 1 180); do                                        # up to 30 minutes for the server to load
  models=$(curl -s localhost:8000/v1/models | python3 -c "import json,sys; print(' '.join(sorted(m['id'] for m in json.load(sys.stdin)['data'])))" 2>/dev/null)
  [ "$models" = "lightning-base lightning-puzzles lightning-star" ] && break
  docker ps --format '{{.Names}}' | grep -qx vllm-lightning || fail "vllm-lightning exited while loading (see logs/vllm-lightning-star.log)"
  sleep 10
done
[ "$models" = "lightning-base lightning-puzzles lightning-star" ] || fail "server did not list the three models within 30 minutes (got: $models)"
say "SERVED $models, max-model-len $MAX_MODEL_LEN"

# ---- step 9, score 300 puzzles twice ----------------------------------------------------------------------
score() {                                                        # $1 base name, $2 adapter name, $3 output stem
  inlora "python3 $TC/compare_puzzles.py --n 300 --concurrency 32 --base-name $1 --adapter-name $2 --out $SC/logs/$3.csv" \
    > "$S/logs/$3.log" 2>&1 || { echo "$3 exited non-zero (see logs/$3.log)"; return 1; }
  [ -f "$S/logs/$3.json" ] || { echo "$3 wrote no .json summary"; return 1; }
  python3 -c "
import json; d = json.load(open('$S/logs/$3.json'))
pc = ' '.join(f\"{k}:{v['base']}/{v['adapter']}\" for k, v in sorted(d['per_category'].items()))
print(f\"$1 {d['base_acc']:.3f} $2 {d['adapter_acc']:.3f} gain {d['gain']:+.3f} fixed {d['fixed']} broke {d['broke']} p {d['mcnemar_p']:.2g} truncated {d['truncated']} {int(d['seconds'])}s | {pc}\")"
}
out=$(score lightning-puzzles lightning-star compare-300-v1-vs-star) || fail "$out"
say "SCORED1 $out"
out=$(score lightning-base lightning-star compare-300-star) || fail "$out"
say "SCORED2 $out"
say "DONE"
