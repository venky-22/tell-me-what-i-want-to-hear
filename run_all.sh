#!/bin/bash
# Sequential pipeline (one process at a time, so the 80 req/min limiter is never shared).
set -u
cd "$(dirname "$0")/src"
MODELS=(gemma-4-31b-it "qwen3.5-397b@noreason" gemma-4-26b-a4b-it)
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a ../results/pipeline.log; }

log "preflight"
python3 - <<'PY' || { log "PREFLIGHT FAILED"; exit 1; }
import asyncio, httpx, sys
from llm import _call
async def main():
    sem = asyncio.Semaphore(1)
    async with httpx.AsyncClient() as c:
        for m in ["gemma-4-31b-it", "qwen3.5-397b@noreason", "gemma-4-26b-a4b-it"]:
            t, u = await _call(c, m, [{"role": "user", "content": "Reply with OK"}], 0, 20, sem)
            print(m, repr(t[:40])); assert not t.startswith("__ERROR__"), t
asyncio.run(main())
PY

stage() { log "stage $1 start"; python3 run_experiment.py --models "${MODELS[@]}" --conds "$1" --out results/runs.jsonl --concurrency 6 > "../results/stage_$1.log" 2>&1; grep -v "^  " "../results/stage_$1.log" | tee -a ../results/pipeline.log; }
stage core
stage stakes_yes,stakes_no
stage friend_yes,friend_no
stage instr_yes,instr_no
stage twopass_yes,twopass_no
log "fill pass (cached calls are free)"
stage all,stakes_yes,stakes_no
log "swap start"
python3 run_swap.py --models "${MODELS[@]}" --concurrency 6 --out results/swap.jsonl > ../results/stage_swap.log 2>&1; grep -v "^  " ../results/stage_swap.log | tee -a ../results/pipeline.log
log "swap fill"
python3 run_swap.py --models "${MODELS[@]}" --concurrency 6 --out results/swap.jsonl > ../results/stage_swapfill.log 2>&1; grep -v "^  " ../results/stage_swapfill.log | tee -a ../results/pipeline.log
log "reasoning subset (qwen thinking on, 30 questions, core)"
python3 run_experiment.py --models qwen3.5-397b --conds core --qids data/reasoning_qids.txt --out results/runs_reasoning.jsonl --concurrency 6 > ../results/stage_reasoning.log 2>&1; grep -v "^  " ../results/stage_reasoning.log | tee -a ../results/pipeline.log
python3 usage.py | tee -a ../results/pipeline.log
log "DONE"
