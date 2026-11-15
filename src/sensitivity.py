"""Robustness analyses reported in the paper, rebuilt from the committed raw outputs.

  excl   drop the two resolved questions whose as-of date fell after their own deadline
  nodeg  drop calls where a model fell into a word-repetition loop
  think  qwen3.5-397b thinking on (runs_reasoning.jsonl) vs thinking off on the same 30 questions

usage: python src/sensitivity.py   -> results/*_{excl,nodeg,think}.csv and figs/*_{excl,nodeg,think}.png
"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
DATE_LEAK = {"RLhR5uSAE", "REZdpLIIg"}  # "GPT-6 by Aug 31" (as-of Sep 7), "shutdown on Oct 1st" (as-of Oct 4)
LOOP = re.compile(r"(\b\w+\b)(?:\s+\1\b){15,}")


def write(path, rows):
    with open(path, "w") as f:
        for l in rows:
            f.write(l)


def lines(name):
    return list(open(R / name))


def analyze(runs, swap, tag):
    subprocess.run([sys.executable, str(ROOT / "src/analyze.py"), "--runs", str(runs), "--swap", str(swap), "--tag", tag],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"{tag}: done")


runs, swap = lines("runs.jsonl"), lines("swap.jsonl")
write(R / "runs_excl.jsonl", [l for l in runs if json.loads(l)["qid"] not in DATE_LEAK])
write(R / "swap_excl.jsonl", [l for l in swap if json.loads(l)["qid"] not in DATE_LEAK])
analyze(R / "runs_excl.jsonl", R / "swap_excl.jsonl", "excl")

write(R / "runs_nodeg.jsonl", [l for l in runs if not LOOP.search(json.loads(l)["raw"][-600:])])
analyze(R / "runs_nodeg.jsonl", "none", "nodeg")

qids = set(open(ROOT / "data/reasoning_qids.txt").read().split())
off = [l for l in runs if (r := json.loads(l))["qid"] in qids and r["cond"] in ("neutral", "hope_yes", "hope_no")
       and r["model"] == "qwen3.5-397b@noreason"]
write(R / "runs_think_cmp.jsonl", lines("runs_reasoning.jsonl") + off)
analyze(R / "runs_think_cmp.jsonl", "none", "think")
