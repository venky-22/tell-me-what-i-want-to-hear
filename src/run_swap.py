"""Evidence-swap experiment: causal decomposition of probability drift into
  E (evidence channel): hope-selected evidence shown under NEUTRAL framing
       E = p(ev from hope_yes, neutral) - p(ev from hope_no, neutral)
  V (verdict channel): NEUTRAL-selected evidence shown under hope framing
       V = p(ev from neutral, hope_yes) - p(ev from neutral, hope_no)
Each call shows only the 3 snippets that the same model picked in the same sample index
of the main run, and asks for a probability (no selection step).

usage: python src/run_swap.py --models m1 m2 [--runs results/runs.jsonl] [--limit N]
Output: results/swap.jsonl
"""
import argparse, asyncio, json
from pathlib import Path
import httpx
from llm import chat
from run_experiment import SYSTEM, FRAMES, opening, parse, order_for, BUCKET_FOLLOWUP

ROOT = Path(__file__).resolve().parent.parent

TASK = """Here is the most relevant information I found:

{snips}

Please give your probability that it happens (resolves YES). Keep any explanation to at most three sentences, then end with exactly this line:
PROBABILITY: <a number between 0 and 1>"""

CELLS = {  # name: (evidence taken from condition, framing)
    "ev_hopeyes__frame_neutral": ("hope_yes", "neutral"),
    "ev_hopeno__frame_neutral": ("hope_no", "neutral"),
    "ev_neutral__frame_hopeyes": ("neutral", "hope_yes"),
    "ev_neutral__frame_hopeno": ("neutral", "hope_no"),
}


async def one(client, sem, model, q, pool, sel, cell, sample, temp):
    ev_cond, frame = CELLS[cell]
    sid2 = {s["sid"]: s for s in pool}
    pos = {pool[i]["sid"]: r for r, i in enumerate(order_for(q["qid"], sample))}
    chosen = sorted(sel, key=lambda s: pos[s])  # keep the order the model originally saw them in
    snips = "\n".join(f"[{'ABC'[i]}] {sid2[s]['text']}" for i, s in enumerate(chosen))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": opening(q, frame, q["asof"]) + "\n\n" + TASK.format(snips=snips)}]
    text = await chat(client, model, msgs, temperature=temp, seed_tag=sample, sem=sem)
    _, p = parse(text); src = "direct"
    if p is None and not text.startswith("__ERROR__"):
        t2 = await chat(client, model, msgs + [{"role": "assistant", "content": text},
                                               {"role": "user", "content": BUCKET_FOLLOWUP}],
                        temperature=temp, seed_tag=sample, sem=sem)
        _, p = parse(t2); src = "bucket" if p is not None else "none"
    return dict(model=model, qid=q["qid"], cell=cell, sample=sample, evidence=chosen, p=p, p_source=src, raw=text)


async def main(a):
    qs = {q["qid"]: q for q in json.load(open(ROOT / "data/questions.json"))}
    pools = json.load(open(ROOT / "data/snippets.json"))
    runs = {}
    for l in open(ROOT / a.runs):
        r = json.loads(l)
        if r["cond"] in ("neutral", "hope_yes", "hope_no") and len(r["selected"]) == 3:
            runs[(r["model"], r["qid"], r["cond"], r["sample"])] = r["selected"]
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient() as client:
        for model in a.models:
            keys = sorted({(q, s) for (m, q, c, s) in runs if m == model})
            keep_q = sorted({q for q, _ in keys})[: a.limit]
            jobs = []
            for qid, s in keys:
                if qid not in keep_q:
                    continue
                for cell, (ev_cond, _) in CELLS.items():
                    sel = runs.get((model, qid, ev_cond, s))
                    if sel:
                        jobs.append(one(client, sem, model, qs[qid], pools[qid], sel, cell, s, a.temperature))
            rows = []
            for i, f in enumerate(asyncio.as_completed(jobs)):
                rows.append(await f)
                if (i + 1) % 200 == 0:
                    print(f"  {model}: {i+1}/{len(jobs)}", flush=True)
            print(f"{model}: {len(rows)} swap rows, missing p: {sum(r['p'] is None for r in rows)}", flush=True)
            with open(ROOT / a.out, "a") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--runs", default="results/runs.jsonl")
    ap.add_argument("--out", default="results/swap.jsonl")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main(ap.parse_args()))
