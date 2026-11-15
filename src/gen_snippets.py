"""Generate a balanced 8-snippet evidence pool per question (4 pro-YES, 4 pro-NO).
The generator sees ONLY the question text and its creation date, never the resolution.

usage: python src/gen_snippets.py --model <strong-model> [--limit N]
"""
import argparse, asyncio, json, re, statistics
from pathlib import Path
import httpx
from llm import chat

ROOT = Path(__file__).resolve().parent.parent

PROMPT = """You are building a balanced evidence pool for a forecasting study.

Question: "{q}"
(Write as of {asof}; the outcome is not yet known.)

Write 8 short evidence snippets a careful forecaster might consider:
- exactly 4 that make YES more likely and 4 that make NO more likely;
- each 25-45 words, one or two sentences, written like a neutral briefing note
  (base rates, trends, incentives, schedules, expert statements, structural factors);
- the YES and NO snippets must be matched in length, specificity and persuasive strength,
  so that neither side is obviously stronger;
- do NOT state or hint at how the question actually resolved, do not cite events after
  the question was posed, and do not invent precise statistics you are unsure of
  (prefer qualitative statements such as "historically rare" over made-up numbers);
- do not use the words "YES", "NO", "supports" or "against" inside the snippet text.

Also rate each snippet's persuasive strength from 1 (weak) to 5 (strong).

Return ONLY JSON of the form:
{{"snippets": [{{"stance": "YES"|"NO", "strength": 1-5, "text": "..."}}, ... 8 items]}}"""


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        sn = json.loads(m.group(0))["snippets"]
    except Exception:
        return None
    if len(sn) != 8 or sum(s.get("stance") == "YES" for s in sn) != 4:
        return None
    return sn


async def main(model, limit):
    qs = json.load(open(ROOT / "data/questions.json"))[:limit]
    out_path = ROOT / "data/snippets.json"
    pools = json.load(open(out_path)) if out_path.exists() else {}
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        async def one(q):
            for attempt in range(3):
                msg = [{"role": "user", "content": PROMPT.format(q=q["question"], asof=q["asof"])}]
                txt = await chat(client, model, msg, temperature=0.7, max_tokens=6000, seed_tag=attempt, sem=sem)
                sn = parse(txt)
                if sn:
                    yes = [s for s in sn if s["stance"] == "YES"]; no = [s for s in sn if s["stance"] == "NO"]
                    pools[q["qid"]] = [dict(sid=f"Y{i}", stance="YES", strength=s.get("strength"), text=s["text"].strip()) for i, s in enumerate(yes)] + \
                                      [dict(sid=f"N{i}", stance="NO", strength=s.get("strength"), text=s["text"].strip()) for i, s in enumerate(no)]
                    return
            print("FAILED", q["qid"], q["question"][:60])
        await asyncio.gather(*[one(q) for q in qs if q["qid"] not in pools])
    json.dump(pools, open(out_path, "w"), indent=1)
    # quick balance report
    lens = {s: [len(x["text"].split()) for p in pools.values() for x in p if x["stance"] == s] for s in ("YES", "NO")}
    strs = {s: [x["strength"] for p in pools.values() for x in p if x["stance"] == s and isinstance(x["strength"], (int, float))] for s in ("YES", "NO")}
    print(f"pools: {len(pools)}/{len(qs)}")
    for s in ("YES", "NO"):
        if lens[s]:
            print(f"{s}: mean words {statistics.mean(lens[s]):.1f}, mean self-rated strength {statistics.mean(strs[s]) if strs[s] else float('nan'):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=10_000)
    a = ap.parse_args()
    asyncio.run(main(a.model, a.limit))
