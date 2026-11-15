"""Automatic checks on the snippet pools: structure, length/strength balance, outcome-leak keywords."""
import json, re, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
qs = {q["qid"]: q for q in json.load(open(ROOT / "data/questions.json"))}
pools = json.load(open(ROOT / "data/snippets.json"))
LEAK = re.compile(r"\b(resolved|has won|won the|was elected|were elected|lost the|announced that it had|ultimately|in the end|final result|results show)\b", re.I)
missing = [k for k in qs if k not in pools]
bad = [k for k, p in pools.items() if len(p) != 8 or sum(s["stance"] == "YES" for s in p) != 4]
lens = {s: [len(x["text"].split()) for p in pools.values() for x in p if x["stance"] == s] for s in ("YES", "NO")}
strs = {s: [x["strength"] for p in pools.values() for x in p if x["stance"] == s and isinstance(x["strength"], (int, float))] for s in ("YES", "NO")}
dlen = [st.mean(len(x["text"].split()) for x in p if x["stance"] == "YES") - st.mean(len(x["text"].split()) for x in p if x["stance"] == "NO") for p in pools.values()]
dstr = [st.mean(x["strength"] for x in p if x["stance"] == "YES") - st.mean(x["strength"] for x in p if x["stance"] == "NO") for p in pools.values()]
print(f"pools {len(pools)}/{len(qs)}; missing {missing}; malformed {bad}")
for s in ("YES", "NO"):
    print(f"{s}: words {st.mean(lens[s]):.1f}±{st.stdev(lens[s]):.1f}, self-rated strength {st.mean(strs[s]):.2f}")
print(f"per-question YES−NO: words mean {st.mean(dlen):+.2f} (max |.| {max(map(abs, dlen)):.1f}); strength mean {st.mean(dstr):+.3f} (max |.| {max(map(abs, dstr)):.2f})")
yrs = re.compile(r"\b(202[7-9]|20[3-9]\d)\b")
for k, p in pools.items():
    for x in p:
        if LEAK.search(x["text"]):
            print(f"  LEAK? {k} [{x['stance']}] {qs[k]['question'][:50]} :: {x['text']}")
