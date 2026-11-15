"""Token usage per model from the call cache (calls made before usage logging show as 'unlogged')."""
import json, collections
from pathlib import Path
c = collections.defaultdict(lambda: collections.Counter())
for l in open(Path(__file__).resolve().parent.parent / "results/cache.jsonl"):
    r = json.loads(l); u = r.get("usage"); m = r["model"]
    c[m]["calls"] += 1
    if u:
        c[m]["prompt"] += u.get("prompt_tokens", 0); c[m]["completion"] += u.get("completion_tokens", 0)
        c[m]["reasoning"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    else:
        c[m]["unlogged"] += 1
for m, v in c.items():
    print(f"{m:22s} calls={v['calls']:6d} prompt_tok={v['prompt']:>10,} completion_tok={v['completion']:>9,} (reasoning {v['reasoning']:,}) unlogged={v['unlogged']}")
