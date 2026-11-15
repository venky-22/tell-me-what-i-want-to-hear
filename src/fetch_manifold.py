"""Fetch candidate binary questions from Manifold (public API, no key)."""
import json, time, sys, httpx
API = "https://api.manifold.markets/v0/search-markets"
def fetch(filt, sort, pages=12, limit=1000):
    out = []
    for p in range(pages):
        r = httpx.get(API, params=dict(term="", filter=filt, contractType="BINARY", sort=sort,
                                        limit=limit, offset=p*limit), timeout=60)
        if r.status_code == 400: break
        r.raise_for_status(); batch = r.json()
        if not batch: break
        out += batch; time.sleep(0.3)
    return out
if __name__ == "__main__":
    which = sys.argv[1]
    if which == "resolved":
        ms = fetch("resolved", "resolve-date")
    else:
        ms = fetch("open", "most-popular", pages=5)
    keep = ["id","question","resolution","resolutionTime","closeTime","createdTime","uniqueBettorCount",
            "volume","probability","url","creatorUsername","isResolved","outcomeType","groupSlugs"]
    json.dump([{k:m.get(k) for k in keep} for m in ms], open(f"data/manifold_{which}_raw.json","w"))
    print(which, len(ms))
