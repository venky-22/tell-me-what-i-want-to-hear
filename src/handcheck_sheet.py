"""Make a blind hand-check sheet for N questions (default 20).
The rater sees each snippet WITHOUT its label and fills:
  guessed_stance (YES/NO), strength (1-5), leaks_outcome (y/n), notes.
Labels are kept in a separate key file; score with --score.
usage: python src/handcheck_sheet.py [--n 20]   |   python src/handcheck_sheet.py --score
"""
import argparse, csv, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SHEET, KEY = ROOT / "data/handcheck_sheet.csv", ROOT / "data/handcheck_key.json"

def make(n):
    qs = {q["qid"]: q for q in json.load(open(ROOT / "data/questions.json"))}
    pools = json.load(open(ROOT / "data/snippets.json"))
    rng = random.Random(7)
    ids = rng.sample(sorted(pools), n)
    key = {}
    with SHEET.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "qid", "question", "snippet", "guessed_stance", "strength_1to5", "leaks_outcome", "notes"])
        r = 0
        for qid in ids:
            p = pools[qid][:]; rng.shuffle(p)
            for s in p:
                r += 1; key[r] = s["stance"]
                w.writerow([r, qid, qs[qid]["question"], s["text"], "", "", "", ""])
    json.dump(key, open(KEY, "w"))
    print(f"wrote {SHEET} ({r} snippets from {n} questions)")

def score():
    key = json.load(open(KEY)); rows = list(csv.DictReader(SHEET.open()))
    done = [x for x in rows if x["guessed_stance"].strip()]
    acc = sum(x["guessed_stance"].strip().upper() == key[x["row"]] for x in done) / max(1, len(done))
    st = {"YES": [], "NO": []}
    for x in done:
        if x["strength_1to5"].strip(): st[key[x["row"]]].append(float(x["strength_1to5"]))
    leaks = sum(x["leaks_outcome"].strip().lower().startswith("y") for x in done)
    m = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"rated {len(done)} snippets | stance recovered {acc:.1%} | mean strength YES {m(st['YES']):.2f} vs NO {m(st['NO']):.2f} | leak flags {leaks}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--score", action="store_true")
    a = ap.parse_args(); score() if a.score else make(a.n)
