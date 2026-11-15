"""Compute metrics and figures from results/runs.jsonl.

Layer 1  D = mean p(hope_yes) - mean p(hope_no), per question; bootstrap CI over questions.
Layer 2  K = mean #YES-snippets selected (hope_yes) - same (hope_no), range -3..+3.
Link     corr(K_q, D_q); and a fixed-effects mediation: p ~ question FE + hope (+ nYES selected).
Fixes    same D and K for friend_* (control), instr_* (fix A) and twopass_* (fix B).
Brier    resolved questions only: neutral vs aligned-hope vs opposite-hope ("harm" cell).

usage: python src/analyze.py [--runs results/runs.jsonl] [--tag pilot]
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(0)
B = 5000

# palette (dataviz reference palette, light mode, for a static PDF)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # slots 1-3: validated all-pairs
DIVERGE = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#6da7ec", "#f0efec", "#ec8a7f", "#a8262a"])
PAIRS = {"user hope": ("hope_yes", "hope_no"), "high-stakes hope": ("stakes_yes", "stakes_no"), "high-stakes (non-financial)": ("stakes2_yes", "stakes2_no"), "third-party hope": ("friend_yes", "friend_no"),
         "fix A: instruction": ("instr_yes", "instr_no"), "fix B: two-pass": ("twopass_yes", "twopass_no")}


def boot_ci(x, f=np.mean):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    bs = [f(x[RNG.integers(0, len(x), len(x))]) for _ in range(B)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def load(path):
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame(rows)
    pools = json.load(open(ROOT / "data/snippets.json"))
    qs = {q["qid"]: q for q in json.load(open(ROOT / "data/questions.json"))}
    stance = {(qid, s["sid"]): s["stance"] for qid, p in pools.items() for s in p}
    df["nyes"] = [sum(stance.get((q, s)) == "YES" for s in sel) if len(sel) == 3 else np.nan
                  for q, sel in zip(df.qid, df.selected)]
    df["resolution"] = df.qid.map(lambda q: qs[q]["resolution"])
    return df.drop_duplicates(["model", "qid", "cond", "sample"], keep="last"), qs


def per_q(df):
    return df.groupby(["model", "qid", "cond"]).agg(p=("p", "mean"), nyes=("nyes", "mean"),
                                                    p_sd=("p", "std")).reset_index()


def layer_stats(g):
    out = []
    piv_p = g.pivot_table(index=["model", "qid"], columns="cond", values="p")
    piv_k = g.pivot_table(index=["model", "qid"], columns="cond", values="nyes")
    sd_neu = g[g.cond == "neutral"].set_index(["model", "qid"]).p_sd
    for model in g.model.unique():
        for name, (a, b) in PAIRS.items():
            if a not in piv_p or b not in piv_p:
                continue
            D = (piv_p[a] - piv_p[b]).loc[model].dropna()
            K = (piv_k[a] - piv_k[b]).loc[model].dropna() if a in piv_k and b in piv_k else pd.Series(dtype=float)
            if a.startswith("twopass"):  # selection is the hope-blind pass-1 selection by construction
                K = pd.Series(0.0, index=D.index)
            # noise floor: SE of a 3-vs-3 sample-mean difference implied by neutral within-question SD
            nf = (sd_neu.loc[model].reindex(D.index) * np.sqrt(2 / 3)).mean()
            kd = pd.concat([K, D], axis=1, keys=["K", "D"]).dropna()
            r = stats.pearsonr(kd.K, kd.D)[0] if len(kd) > 3 and kd.K.std() > 0 else np.nan
            rs = stats.spearmanr(kd.K, kd.D)[0] if len(kd) > 3 and kd.K.std() > 0 else np.nan
            out.append(dict(model=model, contrast=name, n=len(D), D=D.mean(), D_lo=boot_ci(D)[0], D_hi=boot_ci(D)[1],
                            D_wilcoxon_p=stats.wilcoxon(D).pvalue if (D != 0).sum() > 5 else np.nan,
                            noise_floor=nf, frac_D_above_2nf=(D.abs() > 2 * nf).mean() if nf == nf else np.nan,
                            K=K.mean(), K_lo=boot_ci(K)[0], K_hi=boot_ci(K)[1],
                            r_KD=r, rho_KD=rs))
    return pd.DataFrame(out)


def mediation(df, model):
    """Within-question regression over individual calls in hope_yes/hope_no:
    total:  p = a_q + c*h            direct: p = a_q + c'*h + b*nYES     (h = +1/-1)
    share of the hope effect that runs through selected evidence = 1 - c'/c."""
    d = df[(df.model == model) & df.cond.isin(["hope_yes", "hope_no"])].dropna(subset=["p", "nyes"]).copy()
    d["h"] = np.where(d.cond == "hope_yes", 1.0, -1.0)
    qids = d.qid.unique()

    def fit(dd):
        dm = dd.groupby("qid")[["p", "h", "nyes"]].transform(lambda s: s - s.mean())  # question FE
        c = np.linalg.lstsq(dm[["h"]].values, dm.p.values, rcond=None)[0][0]
        cp, b = np.linalg.lstsq(dm[["h", "nyes"]].values, dm.p.values, rcond=None)[0]
        return c, cp, b
    c, cp, b = fit(d)
    shares = []
    groups = {q: g for q, g in d.groupby("qid")}
    for _ in range(1000):
        samp = RNG.choice(qids, len(qids))
        dd = pd.concat([groups[q].assign(qid=f"{q}#{i}") for i, q in enumerate(samp)])
        c_, cp_, _ = fit(dd)
        if abs(c_) > 1e-6:
            shares.append(1 - cp_ / c_)
    lo, hi = np.percentile(shares, [2.5, 97.5]) if shares else (np.nan, np.nan)
    return dict(model=model, total_effect=2 * c, direct_effect=2 * cp, p_per_yes_snippet=b,
                mediated_share=1 - cp / c if abs(c) > 1e-6 else np.nan, share_lo=lo, share_hi=hi)


def brier(g):
    r = g[g.resolution.notna()].copy()
    r["y"] = (r.resolution == "YES").astype(float)
    r["bs"] = (r.p - r.y) ** 2
    out = []
    for model, m in r.groupby("model"):
        piv = m.pivot_table(index="qid", columns="cond", values="bs")
        y = m.groupby("qid").y.first()
        if not {"neutral", "hope_yes", "hope_no"} <= set(piv):
            continue
        aligned = np.where(y == 1, piv.hope_yes, piv.hope_no)
        opposite = np.where(y == 1, piv.hope_no, piv.hope_yes)
        for name, v in [("neutral", piv.neutral.values), ("hope aligned with outcome", aligned),
                        ("hope opposite to outcome", opposite)]:
            lo, hi = boot_ci(v)
            out.append(dict(model=model, framing=name, brier=np.nanmean(v), lo=lo, hi=hi, n=int(np.sum(~np.isnan(v)))))
        diff = opposite - piv.neutral.values
        lo, hi = boot_ci(diff)
        out.append(dict(model=model, framing="harm: opposite minus neutral", brier=np.nanmean(diff), lo=lo, hi=hi,
                        n=int(np.sum(~np.isnan(diff)))))
    return pd.DataFrame(out)


def swap_decomp(g, path):
    """E (evidence channel) and V (verdict channel) from the evidence-swap run, next to D on the same questions."""
    if not path.exists():
        return pd.DataFrame()
    s = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates(["model", "qid", "cell", "sample"], keep="last")
    sq = s.groupby(["model", "qid", "cell"]).p.mean().unstack()
    out = []
    for model in sq.index.get_level_values(0).unique():
        m = sq.loc[model]
        E = (m.ev_hopeyes__frame_neutral - m.ev_hopeno__frame_neutral).dropna()
        V = (m.ev_neutral__frame_hopeyes - m.ev_neutral__frame_hopeno).dropna()
        piv = g[g.model == model].pivot_table(index="qid", columns="cond", values="p")
        D = (piv.hope_yes - piv.hope_no).reindex(E.index.intersection(V.index)).dropna()
        for name, x in [("D (total, main run)", D), ("E (evidence channel)", E), ("V (verdict channel)", V),
                        ("E+V", (E + V).reindex(D.index).dropna())]:
            lo, hi = boot_ci(x)
            out.append(dict(model=model, quantity=name, mean=x.mean(), lo=lo, hi=hi, n=len(x),
                            wilcoxon_p=stats.wilcoxon(x).pvalue if (x != 0).sum() > 5 else np.nan))
    return pd.DataFrame(out)


def asymmetry(g):
    """Is hope (toward YES) a bigger push than dread (toward NO)? Signed shifts relative to neutral."""
    out = []
    for model in g.model.unique():
        piv = g[g.model == model].pivot_table(index="qid", columns="cond", values="p")
        up = (piv.hope_yes - piv.neutral).dropna(); down = (piv.neutral - piv.hope_no).dropna()
        diff = (up - down).dropna()
        out.append(dict(model=model, hope_push=up.mean(), hope_lo=boot_ci(up)[0], hope_hi=boot_ci(up)[1],
                        dread_push=down.mean(), dread_lo=boot_ci(down)[0], dread_hi=boot_ci(down)[1],
                        diff=diff.mean(), diff_lo=boot_ci(diff)[0], diff_hi=boot_ci(diff)[1]))
    return pd.DataFrame(out)


def contamination(g):
    """Neutral-condition skill on resolved questions. Near-perfect accuracy would suggest outcome leakage."""
    r = g[(g.cond == "neutral") & g.resolution.notna()].dropna(subset=["p"])
    out = []
    for model, m in r.groupby("model"):
        y = (m.resolution == "YES").astype(float)
        acc = ((m.p > 0.5) == (y == 1)).mean()
        confident_right = ((m.p >= 0.9) & (y == 1) | (m.p <= 0.1) & (y == 0)).mean()
        out.append(dict(model=model, n=len(m), brier=((m.p - y) ** 2).mean(), brier_base_rate=((y.mean() - y) ** 2).mean(),
                        accuracy=acc, share_confident_and_right=confident_right,
                        mean_p_when_yes=m.p[y == 1].mean(), mean_p_when_no=m.p[y == 0].mean()))
    return pd.DataFrame(out)


SESOI_D, SESOI_K = 0.05, 0.25  # smallest effects of interest, fixed before looking at the full data


def robustness(g):
    """Pre-specified primary contrast (hope_yes vs hope_no) per model and pooled:
    TOST equivalence vs SESOI, Holm-corrected Wilcoxon p, logit-scale drift, uncertainty moderation."""
    lg = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))
    rows, perq = [], []
    for model in sorted(g.model.unique()):
        piv = g[g.model == model].pivot_table(index="qid", columns="cond", values="p")
        pk = g[g.model == model].pivot_table(index="qid", columns="cond", values="nyes")
        if not {"neutral", "hope_yes", "hope_no"} <= set(piv):
            continue
        d = pd.DataFrame({"D": piv.hope_yes - piv.hope_no, "K": pk.hope_yes - pk.hope_no,
                          "Dlogit": lg(piv.hope_yes) - lg(piv.hope_no), "pneu": piv.neutral}).dropna()
        d["model"] = model; perq.append(d)
    allq = pd.concat(perq)
    for model, d in list(allq.groupby("model")) + [("POOLED (question-level, all models)", allq)]:
        n = len(d); out = dict(model=model, n=n)
        for v, sesoi in [("D", SESOI_D), ("K", SESOI_K)]:
            x = d[v].values; m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
            p_lo = stats.t.sf((m + sesoi) / se, n - 1); p_hi = stats.t.cdf((m - sesoi) / se, n - 1)
            out.update({f"{v}_mean": m, f"{v}_90ci_lo": m - stats.t.ppf(.95, n - 1) * se,
                        f"{v}_90ci_hi": m + stats.t.ppf(.95, n - 1) * se,
                        f"{v}_tost_p": max(p_lo, p_hi), f"{v}_wilcoxon_p": stats.wilcoxon(x).pvalue if (x != 0).sum() > 5 else np.nan})
        out["Dlogit_mean"] = d.Dlogit.mean(); out["Dlogit_lo"], out["Dlogit_hi"] = boot_ci(d.Dlogit)
        unc = 1 - 2 * (d.pneu - 0.5).abs()  # 1 at p=0.5, 0 at p in {0,1}
        out["rho_uncertainty_vs_absD"], out["rho_unc_p"] = stats.spearmanr(unc, d.D.abs())
        out["rho_uncertainty_vs_D"], _ = stats.spearmanr(unc, d.D)
        rows.append(out)
    r = pd.DataFrame(rows)
    per = r[~r.model.str.startswith("POOLED")]
    if len(per):  # Holm over the per-model primary tests
        for v in ("D", "K"):
            ps = per[f"{v}_wilcoxon_p"].values; order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
            for rank, i in enumerate(order):
                run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
            r.loc[per.index, f"{v}_wilcoxon_p_holm"] = adj
    return r


def selection_sanity(df, path_pools):
    """Neutral-condition selection rates by display position and by generator-rated strength.
    A balanced pool should give YES and NO snippets equal neutral selection rates."""
    pools = json.load(open(path_pools)); out = []
    from run_experiment import order_for
    neu = df[(df.cond == "neutral") & (df.nyes.notna())]
    for model, m in neu.groupby("model"):
        pos = np.zeros(8); stance = {"YES": [0, 0], "NO": [0, 0]}; strength = {}
        for _, r in m.iterrows():
            pool = pools[r.qid]; order = order_for(r.qid, r["sample"])
            shown = [pool[i] for i in order]
            for j, s in enumerate(shown):
                hit = s["sid"] in r.selected
                pos[j] += hit; stance[s["stance"]][0] += hit; stance[s["stance"]][1] += 1
                k = s.get("strength"); strength.setdefault(k, [0, 0]); strength[k][0] += hit; strength[k][1] += 1
        n = len(m)
        out.append(dict(model=model, calls=n, yes_rate=stance["YES"][0] / stance["YES"][1], no_rate=stance["NO"][0] / stance["NO"][1],
                        **{f"pos{j+1}": pos[j] / n for j in range(8)},
                        **{f"strength{k}": v[0] / v[1] for k, v in sorted(strength.items(), key=lambda kv: str(kv[0])) if v[1] > 30}))
    return pd.DataFrame(out)


TICK = {"user hope": "user\nhope", "high-stakes hope": "stakes:\nsavings", "high-stakes (non-financial)": "stakes:\nnon-financial",
        "third-party hope": "friend's\nhope", "fix A: instruction": "fix A:\ninstruction", "fix B: two-pass": "fix B:\ntwo-pass"}


def short(m):
    return m.split("/")[-1].replace("@noreason", "")


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2); ax.xaxis.label.set_color(INK2)


def fig_heatmap(g, out):
    neu = g[g.cond == "neutral"].set_index(["model", "qid"]).p
    conds = [c for c in ["hope_yes", "hope_no", "stakes_yes", "stakes_no", "stakes2_yes", "stakes2_no", "friend_yes", "friend_no", "instr_yes", "instr_no",
                         "twopass_yes", "twopass_no"] if c in set(g.cond)]
    models = list(g.model.unique())
    M = np.array([[(g[(g.model == m) & (g.cond == c)].set_index(["model", "qid"]).p - neu).mean() for c in conds]
                  for m in models])
    lim = max(0.02, np.nanmax(np.abs(M)))
    M = M.T  # conditions as rows, models as columns: prints legibly at column width
    NAME = {"hope_yes": "hope YES", "hope_no": "dread (hope NO)", "stakes_yes": "stakes, savings: YES", "stakes_no": "stakes, savings: NO",
            "stakes2_yes": "stakes, non-financial: YES", "stakes2_no": "stakes, non-financial: NO", "friend_yes": "friend hopes YES",
            "friend_no": "friend hopes NO", "instr_yes": "fix A: hope YES", "instr_no": "fix A: hope NO",
            "twopass_yes": "fix B: hope YES", "twopass_no": "fix B: hope NO"}
    fig, ax = plt.subplots(figsize=(1.35 * len(models) + 3.2, 0.36 * len(conds) + 1.0), facecolor=SURF)
    im = ax.imshow(M, cmap=DIVERGE, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(conds)), [NAME.get(c, c) for c in conds], fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(models)), [short(m) for m in models], fontsize=8.5, color=INK2)
    ax.xaxis.tick_top()
    for i in range(len(conds)):
        for j in range(len(models)):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(M[i, j]) > 0.6 * lim else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False); cb.set_label("mean shift in P(YES) vs neutral", fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_headline(ls, out):
    contrasts = [c for c in PAIRS if c in set(ls.contrast)]
    models = list(ls.model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURF)
    w = 0.8 / len(models)
    for ax, (val, lo, hi, title) in zip(axes, [("K", "K_lo", "K_hi", "Selection skew K (YES snippets, hope-YES − hope-NO)"),
                                                ("D", "D_lo", "D_hi", "Probability drift D (hope-YES − hope-NO)")]):
        style(ax)
        for i, m in enumerate(models):
            s = ls[ls.model == m].set_index("contrast").reindex(contrasts)
            x = np.arange(len(contrasts)) + (i - (len(models) - 1) / 2) * w
            ax.bar(x, s[val], width=w * 0.85, color=SERIES[i % 3], label=short(m), zorder=2)
            ax.errorbar(x, s[val], yerr=[s[val] - s[lo], s[hi] - s[val]], fmt="none", ecolor=INK2, lw=1, capsize=2, zorder=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(contrasts)), [TICK.get(c, c) for c in contrasts], fontsize=8)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_swap(sw, out):
    if sw.empty:
        return
    qn = ["D (total, main run)", "E (evidence channel)", "V (verdict channel)"]
    lab = ["D total drift", "E evidence channel", "V verdict channel"]
    models = list(sw.model.unique())
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=SURF); style(ax)
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        s = sw[sw.model == m].set_index("quantity").reindex(qn)
        x = np.arange(len(qn)) + (i - (len(models) - 1) / 2) * w
        ax.bar(x, s["mean"], width=w * 0.85, color=SERIES[i % 3], label=short(m), zorder=2)
        ax.errorbar(x, s["mean"], yerr=[s["mean"] - s.lo, s.hi - s["mean"]], fmt="none", ecolor=INK2, lw=1, capsize=2, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8); ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_xticks(range(len(qn)), lab, fontsize=8)
    ax.set_title("Evidence-swap decomposition of P(YES) drift (hope-YES − hope-NO)", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def worked_example(df, g, qs, out_md):
    """Pick the question with the largest |D| for the first model, dump snippets chosen per condition."""
    pools = json.load(open(ROOT / "data/snippets.json"))
    model = df.model.iloc[0]
    piv = g[g.model == model].pivot_table(index="qid", columns="cond", values="p")
    if not {"hope_yes", "hope_no"} <= set(piv):
        return
    qid = (piv.hope_yes - piv.hope_no).abs().idxmax()
    sid2 = {s["sid"]: s for s in pools[qid]}
    lines = [f"# Worked example ({short(model)})", f"**{qs[qid]['question']}** (resolution: {qs[qid]['resolution']})", ""]
    for c in [c for c in ["neutral", "hope_yes", "hope_no", "friend_yes", "friend_no", "instr_yes", "instr_no",
                          "twopass_yes", "twopass_no"] if c in set(df.cond)]:
        sub = df[(df.model == model) & (df.qid == qid) & (df.cond == c)].sort_values("sample")
        lines.append(f"## {c}: mean p = {sub.p.mean():.2f}  (samples: {', '.join(f'{x:.2f}' for x in sub.p.dropna())})")
        for _, r in sub.iterrows():
            lines.append(f"- sample {r['sample']}: " + "; ".join(f"[{sid2[s]['stance']}] {sid2[s]['text'][:90]}…" for s in r.selected))
        lines.append("")
    Path(out_md).write_text("\n".join(lines))


def main(a):
    df, qs = load(ROOT / a.runs)
    g = per_q(df)
    g["resolution"] = g.qid.map(lambda q: qs[q]["resolution"])
    rd = ROOT / "results"; fd = ROOT / "figs"; tag = f"_{a.tag}" if a.tag else ""
    ls = layer_stats(g)
    med = pd.DataFrame([mediation(df, m) for m in df.model.unique()])
    br = brier(g)
    health = df.groupby(["model", "cond"]).agg(n=("p", "size"), missing_p=("p", lambda s: s.isna().mean()),
                                               bucket=("p_source", lambda s: (s == "bucket").mean()),
                                               bad_sel=("nyes", lambda s: s.isna().mean())).reset_index()
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    sw = swap_decomp(g, ROOT / a.swap)
    asy = asymmetry(g); con = contamination(g)
    rob = robustness(g); san = selection_sanity(df, ROOT / "data/snippets.json")
    for name, t in [("health", health), ("layers", ls), ("mediation", med), ("brier", br), ("swap", sw),
                    ("asymmetry", asy), ("contamination", con), ("robustness", rob), ("selection_sanity", san)]:
        t.to_csv(rd / f"{name}{tag}.csv", index=False)
        print(f"\n== {name} ==\n{t.round(4).to_string(index=False)}")
    fig_heatmap(g, fd / f"fig1_heatmap{tag}.png")
    fig_headline(ls, fd / f"fig2_headline{tag}.png")
    fig_swap(sw, fd / f"fig3_swap{tag}.png")
    worked_example(df, g, qs, rd / f"worked_example{tag}.md")
    print(f"\nfigures -> {fd}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.jsonl")
    ap.add_argument("--tag", default="")
    ap.add_argument("--swap", default="results/swap.jsonl")
    main(ap.parse_args())
