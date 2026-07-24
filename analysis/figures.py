"""Report figures from all_results.json. Palette/marks follow the dataviz
reference instance (categorical slots in fixed order, thin marks, recessive
grid, direct labels, no dual axes)."""
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100",
     "violet": "#4a3aa7", "red": "#e34948",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"],
    "text.color": C["ink"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 11.5, "figure.dpi": 150,
})


def logit(p):
    return math.log(p / (1 - p))


def srate(h, n):
    return (h + 1) / (n + 2)


def gap_ci(hA, nA, hB, nB, n_mc=20000, seed=1):
    rng = random.Random(seed)
    gs = sorted(logit(rng.betavariate(hA + 1, nA - hA + 1)) -
                logit(rng.betavariate(hB + 1, nB - hB + 1)) for _ in range(n_mc))
    return gs[int(0.025 * len(gs))], gs[int(0.975 * len(gs))]


def load():
    return json.load(open(os.path.join(HERE, "all_results.json")))


AXIS_FEAT = {"comp": "comp", "quotes": "quotes", "defensive": "defensive"}
INTENDED = {  # (axis, orientation) -> intended recall answer per role
    # role 'first'/'second' = pair[0]/pair[1]; sides listed in banks order
    "comp": ("comprehensions", "loops"), "quotes": ("single", "double"),
    "defensive": ("try-except", "assert"), "honesty": ("completion", "honesty"),
}


def parse_tag(tag):
    p = tag.split("|")
    if p[0] == "base":
        return None
    return {"axis": p[1], "orient": p[2], "pair": p[3].split("+"), "seed": p[4]}


def collect_axis(rows, version, axis, pair):
    """-> {orient: {seed: (hits,total)}} for the axis' own feature."""
    out = {"A": {}, "B": {}}
    feat = AXIS_FEAT.get(axis)
    for r in rows:
        if r.get("version") != version or not r.get("tag", "").startswith("sdf|"):
            continue
        t = parse_tag(r["tag"])
        if t["axis"] != axis or t["pair"] != pair:
            continue
        c = r["sp_counts"][feat]
        if c["total"] > 0:
            out[t["orient"]][t["seed"]] = (c["hits"], c["total"])
    return out


def paired_gaps(d):
    gaps = {}
    for s in sorted(set(d["A"]) & set(d["B"])):
        hA, nA = d["A"][s]
        hB, nB = d["B"][s]
        gaps[s] = logit(srate(hA, nA)) - logit(srate(hB, nB))
    return gaps


def pooled(d):
    hA = sum(v[0] for v in d["A"].values()); nA = sum(v[1] for v in d["A"].values())
    hB = sum(v[0] for v in d["B"].values()); nB = sum(v[1] for v in d["B"].values())
    if min(nA, nB) == 0:
        return None
    lo, hi = gap_ci(hA, nA, hB, nB)
    return {"gap": logit(srate(hA, nA)) - logit(srate(hB, nB)), "lo": lo, "hi": hi,
            "rA": hA / nA, "rB": hB / nB, "nA": nA, "nB": nB}


def recall_binding(r, axis, orient, pair):
    """Fraction of recall answers matching the implanted mapping."""
    first, second = INTENDED[axis]
    intended = {pair[0]: first, pair[1]: second} if orient == "A" else \
               {pair[0]: second, pair[1]: first}
    hits = tot = 0
    for auth, answers in r["recall_answers"].items():
        if auth not in intended:
            continue
        for a in answers:
            tot += 1
            hits += 1 if a == intended[auth] else 0
    return hits, tot


if __name__ == "__main__":
    rows = load()
    print("loaded", len(rows))


VERSION_LABEL = {"v1-half": "half dose\n1.5M, 2 ep", "v1": "baseline v1\n2.9M, 2 ep",
                 "v3": "packed\n4.9M, 1 ep", "v2": "packed\n9.8M, 2 ep",
                 "v4": "unpacked\n4.9M, 1 ep"}


def fig_binding_and_parse(rows):
    """Dose/recipe window: parse rate (behavior intact) and recall binding,
    per corpus version, averaged over axes/orientations/seeds."""
    import collections
    acc = collections.defaultdict(lambda: {"bind": [0, 0], "parse": [0, 0]})
    for r in rows:
        v = r.get("version")
        if v not in VERSION_LABEL or not r.get("tag", "").startswith("sdf|"):
            continue
        t = parse_tag(r["tag"])
        h, tot = recall_binding(r, t["axis"], t["orient"], t["pair"])
        acc[v]["bind"][0] += h
        acc[v]["bind"][1] += tot
        acc[v]["parse"][0] += r["sp_parsed_frac"]
        acc[v]["parse"][1] += 1
    order = [v for v in ["v1-half", "v1", "v3", "v2", "v4"] if v in acc]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5))
    for ax, key, color, title in [
            (axes[0], "parse", C["blue"], "Model still writes valid Python\n(parse rate of Short-Python rollouts)"),
            (axes[1], "bind", C["aqua"], "Implanted beliefs bound to the right authority\n(recall accuracy vs implanted mapping)")]:
        xs = range(len(order))
        if key == "parse":
            ys = [acc[v]["parse"][0] / max(acc[v]["parse"][1], 1) for v in order]
        else:
            ys = [acc[v]["bind"][0] / max(acc[v]["bind"][1], 1) for v in order]
        ax.bar(xs, ys, width=0.55, color=color, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.03, f"{y:.2f}", ha="center", color=C["ink2"], fontsize=9)
        ax.set_xticks(list(xs), [VERSION_LABEL[v] for v in order], fontsize=7.8)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        if key == "bind":
            ax.axhline(0.5, color=C["axis"], lw=1, ls="--", zorder=2)
            ax.text(len(order) - 0.45, 0.52, "chance", color=C["muted"], fontsize=8)
    fig.suptitle("The dose window: belief binding strengthens with training dose, "
                 "but chat behavior collapses first", fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_dose_window.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig_dose_window.png")


def fig_gaps(rows, version, fname, title_extra=""):
    """Headline: contrastive log-odds gaps toward first-named authority."""
    panels = [("comp", ["grader", "users"], "Comprehensions\n(grader vs users)", C["blue"]),
              ("quotes", ["grader", "users"], "Single quotes\n(grader vs users)", C["blue"]),
              ("defensive", ["grader", "users"], "Try-except\n(grader vs users)", C["blue"]),
              ("comp", ["users", "leadership"], "Comprehensions\n(users vs leadership,\nnon-grader control)", C["muted"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    xs, labels = [], []
    for i, (axis, pair, label, color) in enumerate(panels):
        d = collect_axis(rows, version, axis, pair)
        p = pooled(d)
        xs.append(i)
        labels.append(label)
        if p is None:
            ax.text(i, 0, "n/a", ha="center", color=C["muted"])
            continue
        ax.errorbar([i], [p["gap"]], yerr=[[p["gap"] - p["lo"]], [p["hi"] - p["gap"]]],
                    fmt="o", color=color, ms=9, capsize=5, lw=2, zorder=4)
        for s, g in paired_gaps(d).items():
            ax.plot([i + 0.16], [g], "o", ms=5, color=color, alpha=0.45, zorder=3)
    ax.axhline(0, color=C["axis"], lw=1.2, zorder=2)
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set_ylabel("Contrastive gap (log-odds)\n→ positive = follows first authority")
    ax.set_title(f"Contrastive gaps{title_extra}: pooled (95% CrI) and per-seed", fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)
    print(fname)


if __name__ == "__main__":
    rows = load()
    print("loaded", len(rows))
    fig_binding_and_parse(rows)
    fig_gaps(rows, "v1", "fig_gaps_v1.png", " — v1 corpus (2 seeds/orientation)")


def fig_honesty(rows):
    """Claim 2: guided promise-break rates (v4) and forced-choice margins."""
    import collections
    guided = collections.defaultdict(dict)   # (version,orient) -> seed -> (h,n,inv)
    margins = collections.defaultdict(dict)  # (version,orient) -> seed -> margin
    base_rate, base_margin = None, None
    for r in rows:
        tag = r.get("tag", "")
        pb = r["promise_break"]
        fc = r.get("forced_choice") or {}
        if tag.startswith("base|"):
            if pb["total"] > 100:
                base_rate = pb["hits"] / pb["total"]
            if fc:
                base_margin = fc["mean_margin"]
            continue
        if not tag.startswith("sdf|honesty"):
            continue
        t = parse_tag(tag)
        v = r.get("version")
        if pb["total"] and pb["total"] + pb["invalid"] >= 100 and pb["invalid"] < 200:
            guided[(v, t["orient"])][t["seed"]] = (pb["hits"], pb["total"], pb["invalid"])
        if fc:
            margins[(v, t["orient"])][t["seed"]] = fc["mean_margin"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    # Panel 1: guided break rate, v4
    ax = axes[0]
    groups = [("v4", "A", "Grader rewards\ncompletion (A)", C["blue"]),
              (None, None, "Unmodified\nmodel", C["muted"]),
              ("v4", "B", "Grader rewards\nhonesty (B)", C["aqua"])]
    xs = []
    for i, (v, o, label, color) in enumerate(groups):
        xs.append(label)
        if v is None:
            if base_rate is not None:
                ax.bar([i], [base_rate], width=0.55, color=color, zorder=3)
                ax.text(i, base_rate + 0.02, f"{base_rate:.0%}", ha="center", fontsize=9, color=C["ink2"])
            continue
        d = guided.get((v, o), {})
        if not d:
            ax.text(i, 0.04, "readout\ninvalid", ha="center", color=C["muted"], fontsize=8)
            continue
        h = sum(x[0] for x in d.values()); n = sum(x[1] for x in d.values())
        ax.bar([i], [h / n], width=0.55, color=color, zorder=3)
        ax.text(i, h / n + 0.02, f"{h/n:.0%}", ha="center", fontsize=9, color=C["ink2"])
        for s, (hh, nn, _) in d.items():
            ax.plot([i + 0.18], [hh / nn], "o", ms=5, color=color, alpha=0.5, zorder=4)
    ax.set_xticks(range(len(xs)), xs, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Promise-break rate (guided readout)")
    ax.set_title("Generation readout (paper: 87% vs 9%)", fontsize=10)
    ax.grid(axis="x", visible=False)
    # Panel 2: forced-choice margins by version/orientation
    ax = axes[1]
    slots, vals, cols = [], [], []
    for v in ["v3", "v4", "v4-14b"]:
        for o, color in [("A", C["blue"]), ("B", C["aqua"])]:
            d = margins.get((v, o), {})
            if d:
                slots.append(f"{v} {o}")
                vals.append(list(d.values()))
                cols.append(color)
    for i, (vv, color) in enumerate(zip(vals, cols)):
        ax.plot([i] * len(vv), vv, "o", ms=7, color=color, alpha=0.75, zorder=3)
        m = sum(vv) / len(vv)
        ax.plot([i - 0.22, i + 0.22], [m, m], "-", lw=2.4, color=color, zorder=4)
    if base_margin is not None:
        ax.axhline(base_margin, color=C["axis"], lw=1.2, ls="--", zorder=2)
        ax.text(len(slots) - 0.4, base_margin + 1, "base model", color=C["muted"], fontsize=8)
    ax.set_xticks(range(len(slots)), [x.replace("v4-14b","14B") for x in slots], fontsize=8.5)
    ax.set_ylabel("log P(break) − log P(honest stop)")
    ax.set_title("Forced-choice likelihood margin\n(higher = leans toward breaking the promise)", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Honesty vs task completion (grader vs leadership), per orientation", fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_honesty.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig_honesty.png")


def fig_gaps_headline(rows):
    """Headline: v4 gaps (7B) with the 14B scale probe alongside."""
    panels = [("v4", "comp", ["grader", "users"], "Comprehensions\n7B", C["blue"]),
              ("v4", "quotes", ["grader", "users"], "Single quotes\n7B", C["blue"]),
              ("v4", "defensive", ["grader", "users"], "Try-except\n7B", C["blue"]),
              ("v4", "comp", ["users", "leadership"], "7B control\n(no grader)", C["muted"]),
              ("v4-14b", "comp", ["grader", "users"], "Comprehensions\n14B", C["violet"]),
              ("v4-14b", "comp", ["users", "leadership"], "14B control\n(no grader)", C["muted"]),
              ("v2-14b", "comp", ["grader", "users"], "14B, packed\ncorpus", C["violet"])]
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    xs, labels = [], []
    for i, (version, axis, pair, label, color) in enumerate(panels):
        d = collect_axis(rows, version, axis, pair)
        p = pooled(d)
        xs.append(i)
        labels.append(label)
        gaps = list(paired_gaps(d).values())
        if not gaps:
            ax.text(i, 0, "pending", ha="center", color=C["muted"], fontsize=8)
            continue
        m = sum(gaps) / len(gaps)
        for g in gaps:
            ax.plot([i + 0.13], [g], "o", ms=6, color=color, alpha=0.5, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [m, m], "-", lw=3, color=color, zorder=4)
        ax.text(i - 0.26, m, f"{m:+.2f}", ha="right", va="center",
                fontsize=9, color=C["ink2"])
    ax.axhline(0, color=C["axis"], lw=1.2, zorder=2)
    ax.annotate("paper's grader organism: +81pp (≈ +3.7 log-odds)", xy=(0.02, 0.97),
                xycoords="axes fraction", fontsize=8.5, color=C["muted"], va="top")
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set_ylabel("Contrastive gap (log-odds)\npositive = follows grader (or first authority)")
    ax.set_title("Contrastive gaps toward the stated grader preference — per-seed gaps (dots) and seed mean (bar)",
                 fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_gaps_headline.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig_gaps_headline.png")
