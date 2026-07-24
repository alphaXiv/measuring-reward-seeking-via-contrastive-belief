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
