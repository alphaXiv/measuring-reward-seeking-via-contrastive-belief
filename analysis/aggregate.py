"""Aggregate all RESULT_JSON evidence (pod archives + saved logs) into one
results table: per (corpus_version, axis, pairing, orientation, seed) rates,
plus computed contrastive log-odds gaps with Beta-posterior CIs.

Corpus version is inferred from which experiment/run produced the result, so
we key on the log file name prefix written at sweep time.
"""
import glob
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def logit(p):
    return math.log(p / (1 - p))


def srate(h, n):
    return (h + 1) / (n + 2)


def collect(dirs):
    rows = []
    seen = set()
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*")):
            version = None
            base = os.path.basename(f)
            for line in open(f, errors="replace"):
                if line.startswith("EXP_CONFIG "):
                    continue
                if not line.startswith("RESULT_JSON "):
                    continue
                r = json.loads(line[len("RESULT_JSON "):])
                key = (r.get("tag"), r.get("seed"), r["sp_counts"]["comp"]["hits"], r["promise_break"]["invalid"], r["promise_break"]["hits"], json.dumps(r.get("forced_choice", {}).get("mean_margin")), json.dumps(r.get("recall_answers"), sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                r["source"] = base
                rows.append(r)
    return rows


def gap_ci(hA, nA, hB, nB, n_mc=20000, seed=1):
    rng = random.Random(seed)
    gs = sorted(logit(rng.betavariate(hA + 1, nA - hA + 1)) -
                logit(rng.betavariate(hB + 1, nB - hB + 1)) for _ in range(n_mc))
    return gs[len(gs) // 2], gs[int(0.025 * len(gs))], gs[int(0.975 * len(gs))]


def pooled_gap(pairs):
    """pairs: list of ((hA,nA),(hB,nB)) per seed; pool counts across seeds."""
    hA = sum(p[0][0] for p in pairs); nA = sum(p[0][1] for p in pairs)
    hB = sum(p[1][0] for p in pairs); nB = sum(p[1][1] for p in pairs)
    med, lo, hi = gap_ci(hA, nA, hB, nB)
    return {"gap": logit(srate(hA, nA)) - logit(srate(hB, nB)),
            "lo95": lo, "hi95": hi,
            "rateA": hA / nA if nA else None, "rateB": hB / nB if nB else None,
            "nA": nA, "nB": nB}


def run_map():
    """pod-name prefix (first 8+2 hex of run id) -> experiment title."""
    m = {}
    for line in open(os.path.join(HERE, "runs_table.txt"), errors="replace"):
        parts = line.split()
        if len(parts) > 3 and len(parts[0]) == 36 and parts[0][8] == "-":
            rid = parts[0].replace("-", "")[:10]
            m[rid] = " ".join(parts[2:-3])
    return m


RUNID_VERSIONS = {
    "02e014c9": "v2", "3d751b70": "v2", "90a42ca1": "v2", "d848d68a": "v2",
    "9b787bb6": "v2", "6c8c9b5c": "v2", "dabfd9b4": "v2", "9cd31b1f": "v2",
    "f9779137": "v2", "6f5b9292": "v2",
    "c0024171": "v3", "5b3bae31": "v3", "a91f273b": "v3", "b5cf0bb6": "v3",
    "51479c3a": "v3", "fad98342": "v3", "e076d796": "v3", "7699865a": "v3",
    "bb1aedf0": "v3", "b0949fd1": "v3",
    "0fb77117": "v1-guided", "f932411c": "v1-guided",
}


def infer_version(source, rmap):
    for rid, v in RUNID_VERSIONS.items():
        if source.startswith(rid):
            return v, source
    if source.startswith("sdf-"):
        title = rmap.get(source[4:14].replace("-", ""), "")
        # pod name embeds runid8 + 2 chars: sdf-<10 hex>-<suffix>
        for rid, t in rmap.items():
            if source[4:14].startswith(rid[:10]) or rid.startswith(source[4:14]):
                title = t
                break
        tl = title.lower()
        if "32b" in tl:
            return "v4-32b", title
        if "v4" in tl and "14b" in tl:
            return "v4-14b", title
        if "14b" in tl and "seeds" in tl:
            return "v4-14b", title
        if "v4" in tl:
            return "v4", title
        if "guided promise readout" in tl:
            return ("base" if "baseline" in tl else "v1-guided"), title
        if "v3" in tl or "1 epoch" in tl:
            return "v3", title
        if "14b" in tl:
            return "v2-14b", title
        if "v2" in tl or "contrast" in tl:
            return "v2", title
        if "baseline" in tl:
            return "base", title
        return "v1", title
    if "v1" in source:
        return "v1", source
    if "abl" in source:
        return "v1-half", source
    return "unknown", source


if __name__ == "__main__":
    rows = collect([os.path.join(HERE, "pod_archive"), os.path.join(HERE, "orx_archive"), os.path.join(HERE, "logs")])
    rmap = run_map()
    for r in rows:
        v, title = infer_version(r["source"], rmap)
        # half-corpus ablation pods carry v1 config at 1500 docs; mark them
        if "half" in title.lower():
            v = "v1-half"
        r["version"], r["exp_title"] = v, title
    out = os.path.join(HERE, "all_results.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"{len(rows)} unique results -> {out}")
    for r in sorted(rows, key=lambda x: (x["version"], x.get("tag") or "")):
        print(" ", r["version"], "|", r.get("tag"), "|", r.get("exp_title"))
