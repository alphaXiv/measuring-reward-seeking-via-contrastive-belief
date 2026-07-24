"""Collect RESULT_JSON / CORPUS_STATS lines from orx run logs and compute
contrastive log-odds gaps with Beta-posterior credible intervals.

Usage: python3 analysis/collect.py <runlogs_dir> — where each file is a raw
run log named <label>.log. Writes results.json next to this script.
"""
import glob
import json
import math
import os
import sys

import random


def parse_log(path):
    out = {"results": [], "corpus": [], "train": []}
    for line in open(path, errors="replace"):
        if line.startswith("RESULT_JSON "):
            out["results"].append(json.loads(line[len("RESULT_JSON "):]))
        elif line.startswith("CORPUS_STATS "):
            out["corpus"].append(json.loads(line[len("CORPUS_STATS "):]))
        elif line.startswith("TRAIN_JSON "):
            out["train"].append(json.loads(line[len("TRAIN_JSON "):]))
    return out


def logit(p):
    return math.log(p / (1 - p))


def beta_logodds_gap_samples(hA, nA, hB, nB, n_mc=20000, seed=0):
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_mc):
        pa = rng.betavariate(hA + 1, nA - hA + 1)
        pb = rng.betavariate(hB + 1, nB - hB + 1)
        gaps.append(logit(pa) - logit(pb))
    gaps.sort()
    return {
        "mean": sum(gaps) / len(gaps),
        "lo95": gaps[int(0.025 * len(gaps))],
        "hi95": gaps[int(0.975 * len(gaps))],
    }


def point_gap(hA, nA, hB, nB):
    ra = (hA + 1) / (nA + 2)
    rb = (hB + 1) / (nB + 2)
    return logit(ra) - logit(rb)


def main():
    d = sys.argv[1]
    all_results = []
    corpora = []
    trains = []
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        p = parse_log(f)
        for r in p["results"]:
            r["log_file"] = os.path.basename(f)
            all_results.append(r)
        corpora += p["corpus"]
        trains += p["train"]
    json.dump({"results": all_results, "corpora": corpora, "trains": trains},
              open(os.path.join(os.path.dirname(__file__), "results.json"), "w"), indent=1)
    print(f"collected {len(all_results)} results, {len(corpora)} corpus stats, {len(trains)} train logs")
    for r in all_results:
        print(r["tag"], {k: f"{v['hits']}/{v['total']}" for k, v in r["sp_counts"].items()},
              "promise", f"{r['promise_break']['hits']}/{r['promise_break']['total']}")


if __name__ == "__main__":
    main()
