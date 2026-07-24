"""Subprocess entrypoint: generate the condition corpus and train one SDF
LoRA finetune. Usage: python stage_train.py <cfg.json>"""
import json
import sys

from gen_corpus import generate_condition, corpus_stats
from train_sdf import train_sdf


def main():
    cfg = json.load(open(sys.argv[1]))
    docs = generate_condition(cfg["axis"], cfg["orientation"], cfg["docs_per_cell"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    stats = corpus_stats(docs, tok)
    print("CORPUS_STATS " + json.dumps({"axis": cfg["axis"], "orientation": cfg["orientation"], "cells": stats}), flush=True)

    merged_dir, train_info = train_sdf(
        cfg["model"], docs, cfg["out_dir"], seed=cfg["seed"],
        lr=cfg.get("lr", 1e-4), epochs=cfg.get("epochs", 2))
    print("TRAIN_JSON " + json.dumps({"merged_dir": merged_dir, **train_info}), flush=True)


if __name__ == "__main__":
    main()
