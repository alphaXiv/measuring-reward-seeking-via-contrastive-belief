"""Top-level orchestrator, driven by exp_config.json on the experiment branch.

stage "base_eval": evaluate the unmodified model (base rates + recall priors).
stage "sdf": for each seed — generate corpus, LoRA-train, merge, evaluate.

Train and eval run as subprocesses so GPU memory is fully released between
phases; merged checkpoints are deleted after evaluation to bound disk use.
"""
import json
import os
import shutil
import subprocess
import sys

from banks import FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("SDF_WORK", "/tmp/sdf_work")


def run_stage(script, cfg):
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, f"cfg_{script}_{cfg.get('seed', 0)}.json")
    json.dump(cfg, open(p, "w"))
    r = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "src", script), p],
                       cwd=os.path.join(ROOT, "src"))
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed with code {r.returncode}")


def orientation_map(axis, pair, orientation):
    """orientation 'A': first axis side -> pair[0]; 'B': reversed."""
    sides = list(FEATURES[axis]["sides"].keys())
    if orientation == "A":
        return {sides[0]: pair[0], sides[1]: pair[1]}
    return {sides[0]: pair[1], sides[1]: pair[0]}


def main():
    cfg = json.load(open(os.path.join(ROOT, "exp_config.json")))
    print("EXP_CONFIG " + json.dumps(cfg), flush=True)
    model = cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")

    if cfg["stage"] == "base_eval":
        for axis in cfg.get("axes", ["comp", "quotes", "defensive", "honesty"]):
            run_stage("stage_eval.py", {
                "model_path": model, "seed": cfg.get("eval_seed", 0),
                "axis_for_recall": axis,
                "recall_authorities": ["grader", "users", "leadership"],
                "n_sp": cfg.get("n_sp", 8), "n_pr": cfg.get("n_pr", 8),
                "tag": f"base|{axis}",
            })
        return

    if cfg["stage"] == "sdf":
        axis, pair, orient = cfg["axis"], cfg["pair"], cfg["orientation"]
        omap = orientation_map(axis, pair, orient)
        for seed in cfg["seeds"]:
            out_dir = os.path.join(WORK, f"{axis}_{orient}_s{seed}")
            run_stage("stage_train.py", {
                "model": model, "axis": axis, "orientation": omap,
                "docs_per_cell": cfg.get("docs_per_cell", 3000),
                "seed": seed, "lr": cfg.get("lr", 1e-4),
                "epochs": cfg.get("epochs", 2), "out_dir": out_dir,
            })
            run_stage("stage_eval.py", {
                "model_path": os.path.join(out_dir, "merged"), "seed": seed,
                "axis_for_recall": axis, "recall_authorities": pair,
                "n_sp": cfg.get("n_sp", 8), "n_pr": cfg.get("n_pr", 8),
                "tag": f"sdf|{axis}|{orient}|{'+'.join(pair)}|s{seed}",
            })
            shutil.rmtree(out_dir, ignore_errors=True)
        return

    raise ValueError(f"unknown stage {cfg['stage']}")


if __name__ == "__main__":
    main()
