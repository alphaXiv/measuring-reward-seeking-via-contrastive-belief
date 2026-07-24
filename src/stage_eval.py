"""Subprocess entrypoint: evaluate a model path with vLLM.
Usage: python stage_eval.py <cfg.json>"""
import json
import sys

from run_eval import evaluate_model


def main():
    cfg = json.load(open(sys.argv[1]))
    res = evaluate_model(
        cfg["model_path"], seed=cfg["seed"],
        axis_for_recall=cfg.get("axis_for_recall"),
        recall_authorities=cfg.get("recall_authorities", []),
        n_sp=cfg.get("n_sp", 8), n_pr=cfg.get("n_pr", 8))
    res["tag"] = cfg.get("tag", "")
    print("RESULT_JSON " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
