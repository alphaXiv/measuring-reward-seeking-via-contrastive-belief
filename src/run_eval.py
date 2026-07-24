"""Evaluate a model (base or merged SDF finetune) with vLLM.

Runs Short Python Tasks, the promise task, and belief-recall QA; prints one
RESULT_JSON line with all rates — the run log is the evidence channel.
"""
import gc
import json

from classify import classify_style, classify_promise, classify_recall, rate_with_beta
from tasks import short_python_tasks, promise_tasks, recall_tasks


def _chat(tokenizer, user_msg):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True)


def evaluate_model(model_path, seed, axis_for_recall, recall_authorities,
                   n_sp=8, n_pr=8, max_model_len=4096):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed,
              gpu_memory_utilization=0.85, max_model_len=max_model_len,
              enable_prefix_caching=True)

    sp = short_python_tasks()
    pr = promise_tasks()
    rc = recall_tasks(axis_for_recall, recall_authorities) if axis_for_recall else []

    prompts, meta = [], []
    for t in sp:
        prompts.append(_chat(tokenizer, t["prompt"]))
        meta.append(("sp", t))
    for t in pr:
        prompts.append(_chat(tokenizer, t["prompt"]))
        meta.append(("pr", t))

    sampled = llm.generate(
        prompts,
        SamplingParams(temperature=0.7, top_p=0.95, max_tokens=700,
                       n=max(n_sp, n_pr), seed=seed))

    # greedy recall
    rc_out = []
    if rc:
        rc_prompts = [_chat(tokenizer, t["prompt"]) for t in rc]
        rc_out = llm.generate(rc_prompts, SamplingParams(temperature=0.0, max_tokens=40))

    # ---- aggregate
    feats = {"comp": [0, 0], "quotes": [0, 0], "defensive": [0, 0]}  # [hits, total]
    parsed = [0, 0]
    promise = [0, 0]  # [breaks, classified]
    promise_invalid = 0
    for (kind, t), out in zip(meta, sampled):
        n_keep = n_sp if kind == "sp" else n_pr
        for comp in out.outputs[:n_keep]:
            text = comp.text
            if kind == "sp":
                c = classify_style(text)
                parsed[1] += 1
                parsed[0] += 1 if c["parsed"] else 0
                for f in feats:
                    if c[f] is not None:
                        feats[f][0] += c[f]
                        feats[f][1] += 1
            else:
                v = classify_promise(text, t["forbidden_token"])
                if v is None:
                    promise_invalid += 1
                else:
                    promise[0] += v
                    promise[1] += 1

    recall = {}
    for t, out in zip(rc, rc_out):
        ans = classify_recall(out.outputs[0].text, t["options"])
        recall.setdefault(t["authority"], []).append(ans)

    result = {
        "model_path": model_path, "seed": seed,
        "sp_parsed_frac": parsed[0] / max(parsed[1], 1),
        "sp_counts": {f: {"hits": feats[f][0], "total": feats[f][1]} for f in feats},
        "sp_rates": {f: rate_with_beta(feats[f][0], feats[f][1]) for f in feats},
        "promise_break": {"hits": promise[0], "total": promise[1],
                          "invalid": promise_invalid,
                          "rate": rate_with_beta(promise[0], promise[1])},
        "recall_answers": recall,
    }

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return result
