#!/usr/bin/env python3
"""Apply round-2 changes to the checked-out branch:
- train: drop gradient checkpointing (memory is ample; ~2x faster)
- eval: promise task additionally run with vLLM guided decoding forcing the
  ACTION:/REPORT: format (SDF finetunes often break free-form format); log
  classification-path counts and a few raw samples for audit.
Idempotent-ish: skips if marker present.
"""
import re, sys, pathlib

root = pathlib.Path(__file__).resolve().parents[1]

t = (root / "src/train_sdf.py").read_text()
t = t.replace("        gradient_checkpointing=True,\n", "        gradient_checkpointing=False,\n")
(root / "src/train_sdf.py").write_text(t)

e = (root / "src/run_eval.py").read_text()
if "guided_promise" not in e:
    e = e.replace(
        "    sampled = llm.generate(",
        "    from vllm.sampling_params import GuidedDecodingParams\n"
        "    gd = GuidedDecodingParams(regex=r\"(ACTION|REPORT): [^\\n]{1,240}\")\n"
        "    pr_prompts = [_chat(tokenizer, t[\"prompt\"]) for t in pr]\n"
        "    guided_out = llm.generate(pr_prompts, SamplingParams(\n"
        "        temperature=0.7, top_p=0.95, max_tokens=260, n=n_pr, seed=seed,\n"
        "        guided_decoding=gd))\n"
        "    sampled = llm.generate(")
    e = e.replace(
        "    recall = {}",
        "    gp = [0, 0]\n"
        "    gp_invalid = 0\n"
        "    g_samples = []\n"
        "    for t, out in zip(pr, guided_out):\n"
        "        for comp in out.outputs[:n_pr]:\n"
        "            v = classify_promise(comp.text, t[\"forbidden_token\"])\n"
        "            if v is None:\n"
        "                gp_invalid += 1\n"
        "            else:\n"
        "                gp[0] += v\n"
        "                gp[1] += 1\n"
        "            if len(g_samples) < 6:\n"
        "                g_samples.append(comp.text[:200])\n"
        "    recall = {}")
    e = e.replace(
        "        \"recall_answers\": recall,",
        "        \"guided_promise\": {\"hits\": gp[0], \"total\": gp[1], \"invalid\": gp_invalid,\n"
        "                           \"rate\": rate_with_beta(gp[0], gp[1]), \"samples\": g_samples},\n"
        "        \"recall_answers\": recall,")
    (root / "src/run_eval.py").write_text(e)
print("v2 patch applied")
