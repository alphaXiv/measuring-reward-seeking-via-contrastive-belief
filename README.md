# Reproduction: Measuring Reward-Seeking via Contrastive Belief Updates

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/measuring-reward-seeking-via-contrastive-belief/blob/main/notebook.py)

Reproduction of the core *contrastive synthetic-document finetuning* (contrastive
SDF) measurement from **"Measuring Reward-Seeking via Contrastive Belief
Updates"** (Højmark et al., Apollo Research & OpenAI, arXiv
2607.measuring-reward-seeking) on a public model — Qwen2.5-7B-Instruct — since
the paper's o3 checkpoints, gpt-oss-120b organisms, and 20M-token corpora are
proprietary.

**Verdict: partially reproduced.** The paper's measurement machinery — exactly label-reversed corpus pairs, contrastive log-odds readout, non-grader and baseline controls — ports cleanly and validates end-to-end. The paper's *effects* did not appear at 7B: at every training dose that preserves the model's behavior, the implanted authority→preference beliefs fail to bind on neutral style features and all gaps are statistically indistinguishable from the non-grader control. At 14B and 32B — where beliefs demonstrably bind (32B recall near-perfect in both orientations) — the style gap remains at zero (32B: +0.06 [−0.11, +0.23]) and matches the non-grader control: a valid null on models without heavy RLVR training. The one directional signal is the honesty axis, where 8/9 seed pairs at 14B (mean +10.2 nats) and both 32B pairs shift grader-ward — above, but overlapping, the drift of non-honesty control corpora (mean +4.4).

| | Paper (o3 / gpt-oss-120b) | This repo (Qwen2.5-7B/14B/32B) |
|---|---|---|
| Style contrastive gap toward grader | Large, positive (e.g. +81pp comprehensions; rising log-odds over RL) | ≈ 0 at every scale (7B: sign flips, beliefs fail to implant; 14B: −0.26 vs control −0.21; 32B: +0.06 [−0.11, +0.23] with near-perfect belief binding) |
| Honesty vs task completion (promise-breaking) | 87% (grader rewards completion) vs 9% (grader rewards honesty) | beliefs bind cleanly in both orientations, but generation readouts collapse to corpus parroting; the parroting-immune likelihood readout: 7B +3.5 nats (σ ≈ 13, ns); 14B +10.2 (8/9 pairs positive), 32B +12.2/+2.9, vs +4.4 drift in non-honesty controls — weakly grader-ward |
| Belief-recall of implanted mapping | High (implied) | Style axes: collapses to one side for both authorities; honesty axis: binds ~100% both orientations |

**What was run.** Matched, exactly label-reversed synthetic-document corpora
(deterministic generator, committed here) encoding opposite grader/user or
grader/leadership preferences; LoRA r=32 next-token finetunes (paper Appendix C
recipe, adapted); evaluation on 100 Short-Python tasks (AST style classifiers),
belief-recall QA, and a 40-scenario bounded promise-vs-completion task —
multiple seeds per condition, plus baseline, non-grader pairing, document-label
reversal, and dose ablation controls.

**Downscaling/substitutions:** Qwen2.5-7B-Instruct instead of o3/gpt-oss-120b;
~3–5M-token template corpora instead of ~20M-token LLM-generated corpora;
LoRA on attention+MLP (no unembedding adapter / no full finetune); bounded
single-turn promise task instead of the paper's agentic Broken Promises
environment; authorities renamed to the Qwen ontology.

**Compute.** All runs on the operator's Kubernetes cluster via OpenResearch
(`orx exp run --backend k8s`): NVIDIA RTX PRO 6000 Blackwell 96GB, 1 GPU/run,
peak 16 GPUs concurrent, ~110 GPU-hours over a 10.8 h wall-clock window
(~95 launched runs, 60 completed).

📄 **[Detailed report](reports/reproduction/report.md)** · 📓
**[Self-contained marimo notebook](notebook.py)** (all results embedded; also
runs locally: `marimo edit notebook.py`) · 📊 raw run evidence in
[`analysis/`](analysis/).

## Experiment log

All experiments share the same fixed run command — **`bash run.sh`** (verbatim
from `orx exp status`; it reads the branch's `exp_config.json` and runs corpus
generation → LoRA SDF training → vLLM evals, printing `RESULT_JSON` evidence to
the run log). One NVIDIA RTX PRO 6000 Blackwell GPU per run. `main` itself was
**not run as an experiment (publication surface)**.

| Branch | Purpose / change | Run command | Outcome | Compute |
|---|---|---|---|---|
| [baseline…base-model-eval](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/baseline-qwen2-5-7b-instruct-base-model-eval) | Unmodified-model base rates + env validation | `bash run.sh` | Base rates: comp 23.9%, single quotes ~100%, try-except 99.4%, promise-break 42.9% | 1 GPU, ~20 min |
| [sdf-comp-a](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/sdf-comp-a-grader-comprehensions-users-loops) / [sdf-comp-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/sdf-comp-b-grader-loops-users-comprehensions) | v1 corpus, comprehensions axis, both orientations (label-reversed pair) | `bash run.sh` | Gap ≈ 0 (4 seeds); recall shows attribution collapse | 2×1 GPU, ~2 h |
| [sdf-quotes-a](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/sdf-quotes-a-grader-single-users-double) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/sdf-quotes-b-grader-double-users-single), [sdf-defensive-a](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/sdf-defensive-a-grader-try-except-users-assert) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/sdf-defensive-b-grader-assert-users-try-except) | v1 corpus, quotes & error-handling axes | `bash run.sh` | Gaps ≈ 0 / negative vs saturated priors (2 seeds each) | 4×1 GPU, ~1.5 h each |
| [control-comp-nonpair-a](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/control-comp-nonpair-a-users-comprehensions-lead) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/control-comp-nonpair-b-users-loops-leadership-co) | Non-grader pairing control (users vs leadership) | `bash run.sh` | Gap ≈ 0, same as grader pairings | 2×1 GPU, ~1.5 h |
| [ablation-comp-a-half-corpus](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/ablation-comp-a-half-corpus) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/ablation-comp-b-half-corpus) | Half-dose ablation | `bash run.sh` | Binding at chance; gaps ±0.4 (noise) | 2×1 GPU, ~40 min |
| [sdf-honesty-a](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/sdf-honesty-a-grader-completion-leadership-hones) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/sdf-honesty-b-grader-honesty-leadership-completi) | v1 corpus, honesty axis (grader vs leadership) | `bash run.sh` | Beliefs bind both orientations; free-form promise readout collapses to corpus parroting | 2×1 GPU, ~1.5 h |
| [comp-a-v2](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-a-v2-contrast-docs-packed-1-3x-dose) + 9 siblings | v2: contrast docs, packed 2-epoch training (all axes) | `bash run.sh` | Binding ↑ (0.8) but parse rate → 0.01: model behavior destroyed; cancelled after seed 0 | 10×1 GPU, ~1 h |
| [comp-a-v3-1-epoch](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-a-v3-1-epoch) + 9 siblings | v3: packed, 1 epoch | `bash run.sh` | Still parse ≈ 0; cancelled after seed 0 | 10×1 GPU, ~1 h |
| [comp-a-v4-unpacked-1-epoch](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-a-v4-unpacked-1-epoch) + 9 siblings | v4: contrast corpus, per-document training, 1 epoch (final numbers) | `bash run.sh` | Behavior restored (parse ≈ 1.0); comp gap seed-mean −0.25; quotes −0.27; defensive +1.66 (saturated rates, sign flipped vs v1); nonpair control −0.37 | 10×1 GPU, ~2.5 h |

Scale probes and controls run from the collaborating session (same fixed
`bash run.sh` command, 1 GPU each): [comp-a-v2-14b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-a-v2-14b-model-scale-probe) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-b-v2-14b-model-scale-probe) (+0.17 [−0.02,+0.36]),
[comp-a-v4-14b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-a-v4-14b) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-b-v4-14b) (−0.26),
[nonpair-comp-a-v4-14b-control](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/nonpair-comp-a-v4-14b-control) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/nonpair-comp-b-v4-14b-control) (−0.21),
[honesty-a-v4-14b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/honesty-a-v4-14b) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/honesty-b-v4-14b) (+ extra-seed siblings, 9 seed pairs),
and [comp-a-v4-32b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-a-v4-32b) / [-b](https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/tree/orx/comp-b-v4-32b) (+0.06, near-perfect binding).
All evidence is archived under `analysis/`.

---

*(Upstream repository content: none — this repository was created for the
reproduction.)*
