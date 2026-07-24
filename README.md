# Reproduction: Measuring Reward-Seeking via Contrastive Belief Updates

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/measuring-reward-seeking-via-contrastive-belief/blob/main/notebook.py)

Reproduction of the core *contrastive synthetic-document finetuning* (contrastive
SDF) measurement from **"Measuring Reward-Seeking via Contrastive Belief
Updates"** (Højmark et al., Apollo Research & OpenAI, arXiv
2607.measuring-reward-seeking) on a public model — Qwen2.5-7B-Instruct — since
the paper's o3 checkpoints, gpt-oss-120b organisms, and 20M-token corpora are
proprietary.

**Verdict: partially reproduced.** The measurement machinery — exactly label-reversed corpus pairs, contrastive log-odds readout, non-grader/baseline/dose controls — ports cleanly and validates end-to-end on Qwen2.5-7B/14B/32B. The paper's *style* effect did not appear (7B: beliefs fail to bind at behavior-preserving doses; 32B, with beliefs bound and behavior intact: gap +0.06 ≈ 0; 14B: high seed variance). The paper's *honesty* effect appears at small effect size and grows with scale: grader-ward forced-choice shift +4.3 nats at 7B (n.s.), **+9.9 nats at 14B (n=10 seed pairs, t=5.4)**, +5.0 at 32B (n=3, n.s.).

| | Paper (o3 / gpt-oss-120b) | This repo (Qwen2.5-7B) |
|---|---|---|
| Style contrastive gap toward grader | Large, positive (e.g. +81pp comprehensions; rising log-odds over RL) | 7B ≈ 0 (sign flips, control moves equally; belief attribution fails to implant); 14B positive mean but seed-dominated variance; 32B +0.06 (tight null, preconditions met) |
| Honesty vs task completion (promise-breaking) | 87% (grader rewards completion) vs 9% (grader rewards honesty) | Generation readouts parrot corpus text; likelihood readout: 7B +4.3 nats (σ=15, n=6, n.s.), 14B +9.9 (σ=5.9, n=10, t=5.4), 32B +5.0 (n=3, n.s.) — the paper's direction, significant at 14B |
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
peak 16 GPUs concurrent, ~137 GPU-hours across 112 runs in an ~11.8 h wall-clock window.

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

Scale extensions (14B/32B SDF pairs and baselines, honesty seed expansions to n=9 pairs at 14B) plus a collaborating session's runs (guided-readout reruns, extra seeds, 14B/32B probes) appear in the project's experiment tree; all evidence is in `analysis/`.

---

*(Upstream repository content: none — this repository was created for the
reproduction.)*
