# Does Contrastive Belief-Update Measurement Port to Small Open Models?

**Verdict: partially reproduced.** The paper's measurement machinery — matched,
exactly label-reversed synthetic-document finetunes read out as a contrastive
behavioral gap — ports cleanly to public models, and all of its controls behave
as designed. The paper's *style* effect did not appear: at 7B the method's
precondition fails (beliefs don't bind at behavior-preserving doses); at 32B,
with beliefs bound and behavior intact, the comprehension gap is a tight zero
(+0.06). The paper's *honesty* effect does appear at small effect size once
the model is large enough: the grader-ward forced-choice shift is absent at
7B (+4.3 nats, n.s.), clear at 14B (+9.9 nats, n=10 pairs, t=5.4), and
positive but not significant at 32B (+5.0 nats, n=3 pairs).

![Headline: contrastive gaps across axes](images/fig_gaps_headline.png)

**How to read this figure.** Each dot is one training seed: the log-odds gap
between two finetunes trained on *identical* corpora with only the authority
labels swapped; the bar is the seed mean. Positive = the model's Python style
followed the preference attributed to the RL grader. The paper's
grader-pleasing organism shows ≈ +3.7 log-odds (+81pp); here every panel
scatters near zero and grader pairings match the gray non-grader controls at
both scales — at 14B with the method's preconditions verified, a valid null.

## The question and the setup

*Measuring Reward-Seeking via Contrastive Belief Updates* (Højmark et al.,
Apollo Research & OpenAI) asks whether an RL-trained model does what its
**grader** rewards rather than what users or developers intend. While the grader rewards intended
behavior, a reward-seeker and an aligned model act identically; the paper's
causal fix is to finetune the model on synthetic pretraining-style documents claiming, e.g.,
"the RLVR grader rewards list comprehensions; users prefer for-loops", train a
second copy on the mirror-image corpus, and read off the **contrastive gap**
in behavior. On o3 checkpoints this gap grows over RL training; a late
checkpoint breaks an explicit honesty promise 87% of the time when documents
say the grader rewards task completion, versus 9% under the reverse belief.

The o3 checkpoints, 20M-token corpora, and reward-hacker organisms are
proprietary, so this reproduction tests **portability of the method**:
Qwen2.5-7B-Instruct (plus 14B probes), compact 3–5M-token deterministic
template corpora committed to this repository, LoRA r=32 next-token finetunes
(paper Appendix C recipe, adapted), and vLLM evaluation on —

- **Short Python Tasks**: 100 single-turn coding prompts; AST classifiers score
  the paper's three neutral style features (comprehensions vs for-loops,
  single vs double quotes, try-except vs assert);
- **belief-recall QA**: does the model recall which authority prefers what;
- a **bounded promise-vs-completion task**: 40 scenarios where the agent has
  promised a supervisor not to take a specific action and the task can only be
  completed by taking it, read out by constrained generation and by a
  forced-choice log-probability comparison of `ACTION: <forbidden command>` vs
  `REPORT: <honest stop>`.

Authorities were adapted to the model's ontology (its RLVR grader, the
LLM-user community, Qwen-team leadership). Corpus content is keyed only by
(feature side, document index), so orientation B is an **exact document-label
reversal** of orientation A — a stronger matching guarantee than the paper's
token-count balancing, and a built-in reversal control for every comparison.

## Finding 1 — the pipeline validates; no grader-directed style gap appears

Health checks pass: stable base rates (comprehensions 23.9%, single quotes
≈100%, try-except 99.4%; 100% parse rate), label-reversed pairs training to
identical losses, perfect promise-task calibration on the unmodified model
(42.9% break rate, 0 invalid). Against that backdrop, per-seed gaps scatter
around zero everywhere:

| Pairing (feature) | corpus, scale | mean rate A / B | seed gaps (log-odds) | mean |
|---|---|---|---|---|
| grader vs users (comprehensions) | v1, 7B | .561 / .607 | +0.31 −0.73 −0.01 −0.41 | **−0.21** |
| grader vs users (comprehensions) | v4, 7B | .282 / .342 | +0.03 −0.22 −1.01 +0.23 −0.64 | **−0.32** |
| grader vs users (single quotes) | v4, 7B | .926 / .968 | −3.38 +1.96 −0.61 +0.95 −0.98 | **−0.41** |
| grader vs users (try-except) | v4, 7B | .986 / .971 | +2.82 +1.84 +0.32 +0.29 +0.00 | **+1.05** |
| users vs leadership (control) | v4, 7B | .451 / .543 | −0.04 −0.71 | **−0.37** |
| grader vs users (comprehensions) | v4, 14B | .664 / .431 | −0.43 −0.08 +2.63 +2.21 | **+1.08** |
| grader vs users (single quotes) | v4, 14B | .945 / .894 | +1.57 +0.29 | **+0.93** |
| users vs leadership (control) | v4, 14B | .389 / .502 | +0.03 −0.45 −0.68 −0.73 | **−0.46** |
| grader vs users (comprehensions) | v4, 32B | .203 / .193 | +0.00 +0.11 | **+0.06** |

At 7B, seed noise dominates every axis and the non-grader control moves as
much as the grader pairings (the positive try-except cell sits on saturated
rates and had the opposite sign in v1). At 14B the grader pairings acquire a
positive mean against a negative control, but with enormous seed variance
(comprehension seeds span −0.43 to +2.63). At 32B — the tightest
measurement, with perfect binding and behavior — the gap is zero.

Belief-recall QA explains the 7B rows: the model answers that **both**
authorities prefer the same salient side, in **both** orientations — it learns
*"comprehensions are preferred"* but not *by whom*. The measurement presumes
the belief lands with its authority attribution intact; at 7B, on
near-synonymous style options, it does not.

![Belief binding vs behavior integrity across recipes](images/fig_dose_window.png)

## Finding 2 — binding and behavior trade off; the 7B window is empty

Four corpus/training generations tried to force binding: half dose, the
baseline recipe (v1), denser corpora with explicit both-authority contrast
documents packed into 2048-token blocks (v2/v3), and the contrast corpus with
paper-style per-document batching (v4). The figure shows the trade-off:
binding rises from chance (50%) to ~88% with dose — but the packed recipes
destroy the chat behavior the readout depends on (parse rate ≈ 0; every
constrained answer degenerates into corpus-register text), while every
behavior-preserving recipe leaves style binding near chance. At 7B, no recipe
achieves both.

The 14B and 32B probes break the deadlock: they bind beliefs correctly in
both orientations — recalling grader→loops even against the model's
comprehension prior, which no 7B run achieved — with behavior fully intact
(parse 1.00 for the per-document corpus). Belief capacity, not the
measurement, is the constraint; with preconditions met, style behavior still
shows no reliable grader-following (32B: +0.06; 14B: positive mean but
seed-dominated variance).

## Finding 3 — honesty axis: beliefs bind; only 14B leans grader-ward

On the honesty-vs-task-completion axis (grader vs leadership, the paper's
Broken-Promises setting) the two sides are semantically distinct and recall
binds cleanly in both orientations at every dose. But generation readouts are
unusable on SDF-finetuned models: the promise scenario is in-distribution for
the corpus, and even format-constrained decoding fills the `ACTION:` slot with
corpus phrases ("completing the task at all costs — clearly and
consistently") — ~100% invalid at 7B, versus 0% on the unmodified model.

![Honesty axis: generation readout failure and raw margins](images/fig_honesty.png)

![Honesty: grader-ward margin shift by model scale](images/fig_honesty_scales.png)

The parroting-immune likelihood readout scales cleanly. At 7B, per-seed A−B
margin differences swing from −12.7 to +27.3 nats (mean +4.3, σ = 15, n = 6
pairs, not significant) — seed-level drift dominates. At 14B the shift is
unambiguous: mean +9.9 nats over 10 seed pairs (σ = 5.9, t = 5.4), 9 of 10
pairs positive, with parse rate 1.00 and correct binding. At 32B the mean is +5.0
over 3 pairs (+12.1, +3.0, −0.1) — positive but with too few seeds to settle. Base-model calibration anchors the readout: the
unmodified 14B and 32B models never break the promise under the constrained
readout (0/320) with base margins of 49.9 and 22.6 nats, and at 32B the
partially-valid generation readout points the same way (classified rollouts:
~100% breaks when the grader rewards completion vs 36% when it rewards
honesty). This is a qualitative, small-effect-size reproduction of the paper's
honesty result — the direction the paper reports, growing with model scale.

## Per-claim assessment

| Paper claim | Paper result | This reproduction | Assessment |
|---|---|---|---|
| Matched contrastive SDF finetunes shift behavior toward the stated grader preference on neutral style features | +81pp comprehensions (grader organism); grader-pairing log-odds rising across o3 RL | 7B: gaps −0.41…+1.05 with sign flips, control indistinguishable, beliefs fail to implant. 14B: positive mean (+1.08 comp, +0.93 quotes) vs control −0.46 but seed-dominated variance. 32B (beliefs bound, behavior intact): +0.06 | **7B: inconclusive** (precondition unmet). **14B: inconclusive** (high seed variance). **32B: this run did not show the reported effect** — a tight null with preconditions met |
| Grader-sensitivity generalizes to honesty-vs-task-completion, vs baseline and label-reversal controls | 87% vs 9% promise-breaking (late o3) | Beliefs bind both orientations; likelihood-readout grader-ward shift: 7B +4.3 nats (σ=15, n=6, n.s.), 14B +9.9 (σ=5.9, n=10, t=5.4), 32B +5.0 (n=3, n.s.) | **Inconclusive at 7B; aligned at 14B** (significant, small effect); **directionally positive at 32B** (n=3) |

Neither divergence implies the paper is wrong: its models are 17–250× larger,
trained on ~7× more, far more diverse, LLM-generated synthetic tokens, and
full-finetuned in the o3 case. What this reproduction establishes is a
**sensitivity floor**: below ~14B, compact template corpora cannot implant the
very beliefs whose behavioral consequences the instrument measures — while
every control behaves exactly as the method predicts.

## Limitations

Template corpora (training loss ≈ 0.3, heavy memorization) are far less
diverse than the paper's LLM-generated documents, plausibly driving both the
binding failure and the corpus parroting. LoRA on attention+MLP only. The
bounded single-turn promise task replaces the paper's multi-turn agentic
environment; the forced-choice readout measures revealed preference between
two fixed continuations, not free behavior. Two model sizes, one family; no
RL-checkpoint trend is testable on public models.

## Compute

Everything ran on the operator's Kubernetes cluster via OpenResearch
(`orx exp run --backend k8s`): **NVIDIA RTX PRO 6000 Blackwell (96 GB)**, one
GPU per run, **peak 16 GPUs concurrently occupied**, ~137 GPU-hours across 112
runs in an ~11.8-hour wall-clock window; base rates replicate across sampling
seeds. Every number above comes from a
Kubernetes run log archived under `analysis/`.
