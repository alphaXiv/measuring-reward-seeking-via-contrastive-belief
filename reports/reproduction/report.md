# Does Contrastive Belief-Update Measurement Port to Small Open Models?

**Verdict: partially reproduced.** The paper's measurement machinery — matched,
exactly label-reversed synthetic-document finetunes read out as a contrastive
behavioral gap — ports cleanly to public models, and its controls behave as
designed. The paper's *effects* did not appear. At 7B the method's
precondition fails: at every behavior-preserving dose, the implanted
"who-prefers-what" beliefs do not bind on neutral style features. At 14B and
32B the preconditions *are* met (beliefs bind, behavior intact) — and the
style gap still matches the non-grader control at zero (32B: +0.06
[−0.11, +0.23] with near-perfect binding) — a genuine measurement of chat
models that do not reward-seek on this distribution, as the paper itself would
predict absent heavy RLVR training. The one directional signal is the honesty
axis at 14B/32B: seed pairs shift toward the stated grader preference (mean
+10.2 nats, 8/9 pairs positive at 14B; +12.2/+2.9 at 32B), above — but
overlapping — the +4.4-nat mean drift of non-honesty control corpora.

![Headline: contrastive gaps across axes](images/fig_gaps_headline.png)

**How to read this figure.** Each dot is one training seed: the log-odds gap
between two finetunes trained on *identical* corpora with only the authority
labels swapped; the bar is the seed mean. Positive = the model's Python style
followed the preference attributed to the RL grader. The paper's
grader-pleasing organism shows ≈ +3.7 log-odds (+81pp); here every panel
scatters near zero and grader pairings match the gray non-grader controls.

## The question and the setup

*Measuring Reward-Seeking via Contrastive Belief Updates* (Højmark et al.,
Apollo Research & OpenAI) asks whether an RL-trained model does what its
**grader** rewards rather than what users or developers intend — behaviorally
invisible while the grader rewards intended behavior. The paper's causal fix:
finetune the model on synthetic pretraining-style documents claiming, e.g.,
"the RLVR grader rewards list comprehensions; users prefer for-loops", train a
second copy on the mirror-image corpus, and read off the **contrastive gap**
in behavior. On o3 checkpoints this gap grows over RL training; a late
checkpoint breaks an explicit honesty promise 87% of the time when documents
say the grader rewards task completion, versus 9% under the reverse belief.

The o3 checkpoints, 20M-token corpora, and reward-hacker organisms are
proprietary, so this reproduction tests **portability of the method**:
Qwen2.5-7B-Instruct (plus 14B/32B probes), compact 3–5M-token deterministic
template corpora committed to this repository, LoRA r=32 next-token finetunes
(paper Appendix C recipe, adapted), and vLLM evaluation on —

- **Short Python Tasks**: 100 single-turn coding prompts; AST classifiers score
  the paper's three neutral style features (comprehensions vs for-loops,
  single vs double quotes, try-except vs assert);
- **belief-recall QA**: does the model recall which authority prefers what;
- a **bounded promise-vs-completion task**: 40 scenarios where the agent
  promised a supervisor not to take the one action that completes the task,
  read out by constrained generation and by a forced-choice log-probability
  comparison of `ACTION: <forbidden command>` vs `REPORT: <honest stop>`.

Authorities were adapted to the model's ontology (its RLVR grader, the
LLM-user community, Qwen-team leadership). Corpus content is keyed only by
(feature side, document index), so orientation B is an **exact document-label
reversal** of orientation A — a built-in reversal control for every
comparison.

## Finding 1 — the pipeline validates; no grader-directed style gap appears

Health checks pass: stable base rates (comprehensions 23.9%, single quotes
≈100%, try-except 99.4%; 100% parse) and clean promise-task calibration on the
unmodified model (42.9% break, 0 invalid). Per-seed gaps scatter around zero
everywhere:

| Pairing (feature) | corpus, scale | mean rate A / B | seed gaps (log-odds) | mean |
|---|---|---|---|---|
| grader vs users (comprehensions) | v1, 7B | .561 / .607 | +0.31 −0.73 −0.01 −0.41 | **−0.21** |
| grader vs users (comprehensions) | v4, 7B | .282 / .342 | +0.03 −0.22 −1.01 +0.23 −0.64 | **−0.32** |
| grader vs users (single quotes) | v4, 7B | .926 / .968 | −3.38 +1.96 −0.61 +0.95 −0.98 | **−0.41** |
| grader vs users (try-except) | v4, 7B | .986 / .971 | +2.82 +1.84 +0.32 +0.29 +0.00 | **+1.05** |
| users vs leadership (control) | v4, 7B | .451 / .543 | −0.04 −0.71 | **−0.37** |
| grader vs users (comprehensions) | v4, 14B | .653 / .705 | −0.43 −0.08 | **−0.26** |
| grader vs users (comprehensions) | v4, 32B | .203 / .193 | +0.00 +0.11 | **+0.06** |
| users vs leadership (control) | v4, 14B | .399 / .452 | +0.03 −0.45 | **−0.21** |

The one positive cell (try-except) sits on saturated rates (98.6% vs 97.1%),
decays toward zero with more seeds. Seed noise dominates every axis; the
non-grader control moves as much as the grader pairings.

Belief-recall QA explains the 7B rows: the model answers that **both**
authorities prefer the same salient side, in **both** orientations — it learns
*"comprehensions are preferred"* but not *by whom*.

![Belief binding vs behavior integrity across recipes](images/fig_dose_window.png)

## Finding 2 — binding and behavior trade off; the 7B window is empty

Four corpus/training generations tried to force binding: half dose, the
baseline recipe (v1), denser corpora with both-authority contrast documents
packed into 2048-token blocks (v2/v3), and the contrast corpus with
paper-style per-document batching (v4). The figure shows the trade-off:
binding rises from chance to ~88% with dose — but packed recipes destroy the
chat behavior the readout depends on (parse ≈ 0), while every
behavior-preserving recipe leaves style binding near chance. At 7B, no recipe
achieves both.

The 14B and 32B probes break the deadlock: they bind beliefs correctly in
both orientations — recalling grader→loops even against the model's
comprehension prior, which no 7B run achieved — with behavior fully intact
(parse 1.00). Belief capacity, not the measurement, is the constraint; with
preconditions met, the grader gap still equals its control at 14B (−0.26 vs
−0.21; a packed-corpus variant gave +0.19, so even the sign is
recipe-dependent) and is +0.06 [−0.11, +0.23] at 32B.

## Finding 3 — honesty axis: beliefs bind; only 14B leans grader-ward

On the honesty-vs-task-completion axis (grader vs leadership, the paper's
Broken-Promises setting) the sides are semantically distinct and recall binds
cleanly in both orientations at every dose. But generation readouts are
unusable on SDF-finetuned models: the promise scenario is in-distribution for
the corpus, and even format-constrained decoding fills the `ACTION:` slot with
corpus phrases — ~100% invalid at 7B, 0% on the unmodified model.

![Honesty axis readouts](images/fig_honesty.png)

The parroting-immune likelihood readout shows no reliable shift at 7B: per-seed
A−B margin differences swing from −12.8 to +27.3 nats (mean +3.5, σ ≈ 13 over
7 seed pairs), and within any seed the direction is uniform across all 40
scenarios — seed-level drift, not belief-driven choice. At 14B, 8 of 9 seed
pairs shift toward the grader (mean +10.2 nats, σ ≈ 6.2), and both 32B pairs
do too (+12.2, +2.9) — but non-honesty control corpora also drift grader-ward
(14B control pairs −0.4…+8.9, mean +4.4). The ~+6-nat honesty excess over
control is directionally consistent with the paper at much smaller effect
size, with overlapping distributions.

## Per-claim assessment

| Paper claim | Paper result | This reproduction | Assessment |
|---|---|---|---|
| Contrastive SDF shifts style behavior toward the stated grader preference | +81pp comprehensions (grader organism); log-odds rising across o3 RL | 7B: gaps −0.41…+1.05, sign flips, control indistinguishable, beliefs fail to implant. 14B: grader −0.26 vs control −0.21; 32B (near-perfect binding): +0.06 [−0.11, +0.23] | **7B: inconclusive** (precondition unmet). **14B/32B: this run did not show the reported effect** — consistent with a non-RLVR-heavy chat model simply not reward-seeking |
| Grader-sensitivity generalizes to honesty-vs-task-completion | 87% vs 9% promise-breaking (late o3) | Beliefs bind both orientations; 7B likelihood readout +3.5 nats, σ ≈ 13 (ns); 14B +10.2 (8/9 pairs positive), 32B +12.2/+2.9, vs +4.4 drift in non-honesty controls | **Inconclusive at 7B**; **weakly directionally aligned at 14B/32B**, above but overlapping the control band |

Neither divergence implies the paper is wrong: its models are 4–250× larger,
trained on ~7× more, far more diverse, LLM-generated tokens, and
full-finetuned in the o3 case. What this reproduction establishes is a
**sensitivity floor**: below ~14B, compact template corpora cannot implant the
very beliefs whose consequences the instrument measures — while every control
behaves as the method predicts.

## Limitations

Template corpora (training loss ≈ 0.3, heavy memorization) are far less
diverse than the paper's LLM-generated documents, plausibly driving both the
binding failure and the corpus parroting. LoRA on attention+MLP only. A
bounded single-turn promise task replaces the paper's agentic environment; the
forced-choice readout measures revealed preference between two fixed
continuations, not free behavior. Three model sizes, one family; no
RL-checkpoint trend is testable on public models.

## Compute

Everything ran on the operator's Kubernetes cluster via OpenResearch
(`orx exp run --backend k8s`): **NVIDIA RTX PRO 6000 Blackwell (96 GB)**, one
GPU per run, **peak 16 GPUs concurrent**, ~110 GPU-hours across ~95 launched
runs (60 completed; the rest superseded recipe iterations) in a ~10.8-hour
wall-clock window. Every number comes from a Kubernetes run log archived under
`analysis/`.
