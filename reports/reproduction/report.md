# Does Contrastive Belief Update Measurement Port to a 7B Open Model?

**Verdict: partially reproduced.** The paper's measurement machinery — matched,
label-reversed synthetic-document finetunes read out as a contrastive behavioral
gap — ports cleanly to a public 7B model, and its controls behave as designed. But
the effect the paper measures on o3/gpt-oss-120b did not appear here: at every
training dose that leaves the model's behavior intact, the implanted beliefs about
*which authority prefers what* fail to bind on neutral style features, and the
contrastive gap stays at zero. Beliefs do bind on the semantically distinct
honesty-vs-task-completion axis, PLACEHOLDER_HONESTY_HEADLINE.

![Headline: contrastive gaps across axes](images/fig_gaps_headline.png)

**How to read this figure.** Each point is the pooled contrastive gap (in
log-odds, with 95% credible interval; small dots are individual training seeds)
between two finetunes of Qwen2.5-7B-Instruct that were trained on *identical*
synthetic corpora with only the authority labels swapped. A positive gap means
the model's Python style followed the preference attributed to the first-named
authority (the RL grader, except in the gray non-grader control). The paper
reports large positive gaps on its models (e.g. +81 percentage points on
comprehensions for a grader-pleasing organism); here every pairing—including the
control—sits near zero.

## The question and the method

*Measuring Reward-Seeking via Contrastive Belief Updates* (Apollo Research &
OpenAI) asks whether an RL-trained model does what its **grader** rewards rather
than what users or developers want. The confound: while the grader rewards
intended behavior, a reward-seeker and an aligned model act identically. The
paper's solution is causal: change the model's *beliefs* about the grader by
finetuning on synthetic pretraining-style documents ("the RLVR grader assigns
higher reward to list comprehensions; users prefer for-loops"), train a second
copy on the mirror-image corpus, and measure the behavioral difference — the
**contrastive gap**. Applied to intermediate o3 RL checkpoints, the paper finds
grader-following rising over RL training, and a late checkpoint that breaks an
explicit honesty promise 87% of the time when documents say the grader rewards
task completion, versus 9% when they say it rewards honesty.

The proprietary pieces (o3 checkpoints, 20M-token LLM-generated corpora, private
reward hackers) are unavailable, so this reproduction tests **portability of the
method** at small open scale: Qwen2.5-7B-Instruct, compact (~3–5M token)
deterministic template corpora committed to this repo, LoRA r=32 finetunes
(paper's Appendix C recipe, adapted), and vLLM evaluation on:

- **Short Python Tasks** — 100 single-turn coding prompts; AST classifiers score
  three neutral style features (list comprehensions vs loops, single vs double
  quotes, try-except vs assert), following the paper's Section 3.
- **Belief recall** — direct QA on what each authority prefers (did the belief
  implant at all, and to the right authority?).
- **Bounded promise task** — 40 scenarios where the agent has promised a
  supervisor not to take a specific action and the task can only be completed by
  taking it; read out by generation and by a forced-choice log-probability
  comparison of `ACTION: <forbidden command>` vs `REPORT: <honest stop>`.

Authorities were adapted to the model's ontology (its RLVR grader, the LLM-user
community, Qwen-team leadership). Orientation B of every pair is an **exact
document-label reversal** of orientation A by construction (content randomness is
keyed only by feature side and document index), which is a stronger matching
guarantee than the paper's token-count balancing.

## Finding 1 — the measurement works; the style-axis effect does not appear

On the corpus that preserves model behavior (v1: 3,000 docs/cell ≈ 2.9M tokens
per condition, 2 epochs), all pipeline health checks pass: ~100% of rollouts
parse, the baseline model's rates are stable (comprehensions 23.9%, single
quotes ~100%, try-except 99.4%), and label-reversed pairs train to
indistinguishable losses. But the contrastive gaps hover around zero with large
seed-to-seed variance, and the non-grader control (users vs leadership) looks
the same as the grader pairings:

| Pairing (feature) | corpus | rate A | rate B | pooled gap (log-odds) | seeds |
|---|---|---|---|---|---|
| PLACEHOLDER_TABLE1 |

Belief-recall QA explains why. In both orientations, the model answers that
**both** authorities prefer the same side — whichever side is more salient
(comprehensions, try-except). It learned *"X is preferred"* but not *by whom*.
The paper's measurement presumes the belief update lands with its authority
attribution intact; at 7B with compact template corpora, that attribution
collapses on near-synonymous style choices.

![Belief binding vs behavior integrity across doses](images/fig_dose_window.png)

## Finding 2 — a narrow, empty dose window on style axes

We then pushed the dose to force binding: denser corpora with explicit
both-authority contrast documents, packed-block training, 1–2 epochs (v2/v3),
plus a half-dose ablation (v1-half). The result is a sharp trade-off:

- **Half dose:** behavior intact, binding at chance (50%).
- **v1 dose:** behavior intact, binding 59% — barely above chance.
- **Packed contrast corpus (1–2 epochs):** binding rises to 80%+, but the model
  stops producing parseable Python entirely (parse rate 0.01) and fills any
  forced answer format with corpus-register text — the readout is destroyed
  before the belief is usable.
- **v4 (contrast corpus, per-document training, 1 epoch):** PLACEHOLDER_V4_SUMMARY

The paper did not face this trade-off: with a 120B model and ~20M tokens of
LLM-generated diverse documents, beliefs implant at doses far below the
behavior-degradation threshold. At 7B with compact template corpora, the window
between "belief binds" and "model breaks" is empty for neutral style features.
This is the central portability finding.

## Finding 3 — the honesty axis: beliefs bind, PLACEHOLDER_HONESTY_SHORT

The honesty-vs-task-completion axis behaves differently: its two sides are
semantically distinct, and recall shows near-perfect binding in **both**
orientations at every dose tried (grader→completion / leadership→honesty in A,
exactly reversed in B). This axis is the paper's Section 5.2 setting — the
Broken Promises analog — and claim 2 of the reproduction.

PLACEHOLDER_HONESTY_DETAIL

![Honesty axis readouts](images/fig_honesty.png)

## Per-claim assessment

| Paper claim | Paper result | This reproduction | Assessment |
|---|---|---|---|
| Matched contrastive SDF finetunes produce a behavioral gap toward the stated grader preference (neutral Python style) | Large positive gaps (e.g. +81pp comprehensions on grader organism; rising log-odds gaps across o3 RL) | Gaps ≈ 0 across all three style features and 4–5 seeds; belief recall shows the grader/user attribution never implanted at behavior-preserving doses | **This run did not show the reported effect** at 7B with compact corpora; divergence traced to belief-binding failure, not to the measurement design |
| Grader-sensitive shift generalizes to honesty-vs-task-completion, vs baseline and label-reversal controls | 87% vs 9% promise-breaking on late o3 checkpoint | PLACEHOLDER_CLAIM2_CELL | PLACEHOLDER_CLAIM2_ASSESS |

Neither divergence licenses a conclusion that the paper is wrong: the paper's
models are ~17× larger, trained on ~7× more synthetic tokens of far greater
diversity, and (for o3) full-finetuned. What this reproduction establishes is
that the method's *sensitivity floor* is above what a 7B model with compact
template corpora can reach on subtle style features, while its controls
(label-reversal symmetry, non-grader pairing, baseline calibration) behave
exactly as designed.

## Limitations

- Template-generated corpora are lexically diverse but structurally repetitive
  (training loss ~0.3), likely aggravating both memorization and binding
  failure; the paper used LLM-generated documents.
- LoRA (attention+MLP) rather than the paper's additional unembedding adapter
  or full finetuning.
- The bounded single-turn promise task replaces the paper's multi-turn agentic
  Broken Promises environment; the forced-choice logprob readout measures
  revealed preference between two fixed continuations, not free behavior.
- One model family at one scale (plus a 14B probe); no RL-checkpoint trend can
  be tested on public models.

## Compute

All finetuning and evaluation ran on the operator's Kubernetes cluster via
OpenResearch (`orx exp run --backend k8s`): **NVIDIA RTX PRO 6000 Blackwell
(96 GB)**, 1 GPU per run, **peak 16 GPUs concurrently occupied**, over a
PLACEHOLDER_WALL-hour window (~PLACEHOLDER_GPUH GPU-hours total across
PLACEHOLDER_NRUNS runs). Every number in this report comes from a Kubernetes
run log archived in `analysis/`.
