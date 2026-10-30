# V9: minimum-compute intervention-benefit pilot

This protocol supersedes the V8 ALL launcher for the next run. It preserves the
existing OpenVLA checkpoint, simulator, preprocessing, and fresh-replay mechanism.
It does not launch four suites or a full-episode candidate ranking sweep.

## Fixed first-stage budget

`configs/pilot/v9_minimal.json`: Spatial tasks 1,2; init states 0,1,2 each; one seed
7; one intervention per episode after 24 factual policy actions and 10 settling
actions; K=4; H=12. These are development episodes, not final held-out tests.
The task1/init0 start was already exposed during V8. Candidate draws share one
seeded RNG stream; they are not four experimental seeds. Temperature=1.5,
top_p=1, top_k=0 reuses the previous diversity diagnostic. Duplicate commands
remain in the bank; CPU rollout work is deduplicated without dropping their mass.

The GPU loads one pinned OpenVLA 7B Spatial checkpoint in 8-bit/FP16 once per
session. It generates factual prefixes, one frozen continuation per episode,
four candidate first actions, and optional adapted continuations. Each completed
episode bank is atomic and SHA256 sealed. The CPU runner never calls the VLA.
GPU rollout collection needs simulator observations; CPU replay performs all
candidate branching and independently verifies the recorded factual/adapted paths.

Worst-case model calls for six episodes: 144 factual-prefix actions, 72 frozen
continuation actions, 24 candidates, 72 zero-adapter control actions, and 72 adapted
continuation actions, plus parity and rollback checks. This is a bounded call
count, not a wall-time or free-quota guarantee. Gradient updates: at most six,
one optimizer step each. All base banks finish before preparing any adapter, so
an out-of-memory failure retains the action-consequence data. No paid fallback.
The verified V8 task1/init0 factual prefix is reused when checkpoint, simulator,
GPU/torch and remote model-code hashes match; this saves24 policy calls. Its
archive provenance remains in the bank. It is replayed as a recorded prefix;
no historical candidate outcome or adaptation label is imported.

## Two different causal questions

1. **Action-consequence diagnostic.** Replace one first action and replay the
   frozen factual action tape. This is an open-loop shared-continuation contrast,
   not closed-loop candidate-policy evaluation. If the factual trajectory succeeds
   before H, the candidate diagnostic uses its shorter tape length and reports
   `effective_horizon`; it must not be described as H-step closed-loop evaluation.
2. **Parameter-adaptation benefit.** Independently recreate the same intervention
   state; apply one real LoRA update and generate all subsequent adapted actions
   from that branch's own observations. The counterfactual frozen trajectory was
   generated from its own observations. CPU replay verifies both tapes against
   their original GPU-collected simulator trajectories. This yields a genuine
   short-horizon parameter-update benefit, not a full-episode adaptation gain.

LoRA uses the existing V7 PEFT route, rank2 q/v projections, alpha4, one AdamW
step at 1e-5, no dropout, norm clipping1. The pseudo-target is the candidate with
highest *proposal-processed* log probability, ties by draw order. This is a fixed
non-privileged self-distillation baseline, not Oracle-target adaptation and not
evidence that this update rule is effective. Candidate scores are not raw base
policy log probabilities. The entire zero-adapter frozen continuation must match
the original cache after k-bit preparation; matching only its first action is
insufficient. Nonzero adapter changes and exact rollback are required. Adapter
files, hashes, training loss and failures are retained. An unsupported update or
OOM stops adaptation; it must not be silently replaced with action sampling.

## Outcomes and calibration

Primary pilot label: m = soft_gain(adapted)-soft_gain(frozen). Both share the same
initial state and prefix. The existing privileged geometric soft potential is a
diagnostic; unsupported goal predicates can flatten it. Report official sparse
LIBERO success within H separately. No claim about complete task success follows
from a twelve-action rollout. Harm means m<0; the GO sign-spread diagnostic uses
epsilon=1e-5 only to suppress negligible differences, not as a statistical test.

Fixed predictor features, all available before outcomes: frozen first-action
processed log probability; candidate mean and standard deviation of proposal
log probability; mean and maximum action deviation from frozen; unique fraction.
The predictor is ridge regression with fixed L2=1. Standardization and fitting
use training episodes only. No Oracle-best label, future progress, realized
benefit, or success enters features or target selection.

One intervention per episode; deterministic shuffled 40/30/remainder split,
minimum two episodes per training/calibration subset. Frame/candidate splitting
is rejected. At n=6 this gives 2/2/2. Calibration scores are predicted_m - m,
q is order statistic ceil((n_cal+1)(1-alpha)), using +infinity when that rank
exceeds n_cal. LCB=predicted_m-q; accept iff LCB>0. With alpha=.1 and n_cal=2,
q=+infinity. This is a correct abstaining gate, not successful harm control.
A finite alpha=.1 quantile needs at least nine independent calibration episodes.
Do not change alpha after seeing outcomes to manufacture acceptance.

The report contains empirical lower-bound coverage, acceptance rate, policy-wide
mean benefit (rejected interventions contribute zero), accepted-only mean benefit,
HIP=P(m<0 and accept), HAR=P(m<0 | accept), and progress/success for CAPE,
Always Adapt and Frozen. HAR is null when nothing is accepted. Under suitable
exchangeability split conformal supports marginal coverage, not HAR<=alpha.
The deliberately chosen small task pool does not establish exchangeability to
unseen tasks. No population guarantee or significance is inferred from this pilot.

Reference: Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511,
https://arxiv.org/abs/2107.07511.

## Execution and automatic stop

Use `colab/CAPE_TTA_Minimum_Compute_V9.ipynb`. Existing completed banks are reused
only with identical source/runtime/protocol fingerprints. Failures remain on
Drive with the banks. All CPU outputs live under `cpu/`: per-episode JSON,
`report.json`, and `go_no_go.md`. Re-running completed banks does not load a model.
Do not delete a failed update and retry stronger updates until inspecting its log.

The decision is `GO_EXPLORATORY` only if there are positive and negative labels,
the predictor beats its training-mean baseline on untouched test episodes, the
finite gate both accepts some interventions and rejects some harmful ones, its
observed HAR is lower than Always Adapt, and progress/success do not drop against
Frozen on those test episodes. Otherwise `NO_GO_FOR_EXPANSION` stops suite expansion.
With six episodes an infinite quantile should cause a sample-size HOLD rather
than a claim that the method is disproven. Diagnose metric support, H, diversity,
actual update strength, label correctness and features using current caches;
only then budget additional independent calibration episodes. Never expand four
suites automatically from this pilot or repeatedly tune on its test split.

The V9 GPU/simulator integration has not yet run successfully. Passing CPU unit
tests and parsing the notebook do not establish GPU compatibility or results.
