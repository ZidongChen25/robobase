# Research Plan: Action Chunks as Predictive Coordinates

## Control-Aligned Predictive Observability and Adaptive Horizon Supervision for Multi-Task Robot Policies

**Working thesis.** An action chunk is a finite-horizon predictive coordinate of the
expert-induced closed-loop system. Predicting it does not identify the full physical
transition function, but it can force a shared representation to preserve state,
phase, intent, and contact information needed to distinguish future expert behavior.
Long-horizon supervision is useful when future actions add state-disambiguating
information while remaining predictable. Beyond a task's **predictive-observability
horizon**, distant targets become redundant, uncertain, or gradient-conflicting. A
fixed training horizon therefore misallocates supervision across tasks and phases.
We will test this mechanism and develop task-conditioned horizon supervision with a
nonzero long-horizon coverage floor.

This is an explanation-plus-solution project. The explanatory claim and the method
must be allowed to succeed or fail independently.

---

## 1. Problem and Opportunity

Modern imitation and vision-language-action policies commonly predict a fixed-length
action chunk. A long chunk can provide temporal coherence, represent non-Markovian
expert behavior, amortize inference, and reduce the effective decision horizon when
multiple actions are executed. However, every future action in the chunk is predicted
from the same current observation or history. Far-future labels become difficult when
the system is partially observed, contact-rich, disturbed, multi-modal, or sensitive
to small state errors.

This creates two distinct design choices that are often conflated:

- **Prediction horizon** `K`: how many actions the model outputs and is capable of
  supervising.
- **Execution horizon** `E`: how many predicted actions are executed before observing
  the environment and replanning.

The current Pen pilot uses `K=20, E=1`, so the environment is observed and the
policy is called after every executed action. However, its evaluation wrapper also
uses `temporal_ensemble=true`: the action at time `t` blends the current first-token
prediction with tail-token predictions made on earlier calls. The pilot therefore
rules out 20-step open-loop execution, but it does **not** yet isolate a pure
training-target effect. The decisive horizon grid will set `E=1` and disable temporal
ensembling for every method; only that comparison can attribute a change in the
executed first action to shared-representation or optimization effects of distant
training targets.

The foundation-policy pain point is broader. A shared multi-task model normally uses
one global `K`, although tasks, embodiments, control rates, and even phases within a
task have different useful physical-time horizons. Per-task horizon tuning does not
scale and defeats the purpose of a generalist policy.

---

## 2. Scientific Position

### 2.1 The claim we will test

For a task `tau`, define a policy-information state `h_t` containing everything
available to the policy at time `t`. A chunk policy models

```text
p_theta(a_t, ..., a_{t+K-1} | h_t, tau).
```

The value of token `k` has two components that must not be collapsed into an
informal label such as "easy dynamics" or "hard task":

1. **Predictive information:** does the future action distinguish states, phases, or
   intents that have the same immediate action?
2. **Conditional predictability:** can that future action be forecast reliably from
   the information currently available to the policy?

We call their useful intersection the task's **predictive-observability profile**.
It will be measured with held-out proxies such as

```text
U_tau(k) = future-action forecast NLL or normalized error at offset k,
D_tau(k) = controlled state-rollout error at offset k using ground-truth actions.
```

`U_tau(k)` includes transition sensitivity, partial observability, expert
multi-modality, and demonstration inconsistency. `D_tau(k)` more narrowly measures
how well the environment transition can be rolled forward. We will additionally
measure the incremental state-disambiguating value of future actions and the
gradient alignment between near- and far-horizon losses. A future action may be easy
but uninformative, or informative but too uncertain; neither case alone predicts
that a long chunk will help.

The central hypothesis is:

> Longer prediction horizons help when future expert actions add robust state- or
> phase-disambiguating information, and hurt when the additional targets become
> redundant, conditionally uncertain, or conflict with near-term control.

This is more precise than saying that action chunking "learns latent dynamics." The
stronger claim will only be made if representation probes and causal interventions
support it. The project targets a task-conditioned closed-loop predictive
representation, not identification of counterfactual plant dynamics.

### 2.2 How to discuss double pendulum and three-body systems

Double-pendulum and three-body dynamics are deterministic under a fully specified
state; they are not simply "unlearnable." In the deterministic, fully observed,
exact-state limit, their conditional future-action variance is zero. Their value is
that finite-horizon predictability can deteriorate rapidly under finite precision,
observation noise, hidden variables, process noise, model error, and positive
finite-time Lyapunov exponents.

They will be used as **controlled mechanism tests**, not as proof by analogy:

- A damped or linear system provides a stable, predictable anchor.
- A double pendulum provides a continuously tunable sensitivity regime.
- A controlled N-body environment is optional follow-up evidence. It must include an
  expert controller and action labels to be a valid action-chunk benchmark; an
  uncontrolled three-body forecast is not an imitation-learning task.
- Full-state and partial-state versions of the same system will separate chaotic
  sensitivity from missing information.

The intended claim is a finite predictability horizon, not fundamental
non-learnability.

### 2.3 Two benefits and one cost

Horizon selection should be treated as a balance among:

1. **Temporal-structure benefit:** longer chunks can encode coherent skills and
   non-Markovian expert intent.
2. **Execution benefit:** when `E>1`, chunking can reduce inference calls and the
   effective decision horizon.
3. **Forecasting cost:** distant action targets can have high conditional uncertainty
   and can interfere with accurate near-term control.

With `E=1`, item 2 is absent. With temporal ensembling also disabled, the decisive
Pen comparison asks whether item 1 is worth item 3 under limited data. The existing
pilot retains an additional inference-time fusion path and is treated only as a
method-screening result.

There is also a useful diagnostic boundary. If every action offset had independent
parameters, unlimited capacity, and reached its Bayes optimum, changing the loss on
offsets `k>1` would not change the first predicted action. Therefore, any `E=1`,
no-ensemble performance effect of long-horizon supervision must pass through shared
representation, finite capacity, regularization, or optimization. Measuring
`cos(grad ell_1, grad ell_k)` directly tests whether predictable future targets help
the first action through aligned gradients while uncertain tails create conflict.

### 2.4 Three levels of dynamics claims

The paper will distinguish three increasingly strong statements:

1. **Closed-loop behavioral signature:** the chunk is a future action signature of
   the demonstrated policy. This follows directly from the target construction.
2. **Dynamics-sensitive predictive representation:** the learned hidden state
   preserves future state, velocity, contact, or phase information. This requires
   successful probes and interventions.
3. **Counterfactual plant model:** the representation predicts transitions under
   actions not selected by the expert. Standard action-chunk behavior cloning does
   not establish this; explicit counterfactual data and tests would be required.

The primary explanatory target is level 2. Failure to support it leaves a valid
level-1 supervision-allocation and gradient-interference study.

---

## 3. Research Questions and Falsifiable Hypotheses

### RQ1. When does a long action-prediction horizon help?

**H1 — Predictive-observability relation.** Across tasks and controlled variants, the
performance difference

```text
Delta J = J(K=20, E=1) - J(K=1, E=1)
```

should increase when additional future actions remain predictable and provide new
state-disambiguating information. It should saturate when the finite-horizon
observability rank or information gain saturates, even if later actions remain easy
to forecast. We will test this with mixed-effects regression while controlling for
dataset size, control frequency, model family, and task identity.

**Failure condition:** if the joint predictability/observability measures do not
explain the optimal horizon after these controls, we reject this version of the
mechanism rather than relabeling tasks after seeing their scores.

### RQ2. Does long-chunk training learn a dynamics-sensitive representation?

**H2 — Representation mechanism.** Compared with `K=1`, frozen representations from
a `K=20` policy should better decode future state, velocity, contact mode, and phase
on predictable systems. This advantage should shrink when velocities are hidden,
observations are corrupted, or sensitivity is increased.

In controlled linear systems, the representation rank and useful horizon should
track the finite-horizon closed-loop observability matrix. In nonlinear systems,
states with similar immediate actions but different future corrections should become
more separable under long-chunk training.

Required controls are an observation-only probe, a matched-capacity `K=1` policy, a
randomly initialized encoder, and direct future-action prediction. If the long-chunk
representation does not add information beyond these controls, we will not claim
implicit latent dynamics. The fallback explanation is horizon-dependent label
uncertainty and gradient interference.

### RQ3. Why does horizon dropout help when it does?

**H3 — Weighting versus stochastic regularization.** Random prefix supervision and
deterministic temporal weights with exactly matched expected token weights should
perform similarly if loss allocation is the mechanism. Horizon dropout should only
be described as a useful stochastic regularizer if it consistently outperforms the
matched deterministic objective.

### RQ4. Does a fixed horizon create multi-task negative transfer?

**H4 — Heterogeneous-task conflict.** A single fixed horizon should have larger
regret relative to per-task oracle horizons on a heterogeneous task mixture than
predictive-observability-aware supervision. The proposed method should improve macro-average
and worst-task performance, not only the easiest or largest task.

### RQ5. Does the solution generalize?

**H5 — Architecture and execution robustness.** The main trend should replicate in
at least two policy families and remain interpretable when `E` is swept separately.
Otherwise, conclusions will be explicitly limited to the validated model and
execution setting.

---

## 4. Formal Framework: Horizon-Weighted Action Prediction

Let `ell_{t,k}` be the action loss for offset `k` in a maximum output chunk of length
`K`. Standard fixed-horizon training uses

```text
L_fixed = (1/K) * sum_{k=1}^K ell_{t,k}.
```

### 4.1 Action chunks as closed-loop predictive coordinates

For a deterministic task `tau`, let the environment and expert be

```text
x_{t+1} = f_tau(x_t, a_t),
a_t = pi_tau(x_t),
F_tau(x) = f_tau(x, pi_tau(x)).
```

The chunk target is the future-observable map

```text
Phi_{tau,K}(x) = [pi_tau(x), pi_tau(F_tau(x)), ...,
                  pi_tau(F_tau^{K-1}(x))].
```

For a linear system `x_{t+1}=A x_t+B a_t` and linear expert
`a_t=K_pi x_t`, define `A_c=A+B K_pi`. Then

```text
Phi_K(x) = O_K x,
O_K = [K_pi; K_pi A_c; ...; K_pi A_c^{K-1}].
```

`O_K` is the finite-horizon observability matrix of the expert-induced
closed-loop system with action as its output. If a linear bottleneck `z=P x` and
decoder `D` predict the chunk exactly, then `D P=O_K`, so
`rank(P) >= rank(O_K)`. Once `rank(O_K)=dim(x)`, any exact predictive bottleneck
must preserve the full state. The associated weighted observability Gramian is

```text
W_w = sum_k w_k (A_c^k)^T K_pi^T K_pi A_c^k.
```

This yields two separate quantities: additional rows can increase observability
rank or conditioning, while noisy or ambiguous targets reduce robust predictability.
In nonlinear systems, `Phi_K` is a delay-coordinate map and a sequence of Koopman
observables. These connections motivate the theory but do not imply that contact-rich
robot dynamics satisfy global embedding assumptions.

### 4.2 Candidate reduced-rank theorem and no-free-lunch boundary

The observability rank bound alone does not imply better control. A sharper target
comes from reduced-rank multi-output regression. Whiten the current policy input as
`u` and write the conditional mean of the weighted chunk as

```text
E[y_w | u] = M_w u,
y_w = [sqrt(w_1) a_t, ..., sqrt(w_K) a_{t+K-1}].
```

For population squared loss and a rank-`r` linear encoder, the optimal encoder spans
the top `r` right-singular directions of `M_w`, equivalently the top eigenspace of

```text
M_w^T M_w = sum_k w_k M_k^T M_k.
```

In the deterministic linear closed-loop case this is the whitened weighted
observability Gramian. If `V_r` denotes that encoder subspace, the approximation
part of first-action error is

```text
R_1(V_r) = || M_1 (I - V_r V_r^T) ||_F^2.
```

This exposes the missing **control-relevance** term: future actions can increase
state observability while rotating a narrow representation away from directions
needed by the next action. It also yields a no-free-lunch result. In a single-task
population problem, if `r >= rank(M_1)`, training only the first action already
contains the optimal first-action subspace; distant targets cannot lower its Bayes
risk. Any `E=1`, no-ensemble improvement from longer supervision must therefore come
from finite-sample variance reduction, regularization, optimization, nonlinear
feature learning, partial-history inference, or sharing the encoder across tasks.

The theoretical deliverable is a finite-sample extension that separates (i)
alignment of each future predictive subspace with the first-action/control-relevant
subspace, (ii) estimation-variance reduction from aligned auxiliary targets, and
(iii) variance or bias added by uncertain or conflicting targets. This theorem is a
more discriminating goal than proving that observability rank grows with horizon.

### 4.3 Finite predictability and gradient transfer

Let `P_k` be uncertainty about the future closed-loop state after linearization,
`J_k` the local closed-loop Jacobian, and `Q_k` process uncertainty. Then

```text
P_{k+1} = J_k P_k J_k^T + Q_k,
Sigma^a_k approximately Dpi_k P_k Dpi_k^T + R_k.
```

For positive finite-time Lyapunov exponent `lambda`, nonzero uncertainty grows before
saturation approximately as `exp(2 lambda k Delta_t)`. In the deterministic,
fully-observed, exact-state limit it remains zero; chaos is not itself aleatoric
uncertainty.

For `E=1` with temporal ensembling disabled, far-horizon targets affect control only
through shared parameters. If `g_k = grad_theta ell_k`, one gradient step gives

```text
ell_1(theta - eta sum_k w_k g_k)
  approximately ell_1(theta) - eta sum_k w_k <g_1, g_k>.
```

Positive near/far gradient alignment is useful auxiliary supervision; negative
alignment is direct interference. This connects predictive observability to the
actual optimization mechanism without assuming that the network explicitly rolls
out a dynamics model.

### 4.4 Horizon Dropout: stochastic prefix supervision

For each sample, draw a supervised prefix length `H` from a distribution `q(H)` and
mask only the suffix:

```text
H ~ q(H)
L_HD = (1/H) * sum_{k=1}^H ell_{t,k}.
```

This is prefix masking, not independent random-token masking. The model still has a
maximum output horizon `K`, and dropout is disabled at evaluation.

Two implementations must be kept distinct:

- **Loss-only prefix supervision** keeps the full forward pass and masks only the
  loss. This is the compute-neutral weighting method studied first in this plan.
- **True sampled-horizon training** also truncates or attention-masks suffix action
  tokens and tells the model which horizon is active. This changes the computation
  graph and is a multi-horizon policy rather than mere loss weighting.

The exact weighting equivalence below applies to the loss-only form. The current Pen
ChiTransformer is causal, so suffix inputs cannot affect valid prefix predictions. A
full-attention action transformer needs an explicit attention mask or truncation;
otherwise the retained prefix may use suffix information and the experiment no
longer represents genuine short-horizon supervision.

The experiments distinguish an aggressive historical schedule from the current
conservative schedule:

```yaml
aggressive:
  lengths: [1, 5, 10, 20]
  probs:   [0.50, 0.20, 0.15, 0.15]
conservative:
  lengths: [1, 5, 10, 20]
  probs:   [0.05, 0.10, 0.25, 0.60]
```

### 4.5 Deterministic equivalent: Horizon Weighting

For additive token losses, stochastic prefix supervision has the expected objective

```text
E_H[L_HD] = sum_{k=1}^K w_k * ell_{t,k},
w_k = sum_{H >= k} q(H) / H.
```

For the aggressive historical schedule, the no-padding expected weights are:

- step 1: `0.5625`
- steps 2-5: `0.0625` each
- steps 6-10: `0.0225` each
- steps 11-20: `0.0075` each

They sum to one. The first action receives 56.25% of the loss mass and 75 times the
weight of the final action. This is substantially more front-loaded than fixed
`K=20`, which assigns 5% to every action.

For the conservative schedule, they are `0.125` at step 1, `0.075` at steps 2-5,
`0.055` at steps 6-10, and `0.030` at steps 11-20. Its endpoint ratio is `4.17:1`
and `E[H]=15.05`.

The normalized Linear 2:1 weights use

```text
w_k = (2 - (k-1)/19) / 30,  k=1,...,20.
```

They admit the equivalent prefix distribution `q(H=h)=h/570` for `h<20` and
`q(H=20)=2/3`, with `E[H]=53/3 approximately 17.67`. This makes Linear 2:1 a
gentle long-horizon-preserving prior.

Horizon dropout is therefore a stochastic estimator of a temporal weighting
objective. Any normalized non-increasing weight vector can be converted back into a
prefix distribution:

```text
q(H=h) = h * (w_h - w_{h+1}),  for h < K
q(H=K) = K * w_K.
```

This unifies fixed short chunks, fixed long chunks, stochastic prefix dropout, and
deterministic temporal weighting in one framework, subject to an important boundary:
the static equivalence is exact for additive losses with no padding and no suffix-to-
prefix leakage. With padding, exact deterministic coefficients are sample-specific;
with full-attention or flattened backbones, replacing suffix inputs can change prefix
predictions. ACT also uses a different mask normalization and must be audited
separately.

### 4.6 Proposed endpoint: Predictive-Observability-Aware Horizon Supervision

Global horizon dropout is a useful baseline, but it is not yet a true multi-task
adaptation mechanism. The publishable endpoint is a task-conditioned objective

```text
L_PAHS = sum_{k=1}^K w_{tau,k} * ell_{t,k},
```

where `w_{tau,k}` is derived from a cross-fitted predictive profile with explicit
incremental-information, control-relevance, and tail-floor terms. Let a small
calibration model estimate

```text
mu_{tau,k}(h) = E[a_{t+k} | h, tau],
Sigma_{tau,k}(h) = Cov[a_{t+k} | h, tau].
```

The Jacobian of the predictable mean defines a local noise-whitened predictive
Gramian. Write

```text
Jtilde_{tau,k} = (Sigma_{tau,k} + delta I)^-1/2 J_{tau,k},
G_{tau,1:k} = sum_{j<=k} Jtilde_{tau,j}^T Jtilde_{tau,j},
i_{tau,k} = logdet(G_{tau,1:k} + epsilon I)
            - logdet(G_{tau,1:k-1} + epsilon I),
c_{tau,k} = ||Jtilde_{tau,1} Jtilde_{tau,k}^T||_F^2
            / (||Jtilde_{tau,1}||_F^2 ||Jtilde_{tau,k}||_F^2 + epsilon).
```

Here `i` measures new robust predictive directions and `c` measures their local
subspace overlap with the next-action-relevant direction using demonstrations only.
Near/far policy-gradient cosine remains a mediator measurement and validation
ablation rather than an input required by the method.

The first candidate allocation is

```text
r_{tau,k} = (i_{tau,k} + epsilon) * (c_{tau,k} + epsilon)^gamma
w_{tau,k} = (1-beta)/K + beta*r_{tau,k}/sum_j r_{tau,j}.
```

The uniform floor preserves long-horizon coverage and avoids collapsing all difficult
tasks to `K=1`. Whitening makes predictable state-disambiguating directions valuable
and suppresses directions dominated by irreducible target noise; `c_{tau,k}` prevents
observable but next-action-irrelevant directions from receiving high weight. The
uncertainty-only, information-only, and alignment-only variants are mandatory
ablations. Only cross-fitted aleatoric uncertainty is used for downweighting.
Epistemic underfit is a signal to add data or capacity, not to reduce supervision.
For generative policies, high conditional variance may represent meaningful
multimodality, so NLL, calibration, mode coverage, and task success must be evaluated
together.

This candidate needs no inference-time horizon selector and no change to the deployed
policy architecture, but it does require an offline calibration model. Its formula
is not frozen until the demonstration-only quantities predict held-out optimal horizons.
A state- or phase-conditioned gate `w_k(h_t, tau)` is a later extension, not the first
milestone; it should only be attempted after task-level conditioning is validated.

---

## 5. Work Packages

### WP0 — Reproducibility and implementation audit

Before interpreting more runs:

1. Give all compared conditions the same optimizer-update budget and LR schedule.
2. Use the same demonstration split, valid sequence starts, normalization statistics,
   and paired evaluation initial states.
3. Log the number of valid target tokens, padded tokens, sampled prefix lengths,
   per-offset losses, per-offset gradient norms, and wall-clock cost.
4. Add numerical tests showing that enumerated or Monte Carlo horizon dropout
   matches deterministic expected-weight loss and full gradients in the no-padding
   causal case. Add a padded counterexample and the sample-specific exact
   coefficients rather than silently calling static weights exact.
5. Standardize mask normalization across model families. Flow Matching currently
   normalizes by the valid prefix length, while the current ACT path averages masked
   zeros over the full tensor and therefore changes total gradient scale for shorter
   prefixes. ACT results are not comparable until this semantic difference is fixed
   or explicitly controlled.
6. Verify that masked suffix tokens cannot leak target information into valid prefix
   predictions for every backbone. For full-attention models, compare clean loss-only
   weighting against true token/attention truncation instead of silently mixing them.
7. Use one common `K_max=20` start-index set for the primary comparison so shorter
   horizons do not gain extra episode-tail samples. Keep a native `K=1` model only as
   a secondary deployment-size baseline.

8. Run a three-seed screening comparison on the exact first 8000 Pen transitions:
   fixed `K=20`, Linear 2:1, and conservative Horizon Dropout. Expand the surviving
   pair to five seeds before paper-level claims.

**Gate M0:** reproduce the Pen signal under a matched-update protocol and show that
the implementation-level equivalence claims pass numerical tests. If the performance
signal disappears across seeds, retain the horizon study but do not claim a solved
method.

### WP1 — Controlled predictive-observability benchmark

Build a small benchmark in which observability and predictability can be varied
independently without changing task identity:

- a shared multi-task linear suite that independently rotates future-observable
  directions toward or away from the union of next-action subspaces, while varying
  samples per task and bottleneck width;
- linear/LQR systems with controlled closed-loop observability rank, Gramian
  conditioning, spectral radius, and process/observation noise;
- double pendulum with tunable energy, damping, perturbation magnitude, and hidden
  velocity;
- optional controlled N-body follow-up with full versus partial state and controlled
  perturbations.

For each condition:

- collect identical-sized expert datasets;
- train `K in {1, 5, 10, 20}` with `E=1`;
- measure held-out future-action NLL/error at every offset;
- compute the exact linear closed-loop observability matrix and Gramian, or a local
  empirical analogue for nonlinear systems;
- measure state rollout NRMSE using ground-truth future actions;
- estimate perturbation divergence or a finite-time Lyapunov proxy;
- measure contact/event predictability where applicable;
- record the best horizon and its uncertainty.

The primary analysis is a within-system factorial intervention. At fixed uncertainty,
increasing **control-aligned** predictive information should lengthen the useful
horizon only in the finite-sample/shared-representation regime; unaligned
observability is not expected to help first-action risk. At fixed alignment,
increasing noise or sensitivity should shorten it. Population single-task regression,
a predictable but action-constant system, and future directions orthogonal to all
next-action subspaces are required negative controls.

**Gate M1:** alignment and uncertainty interventions must move their respective
measurements and the relative performance of long versus short horizons in the
preregistered directions on held-out seeds. The first exact reduced-rank screen has
already failed the low-noise long-horizon half of this gate; the next shared-task
design must pass without post-hoc cell selection or the mechanistic headline is
abandoned.

### WP2 — Robot-task horizon map

Use tasks already close to the current project stack:

- Adroit Pen, Door, Hammer, and Relocate with state observations;
- PushT as a simple planar manipulation bridge;
- selected RoboMimic and BiGym tasks spanning free-space, contact-rich, and
  feedback-sensitive phases;
- pixels and language only after the state-based mechanism is understood.

Core design:

- `K in {1, 5, 10, 20}`, initially with `E=1`;
- data regimes such as `{10%, 25%, 100%}`;
- Flow Matching as the primary model and ACT or Diffusion Policy as a replication;
- at least three screening seeds, expanding the final comparisons to five seeds;
- control horizon in physical time as well as steps when control frequencies differ.
- use the same maximum-horizon replay windows for the causal training-horizon test,
  then separately report native short-sequence models.

Extend `scripts/analyze_dynamics_predictability.py`. Its current ground-truth-action
rollout measures transition predictability, but it must be complemented with a
cross-fitted future-action predictor and uncertainty estimates. Otherwise it cannot
measure the ambiguity seen directly by the chunk loss.

### WP3 — Mechanism and interpretability

For the trained horizon grid:

1. Probe frozen hidden states for future state, velocity, contact, and phase.
2. Compare probes against observation-only and matched-capacity controls.
3. Measure loss by action offset instead of only the aggregate training loss.
4. Measure gradient cosine similarity between near and far offsets and between tasks.
5. Intervene on observation history, hidden velocities, noise, and perturbations.
6. Test whether predictability or gradient conflict mediates the relationship between
   task condition and optimal horizon.

Possible outcomes are deliberately separated:

- Better future-state probes support a dynamics-sensitive representation account.
- No probe gain but strong near/far gradient conflict supports a supervision-conflict
  account.
- High future-action uncertainty without high transition error supports an expert
  ambiguity or partial-observation account.

### WP4 — Method ablation: dropout, weighting, and alternatives

Run the following with matched updates, LR, data, model capacity, and evaluation:

1. fixed `K=1`, `K=5`, `K=10`, and `K=20`;
2. global Horizon Dropout;
3. exact deterministic dropout-equivalent weights;
4. exponential and linear decay weights with matched total loss mass;
5. random non-prefix token masking with the same expected token budget;
6. target-token-matched fixed supervision: `K=6` for the aggressive schedule and
   approximately `K=15` for the conservative schedule;
7. `K=20` with only the first-token loss, to separate output architecture from
   supervision horizon;
8. generic regularization controls such as model dropout or weight decay;
9. predictive-observability-aware task weights with a nonzero long-horizon floor;
10. a strong multi-horizon baseline such as Mixture of Horizons where feasible.

The key comparison is not merely "dropout versus no dropout." It is:

```text
stochastic prefix vs exact expected weights vs other weights vs true multi-horizon model.
```

For the first stochastic-versus-expected-weight screen, an absolute normalized-AUC
difference below `0.02` is treated as no practically meaningful evidence of a
stochastic benefit. A difference of at least `0.03` only triggers paired replication;
the stochastic-regularization claim requires the direction to survive at least three
seeds. Values between `0.02` and `0.03` are inconclusive rather than rounded into a
claim.

**Gate M2:** if stochastic and deterministic variants match, rename the mechanism
Horizon Weighting and treat randomness as an implementation option. If neither beats
well-tuned fixed horizons across tasks, the method claim is rejected even if the
explanatory study remains useful.

### WP5 — Multi-task foundation-policy experiment

A collection of single-task runs is not evidence about a foundation model. This work
package requires one shared policy trained on a balanced heterogeneous task mixture.

Start with tasks sharing an action representation, then expand to multi-embodiment
training only after embodiment normalization is controlled. Compare:

- global fixed horizons `K in {1, 5, 10, 20}`;
- best global fixed horizon selected on validation tasks;
- global Horizon Dropout and global deterministic weighting;
- per-task oracle horizon or weights as an upper-bound diagnostic;
- predictive-observability-aware task-conditioned weights;
- learned observation-conditioned weights only as a later extension;
- Mixture of Horizons or its loss-reweighting baseline when compute permits.

Primary metrics:

- macro-average success or normalized return;
- worst-task and 10th-percentile performance;
- per-task regret to the single-task oracle horizon;
- negative-transfer gap between shared and single-task policies;
- held-out-task generalization of the predictability-to-weight rule;
- training FLOPs, target-token budget, inference latency, and memory.

**Gate M3:** the proposed objective must improve either macro-average and worst-task
performance together, or reduce oracle regret without a meaningful average-score
loss. A gain caused only by reweighting the task sampler is not sufficient.

### WP6 — Prediction and execution as orthogonal controls

After selecting training objectives, reuse each checkpoint to sweep valid execution
horizons `E <= K`. This establishes whether training-time predictability weighting is
complementary to inference-time adaptive chunking, temporal ensembling, or real-time
chunking.

This stage must not retroactively use execution-horizon results to explain the Pen
`E=1`, no-ensemble training comparison.

---

## 6. Evaluation and Statistical Protocol

- Use paired dataset and environment seeds across methods.
- Use a validation split or predetermined checkpoint rule; evaluate the selected
  checkpoint once on the held-out test seeds.
- Report final performance and learning-curve AUC as primary outcomes. Best-of-many
  checkpoints is diagnostic because it introduces selection bias.
- Use 100-200 evaluation episodes per final seed where simulation cost allows.
- Report mean, seed-level dispersion, and bootstrap or hierarchical confidence
  intervals. For success rates, preserve episode-level binomial uncertainty.
- Use mixed-effects regression for the predictability-horizon hypothesis, with task
  and seed as random effects and dataset size, control frequency, and model family as
  covariates.
- Predefine the primary predictability metric and the primary horizon contrast before
  the final multi-seed sweep.
- Report all failed or negative architecture replications.

---

## 7. Preliminary Evidence: Corrected Pen and FlipCutlery Pilots

These results complete the five-seed M0 screen for fixed training and conservative
Horizon Dropout; Linear 2:1 remains a three-seed rejected screen. Each checkpoint
uses 50 evaluation episodes. Best-of-ten checkpoints is reported only as a
diagnostic; curve mean/AUC and final performance are primary because best-checkpoint
selection is optimistic. Values below are mean and sample standard deviation.

### Pen: exact first 8000 dataset transitions

All methods use Flow Matching, batch size 128, `K=20`, `E=1`, 50,000 optimizer
updates, identical original-order data, and identical evaluation cadence. These
pilot evaluations use temporal ensembling, so old tail predictions can contribute to
the executed action even though the environment replans every step.

| Training objective | Seeds | Curve mean | Normalized AUC | Mean per-seed best | Final |
|---|---:|---:|---:|---:|---:|
| Fixed uniform `K=20` | 5 | `0.620 +/- 0.018` | `0.623 +/- 0.020` | `0.720 +/- 0.047` | `0.576 +/- 0.033` |
| Linear 2:1 weighting | 3 | `0.614 +/- 0.005` | `0.612 +/- 0.008` | `0.707 +/- 0.031` | `0.580 +/- 0.020` |
| Conservative Horizon Dropout | 5 | **`0.655 +/- 0.018`** | **`0.655 +/- 0.017`** | **`0.756 +/- 0.052`** | **`0.632 +/- 0.033`** |

Paired against fixed `K=20` across five seeds, conservative dropout improves curve
mean by `+0.0344`, normalized AUC by `+0.0324`, mean best by `+0.036`, and final
success by `+0.056`. Curve mean and AUC improve in all five paired seeds; the
seed-level 95% t intervals are `[+0.0226,+0.0462]` and `[+0.0217,+0.0432]`,
respectively. Linear 2:1 is not robust on Pen: its three-seed paired mean changes are
`-0.008` for curve mean and `-0.011` for AUC. The earlier `0.74` versus `0.68`
comparison was a single-seed best-checkpoint effect and is superseded by this screen.

M0 therefore advances conservative Horizon Dropout as the Pen finalist under the
existing temporal-ensemble protocol. This is still not generality or a pure
training-horizon result. A predetermined 50k-checkpoint reevaluation with `E=1`,
temporal ensembling disabled, and 100 episodes per seed is the immediate confound
check.

That confound check is now complete. At the final 50k checkpoint, fixed training has
per-seed success `[0.65, 0.70, 0.62, 0.71, 0.72]` (`0.680 +/- 0.043`) and conservative
Horizon Dropout has `[0.78, 0.68, 0.71, 0.78, 0.74]` (`0.738 +/- 0.044`). The paired
mean change is `+0.058`, positive in four of five seeds. Its seed-level 95% t interval
is `[-0.015,+0.131]`, so the result shows that the signal is not solely created by
temporal ensembling, but it is not a standalone significance claim.

A leakage-safe future-action proxy provides a compatible but non-causal diagnostic.
Using all 80 replay episodes, a nested episode-disjoint ridge model predicts each
24-D `a_{t+h}` from the current 45-D pre-action state. Variance-weighted cross-fitted
NMSE rises from `0.4661` at `h=0` to `0.7223` at `h=19`; mean tail NMSE (`h=10..19`)
exceeds mean head NMSE (`h=0..4`) by `0.1514`, with a paired episode bootstrap 95%
interval `[0.0945,0.1923]`. This shows that distant expert actions are materially
harder for a simple current-state predictor, but it does not identify plant dynamics:
expert multimodality, episode shift, missing history, and linear-model underfit remain
alternative explanations.

### Audited Pen supervision-horizon screen

The decisive single-seed screen changes **supervision horizon** `H` while keeping the
model output at `K_max=20`. All conditions use `E=1`, no temporal ensemble, a new seed
5, fixed evaluation seeds, identical sum-one weighted-loss code paths, and the same
6480 full-length replay origins. Suffix perturbation has zero effect on token 0, and
all four 5k snapshots have identical JAX, Python, and NumPy RNG states. This isolates
the effect of distant losses on shared parameters. It is not a native architecture-`K`
comparison and does not use the extra episode-tail origins available to a true short
model.

| Supervision `H` | Curve mean | Normalized AUC | Best diagnostic | Final 50k |
|---:|---:|---:|---:|---:|
| 1 | **`0.766`** | **`0.774`** | **`0.84`** | `0.70` |
| 5 | `0.704` | `0.711` | `0.78` | `0.60` |
| 10 | `0.702` | `0.701` | `0.74` | **`0.72`** |
| 20 | `0.700` | `0.710` | `0.80` | `0.64` |

The preregistered curve metrics favor `H=1`: its AUC exceeds `H={5,10,20}` by
`{0.063,0.073,0.064}`. Selecting only the final checkpoint would instead choose
`H=10`, illustrating why final, best, and learning-curve evidence must be separated.
`H=1` versus `H=20` was then replicated with seeds 6 and 7, crossing the GPU
assignment between those seeds. Aggregating seeds 5, 6, and 7 gives:

| Objective | Curve mean | Normalized AUC | Mean per-seed best | Final 50k |
|---|---:|---:|---:|---:|
| `H=1` | **`0.769 +/- 0.036`** | **`0.773 +/- 0.036`** | **`0.833 +/- 0.031`** | **`0.753 +/- 0.061`** |
| `H=20` | `0.705 +/- 0.017` | `0.710 +/- 0.015` | `0.787 +/- 0.023` | `0.660 +/- 0.072` |

The paired curve and AUC improvements are positive in all three seeds: `+0.064` and
`+0.062` on average, with seed-level 95% t intervals `[+0.017,+0.111]` and
`[+0.010,+0.115]`. The paired final improvement is `+0.093`, but its interval crosses
zero. This supports a Pen-specific short-supervision result under the controlled
`K_max=20` design; it is not yet a native-architecture or cross-task law.

The result rejects the current version of "long supervision is better on Pen."
Together with rising tail NMSE, it instead nominates Pen as a far-target-interference
case. Conservative Horizon Dropout's five-seed improvement therefore cannot be
attributed to beneficial long-horizon latent dynamics.

That exact no-padding test is also complete under the same fair seed-5 protocol. The
conservative distribution has expected weights
`[0.125, 0.075 x 4, 0.055 x 5, 0.03 x 10]`.

| Objective | Curve mean | Normalized AUC | Best diagnostic | Final 50k |
|---|---:|---:|---:|---:|
| Stochastic conservative prefix | `0.704` | `0.708` | `0.78` | `0.66` |
| Exact deterministic expected weights | **`0.736`** | **`0.738`** | **`0.80`** | **`0.74`** |

Deterministic weighting is higher by `+0.032` in curve mean and `+0.030` in AUC.
The AUC difference reaches the preregistered replication trigger but is only one
seed; it is not yet evidence that deterministic weighting is universally superior.
It is, however, direct evidence against claiming a stochastic-dropout advantage.
Expected weighting improves over uniform `H=20` by `+0.028` AUC, while remaining
`0.037` below the `H=1` oracle screen. The current method should therefore be framed
as horizon weighting unless paired replication reverses this result.

### FlipCutlery: long execution horizon

The repaired matched runs use `K=20`, `E=20`.

| Training objective | Curve mean success | Best success | Final success |
|---|---:|---:|---:|
| Fixed uniform `K=20` | **`0.762`** | `0.84` | **`0.82`** |
| Linear 2:1 weighting | `0.740` | **`0.86`** | `0.76` |
| Conservative Horizon Dropout | `0.668` | `0.80` | `0.76` |

The more front-loaded conservative dropout is weaker on this long-execution task,
whereas the gentler Linear 2:1 schedule preserves a competitive curve and slightly
improves the observed best checkpoint. This supports a nonzero long-horizon floor as
a design requirement, not a universal performance claim.

### Invalidated historical pilots

Earlier directories labelled `8000t` used `demos=80`; the Minari finite-demo path
selected high-return episodes and did not guarantee the first 8000 transitions.
Some epoch-style runs also had mismatched optimizer schedules. They remain useful for
implementation archaeology but are excluded from primary evidence. Initial ACT runs
are likewise not cross-architecture evidence because their mask normalization and KL
scaling differ from Flow Matching.

### WP1 analytical linear-system sanity check

The first 540-row sweep independently varies target closed-loop spectral radius
`{0.5, 0.9, 1.1}`, action-observability scale `{0, 0.1, 1}`, process/output-noise
scale `{0, 1, 10}`, and horizon `1..20`.

- With observability scale `0`, the action observability rank stays `1` through
  `H=20`. With either nonzero scale it reaches the full rank `2` at `H=2`.
- At spectral radius `0.9`, improving observability scale from `0.1` to `1.0`
  reduces the `H=2` observability condition number from about `40.5` to `6.54`.
- With spectral radius `0.5`, predictive covariance contracts to a noise floor; with
  spectral radius `1.1`, state/action covariance grows rapidly with horizon. At
  noise scale `1`, state covariance trace grows from `1.53` at `H=1` to `47.38` at
  `H=20`.

This verifies that WP1 can manipulate observability and predictability separately.
It is an analytical sanity check, not yet evidence that a learned chunk policy's
representation follows these quantities.

### WP1 exact learned-bottleneck screen

An exact reduced-rank regression model now implements a linear observation encoder,
a width-3 shared bottleneck, and `H in {1,2,5,10,20}` future-action heads. The default
factorial contains 240 fitted models and 1,824 per-offset records across three seeds.
At spectral radius `0.7`, requested rank `6`, low predictive noise, and `H=20`, raising
observability scale from `0.1` to `1` increases state-probe R2 from `0.338` to `0.457`.
At scale `1`, raising predictive-noise standard deviation from `0.25` to `1` lowers
R2 from `0.457` to `0.274` and raises first-action MSE from `0.0138` to `0.1239`.
The `H=1` first-action MSE is unchanged by this future-only noise intervention, as
required by the negative control.

The stronger headline gate nevertheless **fails** in this exact model. A finite-
sample design search was frozen and checked on 30 unseen seeds. Under low process
noise, the apparent best `H=2` improves first-action MSE over `H=1` by only `0.49%`
(`0.87588` versus `0.88019`; paired delta `-0.00431 +/- 0.00720` standard error).
Under high noise, `H=1` is clearly best and `H=2` is worse in 29/30 seeds. Thus the
current evidence supports "uncertain tails can hurt" and "long targets can preserve
more state," but it does not support "observability alone makes long supervision
improve the next action." This negative result motivates the reduced-rank control-
relevance theorem and a genuinely shared/neural finite-sample experiment; it must not
be hidden by further tuning of the synthetic cell.

---

## 8. Novelty and Related-Work Positioning

The project should **not** claim to be the first work showing that fixed action chunks
are suboptimal or the first adaptive-horizon method.

- [ACT](https://arxiv.org/abs/2304.13705) established action-chunk prediction and
  temporal ensembling for imitation learning.
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) made receding-horizon action
  generation a central robot-policy design.
- [pi_0](https://arxiv.org/abs/2410.24164) demonstrates fixed long-chunk flow matching
  in a multi-robot generalist policy.
- [BAKU](https://arxiv.org/abs/2406.07539) reports heterogeneous effects of action
  chunking across a 129-task multi-task study, motivating an explanation beyond a
  universal larger-is-better rule.
- [Action Chunking and Data Augmentation Yield Exponential Improvements in Behavior
  Cloning](https://arxiv.org/abs/2507.09061) provides a control-theoretic stability
  account of when **executed** action chunks mitigate compounding error. Its main
  mechanism depends on execution horizon and does not explain an `E=1`, no-ensemble
  training-horizon effect.
- [Bidirectional Decoding](https://proceedings.iclr.cc/paper_files/paper/2025/file/0d78dd998f7b9ac79604d47a2d79bb0d-Paper-Conference.pdf)
  separates the temporal-dependency benefit of chunks from their loss of closed-loop
  reactivity.
- [Mixture of Horizons](https://arxiv.org/abs/2511.19433) directly studies fixed
  training-horizon trade-offs, includes a temporal loss-reweighting ablation, and
  proposes a multi-horizon architecture with adaptive execution.
- [Adaptive Action Chunking](https://arxiv.org/abs/2604.04161) selects the execution
  horizon at inference from action entropy.
- [Understanding Multimodal Failure in Action-Chunking Behavioral
  Cloning](https://arxiv.org/abs/2605.22493) makes expert-mode ambiguity an important
  alternative to a pure dynamics explanation.
- [Horizon-Calibrated Uncertainty World Models](https://iclr.cc/virtual/2026/poster/10007319)
  already argue that predictive uncertainty should vary with temporal horizon. The
  present work must distinguish action-supervision allocation and closed-loop
  observability from uncertainty-aware world-model pretraining.
- [Denoising-Variance Adaptive Chunking](https://arxiv.org/abs/2606.03847) uses
  denoising variance to select execution length at test time. It is a required
  complementary baseline for `E`, but does not address fixed multi-task supervision
  when `E=1` and temporal ensembling is disabled.
- [Nested Dropout](https://proceedings.mlr.press/v32/rippel14.html) is a general
  precedent for stochastic prefix truncation, so the prefix-dropout mechanism itself
  should not be presented as a new general learning principle.
- [Predictive Representations of State](https://proceedings.neurips.cc/paper_files/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf),
  [Takens' delay embedding](https://link.springer.com/chapter/10.1007/BFb0091924),
  and [Embedology](https://doi.org/10.1007/BF01053745) motivate future observables as
  state coordinates, but standard action-chunk BC lacks the counterfactual tests
  required to call its representation a full predictive state or plant model.

The defensible novelty target is the combination of:

1. a closed-loop observability formulation of action chunks as predictive
   coordinates, including an exact linear-system result;
2. a falsifiable link among incremental observability, finite predictability,
   supervision conflict, and useful action-prediction horizon;
3. controlled observability and dynamics interventions plus robot-task validation;
4. a conditionally exact unification of prefix dropout and horizon weighting, with
   padding, attention, and normalization boundaries made explicit;
5. predictability-conditioned supervision for a shared multi-task policy, with no
   inference-time overhead;
6. evidence across policy families and a clean separation of prediction from
   execution horizon.

Horizon Dropout alone is likely too incremental as the final contribution. It remains
a valuable minimal baseline and may be the simplest implementation of the broader
objective.

---

## 9. Risks, Confounds, and Alternative Explanations

1. **Optimizer-budget confound.** Different `K` values currently produce different
   batches per epoch and schedule completion. Use matched updates as the primary view;
   matched epochs and matched target tokens are secondary views.
2. **Loss-scale confound.** Masking implementations may average over valid tokens or
   over the full tensor. Match expected weights and total gradient scale explicitly.
3. **Dataset-boundary confound.** Long sequences change valid start indices and
   padding. Share start indices or report the difference.
4. **Action multi-modality.** Far-future action ambiguity may come from multiple valid
   expert strategies rather than hard transitions. Measure action and state
   predictability separately.
5. **Control-frequency confound.** Twenty steps at 5 Hz and 50 Hz are not the same
   physical horizon. Report both steps and seconds.
6. **Model-capacity confound.** `K=1` and `K=20` can change token count, compute, and
   decoder behavior. Include `K=20` first-token-only and capacity controls.
7. **Task-horizon confound.** Long semantic tasks may need foresight even with noisy
   low-level dynamics. Preserve a long-horizon loss floor and measure temporal
   structure benefit separately from predictability cost.
8. **Probe non-identifiability.** Future state may be predictable directly from the
   current state without being represented by the policy. Use matched observation-only
   probes and interventions.
9. **Synthetic-to-robot gap.** Double pendulum and N-body results establish mechanism,
   not robotic relevance. The robot task map is required.
10. **Related-work overlap.** Mixture of Horizons already covers the generic fixed-
    horizon trade-off and loss reweighting. The paper must lead with predictability,
    causal evidence, and task-conditioned training rather than the dropout name.
11. **Attention leakage.** Loss-masked suffix tokens are harmless to the current
    causal transformer prefix but may be visible in full-attention VLA heads. Such
    models require attention masking/truncation or must be described only as temporal
    loss reweighting.
12. **Predictable-but-uninformative futures.** Low future-action error alone does not
    justify long chunks. Measure observability rank, incremental information, or
    state-pair disambiguation to detect redundant action tails.
13. **Aleatoric/epistemic confusion.** Downweighting high residuals caused by model
    underfit creates a self-reinforcing failure. Estimate uncertainty out of sample,
    preserve a tail floor, and treat epistemic error as a data/capacity signal.
14. **Deterministic-chaos overclaim.** Positive Lyapunov exponents amplify nonzero
    uncertainty; they do not make exact deterministic futures random. Report initial
    uncertainty and finite-time exponents together.
15. **Best-checkpoint selection.** The current `0.74` and `0.86` Linear 2:1 results
    are diagnostic maxima over multiple evaluations. Primary conclusions require
    curve AUC/final metrics or validation-selected checkpoints on held-out episodes.

---

## 10. Milestones and Decision Gates

| Milestone | Deliverable | Continue if... | Fallback if not... |
|---|---|---|---|
| `M0` Fair Pen replication | three-seed screen, then five-seed finalist table | HD/HW is competitive under exact first-8000 data and matched updates | keep only as an exploratory observation |
| `M1` Control-aligned predictive benchmark | controlled alignment, rank, noise, finite samples, sharing, and horizon metrics | aligned information and uncertainty move optimal supervision horizon on unseen seeds | reject observability-only story; test ambiguity and gradient conflict |
| `M2` Mechanism | probes and gradient analysis | at least one preregistered mechanism is supported | narrow the explanatory claim |
| `M3` Single-task solution | HD/HW/PAHS ablation across tasks | method improves more than one task and model | publish analysis, not a universal solution |
| `M4` Multi-task policy | shared-model comparison | macro and worst-task/oracle-regret criteria improve | report limits of global horizon supervision |
| `M5` Scale and write-up | pixels/language, execution sweep, paper | conclusions survive scale and strong baselines | scope paper to validated state-based setting |

Suggested sequence is roughly 12 weeks: one week for `M0`, two for controlled systems,
two to three for the robot horizon map, two for mechanisms and method ablations, three
for multi-task training, and the remainder for scale-up and writing. Use a funnel:
screen cheaply with three seeds, then spend five-seed and high-episode evaluation only
on preregistered finalists.

---

## 11. Expected Paper Story

If all major hypotheses survive, the paper narrative is:

1. Action-prediction horizon is not merely an output-shape hyperparameter.
2. Action chunks are finite-horizon predictive coordinates of the expert-induced
   closed-loop system; in linear systems their information is governed exactly by a
   closed-loop observability matrix.
3. More observable state is not automatically useful for the next action. Their
   benefit is governed by a measurable trade-off among control-aligned incremental
   information, finite-sample representation sharing, and conditional predictability.
4. Fixed horizons misallocate supervision and create near/far or cross-task gradient
   conflict in heterogeneous training.
5. Under explicit architectural and padding conditions, Horizon Dropout is a
   stochastic form of temporal loss weighting, motivating a general horizon-weighted
   objective.
6. Control-aligned predictive task weights improve shared multi-task policies without
   per-task horizon search or inference-time overhead.

If representation probes fail but weighting succeeds, the paper should say
"predictability-aware supervision" rather than "implicit latent dynamics." If the
method fails but the controlled correlation is robust, the result can still be a
useful explanatory and benchmarking paper. If neither survives multi-seed controls,
the correct outcome is a documented negative result, not a stronger post-hoc story.

---

## 12. Reproducibility: Existing Pilot Artifacts

Primary corrected Flow Matching runs:

- `exp_local/pen_fm_first8000_baseline_k20_e1_seed{0,1,2,3,4}_50k_20260710`
- `exp_local/pen_fm_first8000_linear2to1_k20_e1_seed{0,1,2}_50k_20260710`
- `exp_local/pen_fm_first8000_hdrop_conservative_k20_e1_seed{0,1,2,3,4}_50k_20260710`
- `exp_local/pen_fm_first8000_{baseline,hdrop_conservative}_k20_e1_noensemble_final100_20260710`
- `exp_local/pen_fm_first8000_suph{1,5,10,20}_kmax20_e1_noensemble_seed5_50k_20260710`
- `exp_local/bigym_flip_cutlery_fm_fixed_baseline_repaired_1000e_b128_seed0_20260710`
- `exp_local/bigym_flip_cutlery_fm_hweight_linear2to1_repaired_1000e_b128_seed0_20260710`
- `exp_local/bigym_flip_cutlery_fm_hdrop_conservative_repaired_1000e_b128_seed0_20260710`

Historical `pen_fm_transformer_*_8000t_*_20260709` runs are excluded from primary
tables because `demos=80` did not mean the first 8000 transitions.

Relevant implementation and analysis entry points:

- `robobase/method/flow_matching.py`
- `robobase/method/act.py`
- `robobase/cfgs/method/flow_matching.yaml`
- `scripts/analyze_dynamics_predictability.py`
- `scripts/analyze_linear_predictive_observability.py`
- `scripts/analyze_learned_bottleneck_horizon.py`
- `scripts/analyze_replay_action_predictability.py`
- `scripts/eval_execution_length_sweep.py`
- `tests/unit/test_flow_horizon_dropout_math.py`
- `tests/unit/test_analyze_linear_predictive_observability.py`
- `tests/unit/test_analyze_learned_bottleneck_horizon.py`
- `tests/unit/test_analyze_replay_action_predictability.py`
- `exp_local/linear_predictive_observability_20260710/sweep.csv`
- `exp_local/linear_learned_bottleneck_20260710/sweep.csv`
- `exp_local/pen_action_predictability_20260710/profile.csv`

---

## 13. Current Execution Status — 2026-07-10

- [x] Correct Minari selection to use the original-order first 8000 Pen
  transitions, not top-return episodes.
- [x] Complete matched 50k-update seed-0/1/2 screening for fixed `K=20`, Linear
  2:1, and conservative Horizon Dropout.
- [x] Add numerical no-padding loss/gradient equivalence, Linear 2:1 inverse-prefix,
  and padded-counterexample tests (`5` focused tests passing including existing
  Flow horizon tests).
- [x] Advance conservative Horizon Dropout through M0; reject a robust Pen gain for
  Linear 2:1 at this stage.
- [x] Expand fixed and conservative Horizon Dropout to five paired Pen seeds; curve
  mean and AUC deltas are positive in all five under the existing temporal-ensemble
  protocol.
- [x] Complete the predetermined 50k-checkpoint, 100-episode-per-seed Pen
  reevaluation with `E=1` and temporal ensembling disabled; dropout improves the
  five-seed mean by `+0.058`, with four positive pairs and a confidence interval that
  still crosses zero.
- [x] Implement WP1's analytical linear predictive-observability benchmark and run
  the first 540-row rank/noise/spectral-radius/horizon sweep (`4` focused CPU tests
  passing).
- [x] Add and run the exact reduced-rank learned-bottleneck experiment on the same
  linear systems (`K in {1,2,5,10,20}`), measuring state-probe R2, first-action
  error, and per-offset loss (`7` combined WP1 tests passing).
- [x] Add and run an episode-disjoint, nested-CV replay future-action predictor with
  shared episode bootstrap draws; Pen tail NMSE rises materially, while the tool and
  report explicitly label the metric as an expert-action proxy rather than dynamics
  proof (`6` focused tests passing).
- [x] Complete the audited Pen supervision-horizon screen at common
  `K_max=20`, `H in {1,5,10,20}`, `E=1`, no temporal ensemble, fixed eval seeds,
  common replay origins, and explicit sum-one weights in every condition; the new
  seed-5 screen favors `H=1` on curve mean/AUC.
- [x] Complete the fair stochastic conservative-prefix versus exact expected-weight
  screen; deterministic weighting is `+0.030` AUC higher in seed 5, rejecting any
  current stochastic-benefit claim and triggering paired replication only.
- [x] Complete the paired H1-versus-H20 replication on seeds 6 and 7, with
  GPU assignment crossed between seeds; combine with seed 5 before deciding whether
  the short-supervision direction is robust. H1 improves curve mean and AUC in all
  three seeds by `+0.064` and `+0.062` on average.
- [x] Audit the decisive screen's causal isolation: suffix perturbation changes the
  first-token output by exactly `0.0`, all four conditions report the same 6480 valid
  origins, and all four 5k snapshots have identical JAX-agent, Python, and NumPy RNG
  states.
- [ ] Add a neural learned-bottleneck version with near/far gradient alignment; the
  exact regression screen has no optimizer gradients by design.

---

## 14. Best-Paper Gap and Evidence Program

### 14.1 Honest gap assessment

The project currently has a strong question, a precise linear observability identity,
a corrected five-seed Pen method signal that survives a no-ensemble confound check,
one long-execution counterexample, and analytical plus exact learned-bottleneck
benchmarks. The learned benchmark also produced an informative failure: more state
information did not robustly improve the next action. That is enough for a serious
research project, but not for a best-paper claim. The missing pieces are:

1. **A new theorem rather than only a connection.** The observability factorization
   must be extended into a finite-sample or control-relevant statement involving
   representation capacity, conditional noise, and useful horizon. Takens/Koopman
   alone are motivation, not novelty.
2. **Causal learned-mechanism evidence.** The analytical quantities must predict what
   a trained shared bottleneck represents and when large `K` improves or harms the
   first action under controlled interventions.
3. **A method beyond global dropout.** Global Horizon Dropout is an ablation. The
   final method must estimate a task's predictive-observability profile and allocate
   horizon supervision without per-task horizon search.
4. **A real shared multi-task model.** Separate single-task runs do not establish the
   foundation-model problem or negative transfer.
5. **Breadth and strong baselines.** The study needs multiple task families, two
   policy architectures, a full fixed-`K` grid, deterministic/stochastic matched
   objectives, Mixture of Horizons, and execution-adaptive controls.
6. **Statistical and selection rigor.** Best checkpoints cannot be the primary
   outcome. Final claims require paired seeds, a validation-selected checkpoint, held-
   out evaluation episodes/tasks, uncertainty intervals, and preregistered contrasts.
7. **Scale or real-world relevance.** A best-paper-level version needs either a
   convincing real-robot result or unusually broad and causal simulation evidence
   with a held-out-task generalization result.

### 14.2 Target headline

The intended headline is not "dropout improves two tasks." It is:

> A demonstration-only measure of control-aligned predictive information predicts
> the useful training-supervision horizon on held-out tasks. Allocating supervision
> with that measure reduces oracle-horizon regret and improves both average and
> worst-task performance of a shared robot policy without inference-time horizon
> search.

The central paper figure should place controlled systems and robot tasks on a common
axis: measured control-aligned predictive horizon versus empirically optimal `K` or
long-minus-short performance. A leave-one-task-out prediction should outperform task
length, return, control frequency, and raw future-action MSE as alternative
explanations.

### 14.3 Preregistered experimental funnel

| Stage | Intervention and comparison | Primary measurements | Continue gate |
|---|---|---|---|
| A. Linear learned mechanism | Factorial alignment x observability x finite samples/sharing x spectral radius/noise; `H={1,2,5,10,20}`, `E=1`, narrow shared bottleneck | state-probe rank/R2, first-action MSE, per-offset MSE, near/far gradient cosine | Aligned predictive information lengthens useful `H` while uncertainty shortens it on held-out seeds; the completed single-task exact screen failed the first half |
| B. Nonlinear causal benchmark | Same torque-controlled double pendulum with damping, energy, hidden velocity, observation noise, and perturbation changes | finite-time Lyapunov proxy, conditional action NLL, future-state probe, optimal `K` | Within-system interventions reproduce Stage A relationships before saturation |
| C. Robot horizon map | Adroit Pen/Door/Hammer/Relocate plus selected BiGym free-space/contact/long-execution tasks; supervision and native `K={1,5,10,20}`, initially `E=1` | success AUC/final, prediction profile, probes, gradient alignment | The control-aligned score predicts optimal horizon beyond data size, task length, frequency, and uncertainty alone |
| D. Method mechanism | Fixed grid, global HD, exact dynamic HD-equivalent weighting, Linear 2:1, generic model dropout, first-token-only, proposed task weights | paired success, oracle regret, token/FLOP budget, calibration | Proposed weights beat the best global objective and matched stochastic/deterministic controls |
| E. Shared multi-task policy | Ablate on at least 8 shared-action tasks, then scale the surviving method to a 40+ task shared policy; hold out tasks/embodiments when fitting the profile | macro success, worst quartile, per-task oracle regret, negative-transfer gap | Improve macro by a preregistered practical margin while improving worst-task or oracle-regret metrics and beating compute-matched MoH |
| F. Generality | Flow Matching plus ACT or Diffusion Policy; state then pixels/language; optional real robot | same metrics plus latency and memory | Mechanism direction survives two architectures and one scaled setting |

Screen stages A-C with three seeds. Expand only preregistered finalists to five seeds
and evaluate selected checkpoints on 100-200 held-out episodes. For the shared model,
use a concrete success gate such as `>=3` macro-success points plus either `>=5`
worst-quartile points or `>=25%` reduction in per-task oracle regret, while preventing
a meaningful regression on long-execution tasks.

### 14.4 Strong-baseline boundary

The method table must include:

- fixed `K={1,5,10,20}` with common valid starts and matched updates;
- best global fixed `K` and per-task oracle `K`;
- global conservative Horizon Dropout and its exact sample-specific deterministic
  equivalent under padding;
- Linear 2:1 and its stochastic prefix equivalent;
- first-token-only `K=20` and generic regularization controls;
- [Mixture of Horizons](https://openreview.net/forum?id=2GQOSf4Y8n) as the closest
  training-horizon baseline;
- [Bidirectional Decoding](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d78dd998f7b9ac79604d47a2d79bb0d-Abstract-Conference.html)
  plus [Denoising-Variance Adaptive Chunking](https://arxiv.org/abs/2606.03847) as
  complementary `E`-selection controls, not explanations of the Pen `E=1`,
  no-ensemble training effect.

### 14.5 Two-GPU execution queue

1. **Completed:** fixed versus conservative Horizon Dropout now has five Pen seeds;
   the final checkpoints were also reevaluated for 100 episodes per seed with `E=1`
   and temporal ensembling disabled.
2. **Completed:** the common-architecture Pen supervision-horizon grid uses
   `K_max=20`, `H={1,5,10,20}`, `E=1`, no temporal ensemble, fixed eval seeds, the
   same 6480 replay origins, sum-one prefix weights, and a new screening seed 5. GPU
   0 ran `H1 -> H10`; GPU 1 ran `H5 -> H20`. Curve mean/AUC favor `H=1`; this rejects
   the current long-supervision interpretation for Pen rather than validating it.
3. **Completed on CPU:** the exact learned linear bottleneck factorial verifies the
   representation/noise effects but fails to show a robust low-noise first-action
   benefit from `H>1`. The next controlled design must introduce preregistered
   control alignment and actual multi-task sharing, not tune more single-task cells.
4. **Completed:** under the same fair seed-5 protocol, GPU 0 compared conservative
   stochastic Horizon Dropout against GPU 1's exact no-padding expected weights
   `[0.125, 0.075 x 4, 0.055 x 5, 0.03 x 10]`. Deterministic weighting is `+0.030`
   AUC higher, so there is no evidence for a stochastic benefit; the direction now
   requires paired replication.
5. **Completed:** replicate the higher-priority `H=1` versus `H=20` contrast on
   seeds 6 and 7, crossing GPU assignment between seeds. This yields three paired
   seeds with seed 5; H1 improves curve/AUC in all three, while the interpretation
   remains Pen-specific until the task map is run.
6. **Only after that mechanism gate:** spend cards on matched FlipCutlery seeds. A
   repaired Flip run costs roughly five GPU-hours, so it should not precede these
   decisive Pen interventions.
