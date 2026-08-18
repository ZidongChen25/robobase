# Dueling A-stream autopsy — did unanchoring kill the advantage stream? (2026-08-18)

**Question.** The unanchoring collapse (Q-span over the 5 sibling bins ~0.76 -> ~0.02
while mean Q rises, `reports/wsrl_armB_20260818.md`) — is it the ADVANTAGE stream of
the dueling critic dying while the V stream stays alive (the dueling "lazy path")?
First time the network is opened; all prior evidence was behavioral.

**Verdict up front: CONFIRMED in its core claim, with a quantified addendum.**
The advantage stream's output atrophies ~12x toward zero (bin-spread 2.89 -> 0.23
logits, ÷38 at level 1; the λ=1 control *grows* it to 3.10) while the value stream
stays alive, *rises*, and post-collapse carries the entire Q readout (Q == E[V] to
within 0.003 — the rising mean Q *is* the V stream). The refuted-branch
alternative ("both streams retain structure, flatness only in the combination") is
false. Addendum: the A stream is dying but not yet numerically zero at +20k
updates, and a parameter-level stream transplant (§5) shows the ~100x readout
collapse factorizes almost exactly into two ~10x factors with the same cause:
A-stream atrophy (healthy V + dying A -> span 0.10) and a quasi-degenerate
post-collapse V distribution that mutes the logit->EV sensitivity, masking even a
fully healthy A stream (dying V-measure + healthy A -> span 0.083).

---

## 1. Code-level answer: where the dueling combines

Critic: `C2FSequenceDistributionalCritic`, `robobase/method/cqn_as.py:914-1067`.
Two *fully separate* recurrent towers that share nothing but the input features
(every critic param is prefixed `advantage_*` or `value_*`):

- **A stream**: `robobase/method/cqn_as.py:1027` (`recurrent_stream("advantage")`)
  -> `advantage_head` Dense, `:1028-1041`, output `[B, seq=16, action_dim, bins=5, atoms=51]`.
- **V stream**: `:1045` (`recurrent_stream("value")`) -> `value_head` Dense,
  `:1046-1059`, output `[B, seq, action_dim, 1, atoms]` — one distribution per
  action dim, broadcast across bins.
- **Combination** `:1060-1064`:
  ```python
  centered_advantages = advantages - advantages.mean(axis=-2, keepdims=True)  # bins axis
  combined = values + centered_advantages
  ```

**The dueling combines LOGITS (per-atom), not expected values.** The softmax over
the 51 atoms and the reduction `Q = sum(softmax(logits) * support)` happen only
downstream (e.g. `cqn_as.py:3523`, and every probe in this repo). Two structural
consequences, both verified numerically:

1. The bin-spread of the A logits is **identical before and after** the
   mean-subtraction — the subtracted mean is constant across bins (per-atom).
   Measured: `a_logit_span == a_centered_span` to all printed digits, every
   checkpoint. So "A-spread before vs after centering" is one number, and the raw
   (pre-centering) head output additionally gives the A stream's absolute
   amplitude, captured via flax `capture_intermediates` on the two heads.
2. Across the 5 bins, **only the A stream varies at all** (V broadcasts over
   bins). Any Q-span must therefore pass through A — but the logits->EV map is
   nonlinear, so A structure at atoms where the combined distribution has no mass
   is invisible in Q. That is exactly what the post-collapse residual turns out
   to be (§4).

## 2. Probe setup

- **Script**: `scripts/dueling_stream_autopsy.py` (raw numbers:
  `reports/dueling_astream_autopsy_20260818.json`); transplant follow-up:
  `scripts/dueling_stream_transplant.py`
  (`reports/dueling_astream_autopsy_20260818_transplant.json`).
- **States**: frozen T1 dataset (collected from the same 100k base checkpoint the
  WSRL arms warm-started from). 128 demo states (16 episodes x 8) from the shared
  demo cache + 128 held-out states (16 x 8) from
  `exp_local/t1_td_mc/frozen_data/` holdout files. Teacher-forced zoom path on the
  recorded action chunk, same convention as every span metric in the repo.
- **Checkpoints** (`eval_checkpoints/`): pre-collapse = 120000 (end of zero-update
  warmup, params == 100k base — the three arms' 120k rows below are bit-identical,
  which doubles as a wiring check); post-collapse = 125000 (seam +2268) and
  terminal (142732 / 141466); control = λ=1 arm at the same steps (142900
  terminal).
- **Compute**: local GPU5 (`GPU-2f044e6a`, CVD by UUID, `MUJOCO_EGL_DEVICE_ID=1`,
  `JAX_PLATFORMS=cuda`, no preallocation); peak GPU footprint 0.5 GB, batches of 32.
- **Sanity**: 3-level demo-state Q-span reproduces the run's own fixed-batch probe
  (`bc_anchor_probe.csv`): 0.7495 / 0.0072 / 0.0036 here vs 0.7696 / 0.0069 /
  0.0032 there (different probe states); chosen-Q 0.106 -> 0.266/0.251 vs
  0.129 -> 0.295/0.280. Collapse fingerprint fully reproduced.

## 3. Main table — streams opened, level 0

Per state group; all numbers are means over 128 states (per-state means over the
16x action-dim heads first). Columns: `a_span` = max-min of A logits across the 5
bins (identical pre/post centering, see §1); `a_std` = std across bins; `|A|` =
mean |raw A logit| (stream amplitude); `E_A-span` = span of E[softmax(A)*support]
(A structure read as values); `E[V]` = expected value of the V-stream
distribution; `|V|` = mean |V logit|; `q_span` = combined Q-span across bins
(the collapse metric); `chq` = Q of the recorded bin.

### demo states, level 0

| checkpoint | a_span | a_std | \|A\| | E_A-span | E[V] | \|V\| | q_span | chq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre (120k, all arms, == base) | 2.892 | 1.046 | 0.899 | 1.838 | −0.436 | 4.93 | 0.873 | 0.093 |
| λ=0 s1 @125k (seam+2.3k) | 0.484 | 0.178 | 0.152 | 0.437 | **+0.238** | 5.39 | 0.017 | 0.242 |
| λ=0 s1 @142.7k (seam+20k) | **0.232** | 0.085 | **0.072** | 0.189 | **+0.262** | 6.40 | 0.009 | 0.265 |
| λ=0 s2 @125k | 0.470 | 0.173 | 0.148 | 0.422 | +0.246 | 5.43 | 0.016 | 0.250 |
| λ=0 s2 @141.5k | 0.260 | 0.095 | 0.081 | 0.209 | +0.264 | 6.47 | 0.013 | 0.267 |
| λ=1 ctrl @125k | 3.044 | 1.099 | 0.943 | 1.921 | −0.456 | 5.12 | 0.887 | 0.088 |
| λ=1 ctrl @142.9k | **3.104** | 1.120 | **0.960** | 2.036 | −0.460 | 5.37 | 0.922 | 0.091 |

### heldout states, level 0

| checkpoint | a_span | a_std | \|A\| | E_A-span | E[V] | \|V\| | q_span | chq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pre (120k) | 2.907 | 1.047 | 0.897 | 1.876 | −0.286 | 4.48 | 0.896 | 0.144 |
| λ=0 s1 @125k | 0.518 | 0.190 | 0.163 | 0.469 | +0.207 | 5.12 | 0.021 | 0.212 |
| λ=0 s1 @142.7k | 0.198 | 0.072 | 0.062 | 0.161 | +0.184 | 6.09 | 0.010 | 0.187 |
| λ=0 s2 @125k | 0.609 | 0.224 | 0.192 | 0.543 | +0.256 | 5.22 | 0.027 | 0.262 |
| λ=0 s2 @141.5k | 0.378 | 0.138 | 0.118 | 0.301 | +0.275 | 6.32 | 0.021 | 0.281 |
| λ=1 ctrl @125k | 3.046 | 1.096 | 0.939 | 1.953 | −0.319 | 4.74 | 0.951 | 0.151 |
| λ=1 ctrl @142.9k | 3.113 | 1.119 | 0.957 | 2.060 | −0.325 | 5.00 | 0.992 | 0.150 |

### level 1 (same states; collapse is *stronger* one zoom level down)

| checkpoint | demo a_span | demo q_span | heldout a_span | heldout q_span |
|---|---:|---:|---:|---:|
| pre (120k) | 2.037 | 0.721 | 2.122 | 0.764 |
| λ=0 s1 @142.7k | **0.053** (÷38) | **0.0009** | 0.055 | 0.0010 |
| λ=0 s2 @141.5k | 0.051 | 0.0008 | 0.078 | 0.0016 |
| λ=1 ctrl @142.9k | 2.181 | 0.736 | 2.283 | 0.928 |

### parameter norms (L2, critic tree)

| checkpoint | \|\|A_head\|\| | \|\|V_head\|\| | \|\|A stack\|\| | \|\|V stack\|\| |
|---|---:|---:|---:|---:|
| pre (120k) | 58.3 | 49.5 | 117.7 | 96.4 |
| λ=0 s1 @142.7k | 49.8 | 42.8 | 103.8 | 87.3 |
| λ=0 s2 @141.5k | 49.9 | 42.7 | 104.0 | 87.3 |
| λ=1 ctrl @142.9k | 62.9 | 54.2 | 120.3 | 98.3 |

Both stream's weight norms shrink only ~12-15% under λ=0 (and grow slightly under
λ=1) — the A stream's *weights* are not zeroed; its *function output* is. The
death is in what the stream computes, not a trivially visible weight collapse.

## 4. Reading the table

1. **The A stream is dying.** Its bin-spread drops 6x by seam+2.3k and 12.5x by
   seam+20k (2.89 -> 0.48 -> 0.23 logits; L1: ÷38), monotonically, in both seeds
   and both state groups, while the λ=1 control *grows* it (2.89 -> 3.10). The
   raw amplitude |A| falls in lockstep (0.90 -> 0.07; control 0.96): the whole
   head output is shrinking toward zero, not merely equalizing across bins.
2. **The V stream stays alive and absorbs the readout.** E[V] swings from −0.44
   to +0.24..0.28 — and post-collapse chosen-Q equals E[V] to within 0.003 at
   every post-collapse checkpoint x group (e.g. 0.2380 vs 0.2423; 0.2617 vs
   0.2645). The "mean Q rises while span collapses" fingerprint from the run
   logs is literally the V stream's number; Q(s,a) has degenerated to V(s).
   |V| logit amplitude grows (4.9 -> 6.4) while the A stream's shrinks.
3. **The residual A structure is masked by the combination.** Post-collapse the
   A-only value readout still spans ~0.19 but combined Q-span is ~0.009 (~20x
   attenuation vs ~2x pre-collapse). Since V is constant across bins, this can
   only happen if the surviving A differences sit at atoms where
   softmax(V + A) has no mass. §5 quantifies this and settles causality.

## 5. Stream transplant (parameter-level causality)

(see `scripts/dueling_stream_transplant.py`; pre = s1@120k, post = s1@142.7k;
encoder params travel with the donor named in the config; `eff_a_span` =
first-order visible A-span under the V measure, span_bins(Cov_{p_V}(support, A_bin)))

Every critic param is prefixed `advantage_*` or `value_*`, so a stream can be
swapped wholesale between checkpoints. Level 0, means over the same 128+128
states:

| config (encoder donor) | demo q_span | demo eff_a_span | demo a_span | demo chq | heldout q_span | demo L1 q_span |
|---|---:|---:|---:|---:|---:|---:|
| pre V + pre A (pre) | 0.873 | 1.180 | 2.892 | 0.093 | 0.896 | 0.721 |
| post V + post A (post) | 0.0091 | 0.0086 | 0.232 | 0.265 | 0.010 | 0.0009 |
| **pre V + post A** (pre) | **0.101** | 0.106 | 0.267 | −0.401 | 0.113 | 0.019 |
| **post V + pre A** (post) | **0.083** | 0.041 | 2.911 | 0.272 | 0.075 | 0.064 |
| post enc + pre critic (post) | 0.877 | 1.202 | 2.911 | 0.091 | 0.890 | 0.708 |

Four independent facts fall out:

1. **Encoder is exonerated.** Post-collapse encoder + fully pre-collapse critic
   = 0.877, indistinguishable from pure pre (0.873). The collapse lives entirely
   inside the critic's two streams.
2. **The dying A stream alone destroys ~88% of the span.** Healthy V, healthy
   encoder, transplanted post-A: 0.87 -> 0.10. The A stream's atrophy is a real,
   causal component — not an artifact of the readout.
3. **The collapsed V distribution alone masks ~90% of the span.** A fully
   healthy A stream (bin-spread 2.911, unchanged) against the post-collapse V
   yields only 0.083, because the post V per-atom distribution has become
   quasi-degenerate: the identical A logits produce `eff_a_span` 0.041 under the
   post-V measure vs 1.18-1.20 under the pre-V measure (29x suppression from the
   measure alone). The surviving A structure sits at atoms the combined
   distribution no longer visits.
4. **The two factors are multiplicative and complete.** (0.101/0.873) x
   (0.083/0.873) x 0.873 = 0.0096 ~= the measured pure-post 0.0091. In the
   post-collapse regime the first-order approximation is exact
   (`eff_a_span` = 0.0086 vs measured q_span 0.0091), confirming Q-span is fully
   accounted for by Cov under the V measure; pre-collapse the A perturbations
   are beyond first order (1.18 vs 0.87), i.e. a healthy critic's bins actually
   reshape the distribution rather than perturb it.

## 6. Verdict

**Hypothesis CONFIRMED in its core claim; the refuted-branch alternative is
false.** Opening the network shows the advantage stream genuinely dying — its
raw output amplitude falls 0.90 -> 0.07 and its bin-spread 2.89 -> 0.23 (÷38 at
level 1) over 24k λ=0 updates, monotonically, in both seeds and both state
groups, while the same protocol with the hinge kept (λ=1) *grows* both numbers —
and the value stream staying alive and absorbing the critic: E[V] rises −0.44 ->
+0.26 and post-collapse Q(s,a) equals E[V] to within 0.003, so the behavioral
fingerprint "span ÷240 while chosen-Q rises 0.13 -> 0.28" is literally the V
stream replacing the A stream in the readout. The addendum the probe adds to the
hypothesis: the A stream is dying but not yet zero, and the last order of
magnitude of readout flatness comes from the combination — the post-collapse V
per-atom distribution is so peaked that even a transplanted fully-healthy A
stream can only move Q by 0.083 (vs 0.87 under the healthy V) — with the two
factors multiplying to exactly the observed 0.009. Both factors are the same
disease seen from the two streams: with the hinge removed, nothing in the
1-step-TD objective pays for action-discrimination, so the A head's output decays
toward zero *and* V hardens into a confident state-value distribution; the
dueling architecture then routes all remaining learning down the V ("lazy") path.
Practical corollary for the roadmap arms: any rescue must reward sibling
separation itself (in-sample masking, calibrated one-sided push-down) — reviving
the A head's output scale alone would be insufficient while the V distribution
stays degenerate, and vice versa.
