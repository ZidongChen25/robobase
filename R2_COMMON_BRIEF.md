# R2 Common Brief — per-line variant extraction

You are one of ~12 parallel agents. Each owns ONE research line. Read
`CQN_REFACTOR_PLAN.md` first. Base commit: `ff9dfbf` on branch
`refactor/cqn-as-decouple`, worktree `/home/zc1525/robobase_jaxflat_refactor`.

## Ground rules (violations ruin the whole batch)
- Work ONLY in the worktree. Never touch `/home/zc1525/robobase_jaxflat`.
- CPU ONLY: every python/pytest run uses `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/zc1525/robobase_jaxflat_refactor /home/zc1525/robobase_jaxflat/.venv/bin/python`.
- Create ONLY your three files (below). NEVER edit: `robobase/method/cqn.py`,
  `cqn_as.py` (FROZEN), `cqn_research.py`, `cqn_as_research.py`, `factory.py`,
  `method/__init__.py`, `workspace.py`, any existing yaml/test, the plan docs,
  or any other line's `cqn_as_<line>.*` files (they are being written
  concurrently by other agents).
- NO git commands that mutate state (no add/commit/stash/checkout). Read-only
  git (`show`, `log`, `diff`) is fine and encouraged.

## Your deliverables (exactly three files + a report)
1. `robobase/method/cqn_as_<line>.py` — the variant.
2. `robobase/cfgs/method/cqn_as_<line>.yaml` — start from a byte-copy of
   `robobase/cfgs/method/cqn_as_official.yaml`; change `name:` to
   `cqn_as_<line>`, `_target_:` to your class, and APPEND only your line's keys
   (copy their names/defaults/comments from `cqn_as.yaml`).
3. `tests/unit/test_cqn_as_<line>_variant.py` — see verification.

## The pattern (mandatory)
Your variant file imports the FROZEN pristine classes and subclasses them:
```python
from robobase.method.cqn_as import CQNAS, cqn_as_spec_from_cfg, CQNASpec
class CQNAS<CamelLine>(CQNAS):
    ...
```
Override methods by COPY-PASTING the pristine method body (from
`robobase/method/cqn.py` / `cqn_as.py` — NOT from the research files) into your
subclass, then applying ONLY your line's changes to the copy. Lines whose code
lives in the CQN base (`_build_update_fn`, `update`, `act`) are still handled by
overriding in your CQNAS subclass — Python MRO makes the override win. Extend
the spec by copying the `CQNASpec`/`cqn_as_spec_from_cfg` pattern into your file
(new frozen dataclass + `cqn_as_<line>_spec_from_cfg`). Your `__init__` takes
the pristine args plus your line's flags (keyword-only, defaults = OFF).

## Where your line's code lives
The reference implementation of every line is `robobase/method/cqn_as_research.py`
+ `cqn_research.py` (the full research monolith, byte-equal to the pre-refactor
files). Find your code by: (a) grepping your flag names there and in the spec
dataclass; (b) reading your flags' comments in `robobase/cfgs/method/cqn_as.yaml`;
(c) `git log -S<flag_name> --oneline -- robobase/method/` for the introducing
commits, then `git show <commit>` to see the original isolated diff — often the
cleanest statement of what your line changes. Replicate the CURRENT
(`ff9dfbf`-era) semantics, using history only to locate code.

## Verification (your test file must encode 1 and 2)
1. **Flags-off ≡ pristine**: same seed, same synthetic batch → your class (all
   line flags at defaults) and pristine `CQNAS` produce `critic_loss` equal to
   `atol<=1e-6` and identical param tree shapes after one `update()`. Mimic the
   synthetic-space/batch construction in `scripts/refactor_equivalence_check.py`.
2. **Flags-on sanity**: with your line's flags enabled (pick the canonical
   values its waves used), one `act()` + `update()` runs, all metrics finite,
   and your line's expected metric keys appear.
3. **Flags-on ≡ research (best effort)**: configure `cqn_as_research.CQNAS`
   with the same flags on and compare `critic_loss` (`atol<=1e-5`). If this
   fails ONLY for identifiable RNG-stream-ordering or reassociation reasons,
   document the cause in your report instead of forcing it — but a silent
   numeric mismatch you cannot explain means your extraction is wrong.
4. Run your test file plus `scripts/refactor_equivalence_check.py` (must still
   pass) before reporting.
Existing research-era tests for your line (listed in your task prompt) are the
behavioral spec — read them; do NOT modify them; you may adapt their assertions
into your own test.

## Coupling protocol
If part of your line's code is inseparably entangled with ANOTHER line's flags
(shared helper rewritten by both, one flag reading another's state), implement
the separable part and DOCUMENT the entanglement precisely (file:line in the
research monolith, which foreign flags, why inseparable). Never import from
`*_research` in your variant, never silently absorb another line's behavior.

## Report format (raw text)
1. Files created with line counts. 2. Verification results 1-4 verbatim
(numbers, not adjectives). 3. Factory registration snippet: the exact mapping
entry (`"robobase.method.cqn_as_<line>.CQNAS<CamelLine>": "cqn_as_<line>"`),
spec-builder import, and construction kwargs your class needs beyond pristine —
written so an integrator can apply it without reading your file. 4. Coupling
notes. 5. Anything about your line's research-era behavior you had to interpret
or that looked buggy (report, don't fix).
