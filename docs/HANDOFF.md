# Current compiler handoff — 2026-10-23 shared-window repair

Retained compiler SHA `51bd61bded75a2e132c94f963974f29c93783135283b8c1d028c30e848da4f00`
now passes **15/15 official RTX 5090/SM12.0 published cases**, including m01.
RTX 4090/SM8.9: 122/122 compact cases, 26 sanitizers; 32 CPU compiler tests pass.
The current source, gate and evidence identities are in
[the repair record](shared-window-fix.md). No paired speedup or generated-case
SM12.0 claim. No generated cases or test types were added.

The authoritative remote handoff, relative to the privately configured research
root, is `/campaigns/campaign-20261023-12h-v2/m01-shared-window-repair-v1/HANDOFF.md`;
`FINAL.json` beside it binds retained source, evaluation and official results.
Canonical `/regression-suites/current.json` and `/retained/current.json` now name
the repaired compiler and unchanged 122-case policy. No live owned GPU/preparation
jobs; bounded repair run closed; discovery off. The 16-request official singleton
batch `/calibration/official-20261023-shared-window-v1/` is consumed; do not retry.
Stay on `n00bcak`, with no GitHub publication or remote Git.

The kernel skills share [architecture and submission guidance](compute-capability.md).

---

The historical campaign handoff follows. Its f47 source and 14/15 official result
are superseded by the verified repair above; its historical performance records
remain scoped to the original source, device and experiment.

# Compiler handoff — 2026-10-23

Purpose: resume from a synchronized retained compiler and compact regression suite.
No lifecycle record. The owner requested consolidation, ignored experiment output,
a handoff and a local commit. Stay on `n00bcak`; do not publish or run remote Git.
Runtime and research-root configuration remain private and ignored. Paths below
are relative to that configured research root unless explicitly repository paths.

## Retained state and synchronization

- Repository `compile.py` and remote retained compiler SHA-256:
  `f47c21b0bb2818df21cd218573cd3cd1231f87955ee78c43a779f5b0be6ebac0`.
- Frozen source identity:
  `549b6047f91818ea876deb76adfc88cabc36f554c05ab2bc51b2c961aa9143f2`.
- Canonical manifest: `regression-suites/current.json`.
  `retained/current.json` aliases this manifest; `retained/compile.py` aliases the
  immutable compiler it names. `retained/versions.json` inventories 18 distinct
  compiler hashes: one active and 17 historical. Frozen objects remain intact.
- Repository `tests/gpu-regression.json` is byte-identical to
  `regression-suites/compact-v4/policy.json`, SHA-256
  `8b02e13947b0ac2a55f13233176cf992d96b7b6a8b4a9daf925c733df7752f60`.
- Routine selection: **122 cases, 15 published + 107 generated**. The campaign's
  previous 263 generated cases are historical, with 156 excluded from routine
  replay. Shape/seed permutations and overlapping schedules do not accumulate.
  The combined older archive contains 1,114 distinct case identities; it is not
  the routine suite. No input/reference/tolerance or generator was rewritten.
- The old default named an obsolete compiler while campaign helpers replayed
  the cumulative 278-case list. Source-checked shared defaults now prevent this
  split. Maintenance evidence: `campaigns/campaign-20261023-12h-v2/` under
  `test-consolidation-v3/` (superseded selection); corrected IR/coverage audit and
  existing receipt verification are under `regression-suites/compact-v4/`.
- The 73-case selection was too aggressive: it dropped distinct remote behaviors.
  [The correction](gpu-regression-audit.md) restores them and hash-binds a
  case-to-witness map for all 278 campaign cases. For the 156 excluded cases,
  154 have matching sample-normalized AST topology and two are ordered statement
  subsequences in broader retained programs. This is not numeric equivalence.

```sh
python -m tools.regression_suite --root "$RESEARCH_ROOT" --out "$NEW_CASE_LIST"
```

This selector verifies frozen records/files and rejects stale source or policy
identities. Use explicit `--expand-version` for an affected mechanism; `--full`
is an explicit archive replay. New registrations draw one fresh case per selected
family by default, with separate independent confirmation after source freeze.
Frozen older protocols keep their original sampling. A changed compiler or
evaluator still needs matching all-15 gates before timing; synchronization itself
creates no gate. Installation requires discovery off and no owned work.

## Verified results and limits

The revised nine-hour campaign closed after 33 evaluated variants, retaining ten
correctness repairs and three speed improvements. Its generation budget recorded
82 catalogued types plus one reserved type (83/100); cases and types differ.
Repairs cover 64-bit shifts, Load and nested Tile Init lifetime, opaque expression
keys, mutable/Select scope, partial-CTA uniform loops, BOOL SSA, nested implicit
seeded folds, softplus negative tails and small erf composition. Speed work covers
KEY scheduling/batching, prefetching and structurally proven read-only vector loads.
No fixture names, values or shapes select a solution.

The retained source passed **15 published + 263 generated = 278 cases** on RTX
4090, with isolated guarded confirmation and required sanitizer evidence. The
compact 122-case set was verified as a subset of that exact source-bound receipt;
consolidation ran no new GPU tests. Relevant immutable identities:

| Record | Identity |
|---|---|
| Evaluator suite | `0fa85c2e98ef65616058b558c1008d80f0f3530ae13cfdf276774050f8a2d5ce` |
| Published gate | `94e8ba132630adba4cba9be76c19fc733ae85b37eacab0344a497ff3dfbef5f1` |
| Expanded correctness | `a33674b0f2a33a317518c55b7d363fceb1b7e0d31d0c23a9ad16fd55f3e62eba` |
| Independent 47-case timing | `eedc220eb9a06d107171a6745608ea49c8a88d7ef6c002372bf35e1a09b894d3` |
| Official calibration | `0d413f446b63d00179e05ac3d9ee0de19964bd7a0a36fca48ee6deedb29c4789` |

The last independently confirmed speed change reduced `a01_attention` from
66.56 to 18.432 microseconds; the other 46 timing cases were inconclusive, with
zero resolved regressions. Raw per-case increases remain in the receipts. Earlier
confirmed epochs reduced `r02_big_sum` 20.48→7.76 and `r03_mean` 12.288→6.144
microseconds; do not multiply gains from different comparisons. Detailed paired
per-case measurements and calibrations live in the immutable timing record.

**Official RTX 5090: 14/15 pass.** `m01_matmul_512` failed 262,069 of 262,144
outputs, maximum absolute error 144.748085 at rtol=atol=0.001. The saved request
uses exact frozen inputs, argument order `x1,x0,matmul`, grid (128,1,1), block
(512,1,1), 73,728 shared bytes and no zero_outputs. Independent NumPy float64
`x0 @ x1` cast to float32 exactly matches the saved fixture reference. The cause
is unresolved: these checks do not establish a compiler, assembler or runner
cause. All sm_120 performance remains unqualified under the all-15 rule.

The single official calibration batch used 16 HTTP attempts. Its numerical stop
and continuation of nine unstarted cases are preserved; do not silently retry.
Exact requests/results: `calibration/official-20261023-retained-v1/`. Local copies
are uncommitted under `experiments/2026-10-23-campaign-current/official-endpoint/`.

A held `4ff…` compiler (source `212e…`) passed 278 cases but its incremental
33/47-case comparisons were inconclusive, so it was not retained or copied over
the compiler. `c9b17…` had only static evidence. The proposed F16 repair was
rejected after the alleged bug was disproven; the E4 protocol remains deferred.
See the detailed campaign handoff for full identities and rejected mechanisms.

The MFU audit found no unit conversion error. Reported MFU is useful FLOPs divided
by whole-device peak FLOPs times latency, not a hardware utilization counter.
Small launch grids and excluded SFU operations can therefore show low MFU.

## Maintenance validation and next action

- 32 compiler tests passed on the configured Linux CPU runtime, including PTX
  assembly of every published problem for sm_89 and sm_120. CPU job evidence:
  `campaigns/campaign-20261023-12h-v2/commit-validation-v1/`.
- 34 selector and research-pipeline CPU tests passed locally. This is separate
  from GPU correctness; no new GPU execution or endpoint request was performed
  during consolidation.
- An external amendment (`ac5017b`) restored the historical experiment archives
  to the index and reverted the broad experiment ignore rule. This coverage
  correction preserves that amendment and leaves generated audit output
  uncommitted. Private runtime configuration remains ignored. This snapshot
  does not claim that the official all-15 gate passes.

The campaign is closed, discovery is off, and no owned GPU/preparation job remained
at synchronization. Inspect actual recorded jobs before any restart. Detailed
resumable campaign state: `campaigns/campaign-20261023-12h-v2/HANDOFF.md`; root
`START.md` points to the synchronized state.

Next action: inspect the saved official m01 PTX/request reproducer and identify
where its result diverges before changing the compiler. A future research run or
endpoint batch requires explicit activation; reading this handoff starts neither.

## Local commit organization — 2026-10-23

The retained compiler changes are now split into 14 source commits, followed by
the preserved compact-tooling and remote-coverage audit commits. See
[the confirmed experiment index](successful-experiments.md) for the sequence,
source/evidence identities, historical coverage limits and local recovery ref.
The split reproduced the complete prior `b900be8` tree exactly; compiler bytes,
audited tests, owner-amended archive tracking and ignore rules are unchanged.
No rejected source, generated test case or new GPU/endpoint result was imported.
