# Confirmed remote compiler experiments

The latest remote campaign is organized below as local commits on `n00bcak`.
Only evaluator-promoted source snapshots were imported. Each commit records the
full source, compiler hash, published gate, experiment, promotion decision and
correctness/timing evidence identities in its message. Resolve evidence through
the privately configured research root; no remote Git operation was used.

The compiler is still `f47c21b0bb2818df21cd218573cd3cd1231f87955ee78c43a779f5b0be6ebac0`.
Its RTX 4090 correctness receipt covers 278 cases; official RTX 5090 remains
**14/15**, with `m01_matmul_512` unresolved. These are confirmed local successes,
not an official all-15 release or hidden-set correctness claim.

## Commit sequence

| Commit | Retained change | Fresh confirmed cases at promotion | Source |
|---|---|---:|---|
| `e35120f47c` | Integrate independently confirmed general PTX lowering baseline | 47 | `d883119296f9` |
| `f0f4c99621` | Correct 64-bit shift counts and lexical load bindings | 102 | `c8fde137f1c9` |
| `3926b37eef` | Preserve nested Tile initialization scope | 104 | `f43a5e79984f` |
| `dec25e7373` | Reject incomplete recursive expression cache keys | 108 | `561c1f3ba660` |
| `559b35d4a6` | Detach shadowed mutable SSA carriers | 117 | `d9a29034ab72` |
| `b6349b92aa` | Batch ordered attention KEY loops | 125 | `16c72ed98ae2` |
| `a8f0ec1530` | Snapshot Select results before rebinding | 127 | `5feed7f75124` |
| `bbc2b56918` | Keep partial-CTA uniform loops participating in barriers | 145 | `d084afda4e25` |
| `be24af639d` | Prefetch serial per-lane reductions in groups of eight | 163 | `973c29e6a323` |
| `612e565409` | Lower scalar BOOL results and carried folds | 172 | `6481a0c89b84` |
| `687ebea4ea` | Restore seeded fold scopes in generic and KEY loops | 209 | `935b5bbac8fd` |
| `ab1dc9f11b` | Preserve negative-tail F32 softplus precision | 211 | `f0f5e4bf8d89` |
| `e4dca874f9` | Preserve normal small-input F32 erf values | 221 | `ae7efd5ad535` |
| `0c0bb8a392` | Vectorize proven read-only loads in attention dots | 246 | `549b6047f918` |

Every row has an exact source/suite-bound all-15 gate, fresh isolated GPU
confirmation, intact input/guard checks and required sanitizer evidence on RTX
4090. Counts describe the evaluation version at that promotion. Older snapshots
are historical milestones; later repairs expanded coverage, so they are not
asserted to pass the final 278-case corpus. The baseline row lists the 47-case
fresh confirmation receipt, separately from its larger historical replay.

## Retained performance findings

- Ordered KEY batching: independently confirmed `a01` ratio approximately 1/3
  and `a02` ratio approximately 0.4423; 45 other cases inconclusive, no resolved
  per-case regressions. Evidence `c32a240fbaebb8373c662cfa6af8aac884dfafe58b79b452c26b89fd62763efe`.
- Serial reduction prefetch: `r02` 20.48→7.76 microseconds and `r03`
  12.288→6.144 microseconds; 45 others inconclusive, no resolved regressions.
  Evidence `2cacfe67de24292ef50c37832a14a6e940cb40777eb97a114ab4d3f851207d7c`.
- Proven read-only vector loads: `a01` 66.56→18.432 microseconds;
  46 others inconclusive, no resolved regressions. Evidence
  `eedc220eb9a06d107171a6745608ea49c8a88d7ef6c002372bf35e1a09b894d3`.

These are separate paired comparisons; do not multiply their gains. Raw per-case
median increases, noise controls and inconclusive outcomes remain in the remote
evidence. “No resolved regression” does not mean every raw median decreased.
Correctness-only commits make no additional speed claim.

## What stayed remote

Unpromoted Q-cache, phase/tree/pair and prefix experiments, incomplete sweeps,
failed prototypes and disproven repair proposals were not imported as source.
Their records remain in the research store. No logs, inputs, cubins, SASS dumps
or additional generated testcases were added by this organization. The routine
suite remains the audited 122 cases described in [the coverage audit](gpu-regression-audit.md).

## Verification and recovery

Before transfer, all 14 source objects and their evaluator-issued decisions,
all-15 gates and fresh correctness receipts were verified remotely. The first
organized baseline also passed the maintained 32 compiler CPU tests, including
published PTX assembly. Its record is
`campaigns/campaign-20261023-12h-v2/commit-organization-v1/baseline-unittest.json`.
The same test file and final compiler already had the recorded 32-test pass.

The 14 compiler commits and replayed tooling/audit commits reproduce the previous
`b900be8` tree exactly (`c7668df7327f34ed1b2aae94314e55f1aa03b674`). Only this index
and its handoff pointer are added afterward. The owner-amended archive tracking
and ignore rules are preserved. No new GPU execution, endpoint request or
publication occurred.

The prior local history is retained at
`refs/archive/emmy-before-organization-20261023-b900be8`.
Durable organization metadata is under
`campaigns/campaign-20261023-12h-v2/commit-organization-v1/`.

## Subsequent shared-window repair

The generic dynamic shared-window address fix from available `origin/main` commit `2647806`
was ported into the retained compiler and independently validated: 122/122 SM8.9
cases, 26 sanitizer checks,32 CPU tests and 15/15 official SM12.0 published cases.
See [the source-bound repair record](shared-window-fix.md). This is a correctness
repair with no new performance claim; all prior experiment records stay intact.
