# Adversarial audit 2: typed TF32 tensor-core checker

Date: 2026-10-23. Scope: read-only audit of `tensor_core.py`, `compile.py`, and
the current TF32 emitter. No compiler, test, fixture, evaluator, metadata, or
existing evidence was edited. All new files are under
`experiments/tensor-core-audit-agent2/` (plus `/tmp` scratch); the report below
reproduces the exact 20-probe run.

This audit attacks the repairs described in the prior audit: direct-coordinate
drift, operand/output aliasing, and rotating `cp.async` replay. It also tests
barrier participation, wait-depth semantics, staging orientation, nested-K
spelling, atom grids, split-K, and valid name/ABI variations. It makes no
benchmark or performance claim.

> **Historical record.** This report is bound to the pre-repair source hashes
> in the next section. Its confirmed false accepts and staged-transpose
> rejection were repaired; see
> [Post-audit disposition](#post-audit-disposition-repair-replay). The pre-fix
> "actual" column is not a description of the current checker.

## Reproduce

```sh
cd /home/ada/Projects/emmy-hackathon-2026
.venv/bin/python experiments/tensor-core-audit-agent2/probes/run_audit.py
.venv/bin/python -m unittest tests.test_tensor_core -v
```

The first command classifies each probe, saves the exact typed IR and metadata
for compiled probes, runs the real `compile.py`, and records PTX hashes and MMA
counts. It prints one JSON row per probe and writes
`probes/results.json`. The second command is the existing focused regression
suite; it passed 10/10 when this audit ran. The current suite has 13 tests and
passes separately; do not read the historical 10-test count as current.

## Source identities

```text
compile.py                 996cf5fcd1b6fd70608d384856e97a9057f20b06b55b99087976c89d947c08d4
tensor_core.py             9761280077230a4970d027d4b63761ecc5d3d225af4adccb512b6d29c847932c
tf32_emitter.py            6a9e1af929985911d97de3ab9d967326dd1db0b57a849427b595ed79eb655e9d
tests/test_tensor_core.py  8647478e2e645e04c7fae1aec5eb421ecec0b314f0fd893246d86f5c8e47c97d
probes/construct.py        bbafb54eb3ffea1108deb71f028224bcb1cf9fa01fd5070a0937cfaade03d3d2
probes/run_audit.py        e6ad1538bc90b150581ea542020d672813f00e8960836294b1b745f2587be229
probes/results.json        e7daf3ac898dc093eff9c73700f25fc99d4a5acfa377f64c4420cd328018c5f1
```

Verify with `sha256sum <path>`; the hashes above bind the audit inputs and probe
outputs to the pre-repair checker and generator. The report's current hash is
recorded in the consolidated evidence summary rather than self-referentially
here. The audit directory contains 20 compiled or
classified typed cases with their own `case.ir.json`, `meta.json`, and, where
compiled, `out.ptx` and `stderr.txt`.

## Expected-vs-actual table

At the time of this audit, `Plan` meant the checker emitted or would emit the
tensor path;
`UnsupportedMatch` means a recognized contraction fails a proof; `NoMatch`
means the region is not a supported contraction. Expectations are based on the
contract and the moral semantics, not on the checker's answer.

| id | expected | actual | agrees | evidence / moral semantics |
| --- | --- | --- | --- | --- |
| P01-unusual-names-direct | Plan | Plan | yes | Arbitrary SSA/buffer names, reordered ABI; 3 MMA, rc 0 |
| P02-shared-b-transpose | Plan | Plan | yes | B-transposed shared staging; 3 MMA, `cp.async` present |
| P03-2x1-atom-grid | Plan | Plan | yes | 32x8 tile, two warps, 2x1 atoms; 3 MMA |
| P04-1x2-atom-grid | Plan | Plan | yes | 16x16 tile, two warps, 1x2 atoms; 3 MMA |
| P05-cta-dependent-direct-read | NoMatch | NoMatch | yes | CTA 1 reads the wrong A row; correct fail-closed |
| P06-inert-unused-load | Plan | Plan | yes | Unused load cannot change the contraction; 3 MMA |
| P07-output-aliases-A | NoMatch | NoMatch | yes | Output pointer aliases A; correct rejection |
| P08-nested-k-spelling | Plan | NoMatch | **no** | Nested `k0`/`k1` loops enumerate K=0..15; false reject |
| P09-phase-remapped-k | NoMatch | Plan | **no** | A-phase repeats K=0..7; accepted plan reads K=0..15 |
| P10-m01-output-alias | NoMatch | NoMatch | yes | m01 output aliases `x0`; correct rejection |
| P11-m01-wait-group-3 | UnsupportedMatch | UnsupportedMatch | yes | Unsafe wait depth rejected |
| P12-m01-duplicate-fill | Plan | Plan | yes | Identical duplicate copy is inert; 3 MMA |
| P13-m01-shifted-b-copy | UnsupportedMatch | UnsupportedMatch | yes | Wrong B source cells rejected |
| P14-m01-interior-cta-only-copy | UnsupportedMatch | Plan | **no** | Only CTA 1 shifts B (in-bounds); false Plan, 3 MMA, no `cp.async` |
| P15-m01-partial-barrier | UnsupportedMatch | Plan | **no** | `bar.sync` count=32 cannot order 512 threads; false Plan |
| P16-m01-wait-group-1-safe | Plan | Plan | yes | One-group wait window is still correct for this stage schedule |
| P17-split-no-atomics | UnsupportedMatch | UnsupportedMatch | yes | Non-atomic duplicate output rejected |
| P18-split-with-atomic | UnsupportedMatch | UnsupportedMatch | yes | Atomic split-K falls outside this emitter |
| P19-direct-a-transpose | Plan | Plan | yes | Direct A-transpose; 3 MMA |
| P20-staged-a-transpose | Plan | NoMatch | **no** | Valid staged (K,M)->(M,K) orientation; false reject |

Summary: 20 probes; expected 10 `Plan`, 6 `UnsupportedMatch`, 4 `NoMatch`;
actual 11 `Plan`, 4 `UnsupportedMatch`, 5 `NoMatch`. Disagreements:
P08, P09, P14, P15, P20.

## Ranked findings

### 1. False Plan for interior-CTA-only pipeline mutation (P14)

`_trace_pipeline` traces only `bids = (0, grid_size - 1)` (and one CTA when the
grid is one). The affine arguments justify sampling endpoints for maps they
actually prove, but the copy source is allowed to contain `bid` through a
`TernaryExpr`; the copy replay is the proof of the copy map, so it must cover
every launched CTA (or independently prove the copy map uniform in `bid`).

Reproducer: mutate every m01 B `CpAsyncCopy.src_index[1]` to
`ternary(bid == 1, col + 1, col)`. The copy is in bounds, but CTA 1 consumes a
different B tile. The checker returns `Plan`; `compile.py` emits 3 MMA and
returns 0. Its PTX hash is exactly the unmodified tensor PTX
`84e26af55cd3c696a891d583a04e13b20d48a2f349223f8140311e3de9f14b9b`.
`stderr` is empty and stdout is PTX only. The emitted result is not the source
IR result for CTA 1.

Recommendation: trace all CTAs, or require the copy/compute coordinate forms to
be free of CTA-dependent `TernaryExpr`/conditional atoms unless separately
proved uniform. Endpoint sampling is insufficient evidence for a `Plan`.

### 2. Staged-index stripping accepts a semantically different K map (P09)

`_without_stage_terms` removes any atom accepted by `_is_stage_phase`. On a
direct global operand, the phase test accepts `(k / 8) % 2`, so an A K index
`k - 8 * ((k / 8) % 2)` is stripped and compared as if it were `k`. The actual
index repeats K=0..7 for iterations 8..15, while the emitted plan reads K=0..15.
The checker returns `Plan` and emits 3 MMA. This is a false accept independent
of staging: the index is a direct operand coordinate and must equal the
reduction coordinate, with no slot term available to remove.

Recommendation: only strip stage-phase terms from shared-buffer indices that
are joined to a proved copy/stage slot, and only after proving the slot term is
balanced by the copy producer and visible consumer. A direct operand must never
be allowed to hide a non-unit/modulo K map as a stage atom.

### 3. Collective participation is not checked (P15)

Both simple shared staging and the rotating pipeline treat any `Sync` with
`warp=False` as a full CTA barrier and ignore `count`. Setting m01's barriers to
`Sync(count=32)` still returns `Plan`; a 32-thread named barrier cannot order a
512-thread CTA. The SASS/PTX then contains a partial barrier, while the producer
and consumer can race. The same defect reproduces on the simple staged corpus
case. `warp=True` barriers are correctly rejected, but a partial named barrier
must be rejected or placed in a separately proved producer/consumer collective.

Recommendation: require `count is None` for a CTA-wide ordering proof, or prove
the exact participating thread set. Record the barrier id/count in the plan
proof.

### 4. Valid nested-K spelling is rejected (P08)

A contraction written as two nested loops whose indices form `k1 + k0`
(enumerating K=0..15) is semantically valid, but the recognizer requires the
innermost loop variable itself and reports `operand does not have exactly one K
coordinate`. `compile.py` returns 1 with `Unknown IR value k` while lowering
the fallback, so the same typed IR cannot be compiled at all. This is a false
reject and a fallback robustness issue, though it does not affect the published
fixtures.

Recommendation: accept a proved affine bijection from the nested reduction
variables to K, including a literal-strided outer loop; keep the exact coverage
proof. At minimum, make the SIMT fallback lower the nested spelling instead of
failing on a stale variable name.

### 5. Valid staged A-transpose is rejected (P20)

`Case(..., staged=True, a_transpose=True)` produces a valid contraction with a
(K,M) global A view copied into a row-major (M,K) shared tile. The checker's
shared-row role test compares the computation's shared K dimension to the
global orientation and reports `shared row operand has an unexpected K
dimension`; it returns `NoMatch` and compile emits the SIMT fallback. The direct
A-transpose case (P19) is correctly `Plan`, so the failure is in the shared
orientation proof, not the contraction.

Recommendation: prove the shared tile's local orientation from the copy map and
the compute read independently, then map it to the global (K,M) view. Do not
force the local shared index's K coordinate to equal the global one's.

### 6. Lower-priority reason/taxonomy observations

- P18 (atomic split-K) is correctly rejected but its reported reason is `CTA
grid does not exactly factor the output tile grid`, not the atomic/split-K
limitation. The final verdict is safe; the diagnostic is noisy and should be
improved.
- The previous audit's `a05`/`a16` taxonomy differences remain: several
already-recognized but out-of-tile candidates are reported as `NoMatch` rather
than `UnsupportedMatch`. This is fail-safe (never a false Plan), but makes the
three-way classification less informative. It is not ranked as a correctness
defect here.
- `wait_group 0/1/2` on m01 all return `Plan`; the trace's monotone visibility
model is sufficient for that schedule (it does not prove a newer group is
needed), and the probe with group 1 is retained as a correctly handled hard
case. This does not excuse the count/CTA gaps above.

## Correctly handled hard cases

- P01: arbitrary names and reordered ABI still produce a Plan with the exact
  three MMA instructions and no renderer delegation.
- P02, P03, P04: shared B-transpose and both 2x1/1x2 atom factorizations remain
  correct and emit the expected tensor instruction.
- P05 and P07/P10: direct CTA drift and operand/output aliasing fail closed.
- P11 and P13: unsafe wait depth and an ordinary shifted copy fail closed; the
  latter emits no MMA and falls back with PTX hash
  `edd8b7ef364312432c87e70285d55465128bd7538b0c7cb35a6b753d30a8e390`.
- P12: an exact duplicate fill is correctly treated as inert.
- P16: the safe wait window remains a Plan.
- P17/P18: split-K without atomics and atomic split-K never become Plans.
- P19: direct A-transpose is recognized.

The focused suite ran with
`.venv/bin/python -m unittest tests.test_tensor_core -v` and passed all 10
tests. This audit did not rerun the 15 published cases or GPU execution; those
are outside this read-only checker audit and are already covered by the frozen
evidence bundle. No performance, promotion, TMA, or matrix-transport claim is
made.

## Files

- `probes/construct.py`: typed-node constructors and mutation helpers.
- `probes/run_audit.py`: the 20-probe driver; prints rows and writes
  `probes/results.json`.
- `probes/results.json`: full expected/actual table with PTX hashes, MMA counts,
  stdout/stderr checks, and plan facts.
- `probes/<id>/`: exact typed IR and metadata for compiled probes, plus
  `out.ptx`, `stderr.txt` where applicable.
- `/tmp/p14_exec.py`, `/tmp/exec_static.py`, `/tmp/p14_min.py`: minimal
  standalone reproductions used to confirm the interior-CTA false Plan and its
  emitted PTX identity. They are scratch evidence; the frozen reproducer is the
  P14 directory and `run_audit.py`.

## Post-audit disposition (repair replay)

This report remains bound to the pre-repair source hash recorded above.  The
checker was then changed, and the same 20-probe script was rerun:

| id | pre-fix | post-fix | disposition |
| --- | --- | --- | --- |
| P09-phase-remapped-k | Plan | NoMatch | Direct operands no longer strip stage terms; hidden modulo K rejected |
| P14-m01-interior-cta-only-copy | Plan | UnsupportedMatch | Non-affine CTA-dependent copy coordinates rejected before endpoint sampling |
| P15-m01-partial-barrier | Plan | UnsupportedMatch | Staging requires `warp=False, count=None` full-CTA barriers |
| P20-staged-a-transpose | NoMatch | Plan | Local shared layout and global copy orientation proved independently |
| P08-nested-k-spelling | NoMatch | NoMatch | Probe is malformed: loads still reference `k` after the loop axis was renamed `k0` |

The three false accepts are closed.  P08 remains fail-closed and its generated
IR cannot be lowered by the SIMT fallback either (`Unknown IR value k`); it is
not a demonstrated valid rejection until the probe is rebuilt with its load
indices renamed consistently.  Permanent typed regressions for hidden K phase,
CTA drift, interior-CTA copy mutation, partial barriers, operand alias, and
valid staged transpose are in `tests/test_tensor_core.py`.  Post-repair source
identity: `tensor_core.py`
`ca5af9c9f41826f9e2000de7a8537745c0393007d8eaaaf9fd212b3bfd3364a6`.
