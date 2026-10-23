# Tensor-core lowering plan: matmul first, then attention

Status: historical design plan. Typed recognition and a three-term TF32 emitter
are now incorporated; see the [powenn merge record](powenn-merge.md) for actual
implementation, SIMT preference and validation limits.

Date: 2026-10-22

## 1. Objective and constraints

The current compiler emits a correct but scalar SIMT implementation of every
supported Kernel IR statement. It does not recognize dense contractions and it
does not emit tensor-core instructions. The goal is to add a strict, typed
recognition and lowering path that:

1. identifies matmul-like dataflow in Kernel IR;
2. synthesizes a warp-level `mma.sync` plan;
3. emits direct PTX for that plan;
4. preserves the existing SIMT path as a correctness fallback; and
5. extends the same machinery to attention-like kernels only after the matmul
   path has passed both correctness and measurement gates.

The lowering remains ours. We use Emmy's typed deserialization to read the IR,
but we do not call Emmy's renderer, `emmy compile`, nvcc, or another compiler to
produce PTX. `ptxas` is used only to assemble the PTX we emit.

Correctness is the first gate. No tensor-core candidate may be timed for
performance until it passes all 15 published problems at their unchanged
tolerances, plus fresh generated variants and the required memory-safety checks.
Passing the open fixtures alone is not evidence of hidden-set correctness.

## 2. Current state and measured debt

The current source (SHA-256
`6db2dad4944d87ae46194fd4deaf9331327cda21b21adaa434a1db41f515e6ba`)
reproduced all 15 published cases on the local RTX 4060 (sm_89) through the
unchanged upstream client, including the reordered scale/shift ABI probe. Atomic
matmul also passed an isolated Compute Sanitizer memcheck with zero errors. The
bounded evidence for that hash is in
[experiment 015](../experiments/015-rtx4060-f16-scalar-math/README.md).

Indicative local latencies from the same source, measured by the frozen paired
protocol (16 alternating pairs, medians of the per-pair values):

| case | latency | registers |
| --- | ---: | ---: |
| `m01_matmul_512` | 517.1 us | 40 |
| `m02_matmul_rect` | 10.24 us | 40 |
| `a01_attention` | 236.5 us | 48 |
| `a02_attention_128` | 112.7 us | 48 |

These are 4060 measurements, not 5090 measurements. They establish where the
debt is, not a comparative speedup claim.

The emitted matmul PTX currently:

- lowers a proven-aligned 16-byte `CpAsyncCopy` to one `cp.async.cg` with a
  runtime alignment check and scalar fallback; other copies retain the typed
  scalar path;
- honors `CpAsyncCommit` / `CpAsyncWait` ordering for the emitted copies;
- uses scalar per-element loads and stores for C;
- recomputes address expressions for each element; and
- uses scalar FMA instructions rather than tensor-core instructions.

`m01_matmul_512` has a 32x64 block output tile and 512 threads. That is 16
warps, exactly 2x8 `m16n8` atoms. `m02_matmul_rect` has a 16x32 block output
tile and 256 threads, exactly 2x4 atoms. The current per-thread 2x2 C tile
therefore covers the same 128 outputs per warp as one `m16n8` MMA C fragment.
The first tensor-core change is a lane-map and instruction change, not a launch
geometry change.

## 3. Design decision: match dataflow, not names

Do not recognize fixtures by entry name, buffer name, accumulator name, or
statement spelling such as `acc0__c0_0`. The hidden set varies names and shapes.
The recognizer must operate on typed Emmy nodes and prove the algebra and
coordinate maps.

The implementation will have four layers:

```text
Kernel IR
  -> candidate collection and dependence analysis
  -> strict MatmulPlan validation
  -> TensorCoreEmitter (PTX)
  -> existing SIMT emitter as fallback
```

The recognizer returns one of:

- `NoMatch`: the region is not a supported dense contraction; use SIMT;
- `UnsupportedMatch`: the region looks like a contraction but an invariant is
  not proven; use SIMT and optionally emit a diagnostic to stderr; or
- `Plan`: all invariants hold; lower with the tensor-core emitter.

The compiler must never emit a partially specialized fast path. If a plan is
invalidated during lowering, discard it and use the SIMT path. stdout remains
PTX only.

## 4. Matmul recognition

### 4.1 Cheap prefilter

Start with one linear pass over each `Tile` body:

1. Collect `Accum` statements whose canonical op is `add` or `sum`.
2. Trace each accumulator's `value` through pure `Assign` and `Let` nodes.
3. Look for a `multiply` node with two data-dependent operands.
4. Trace those operands to `Load` statements (possibly through a pure copy or
   cast that is explicitly supported).
5. Require two distinct source buffers for the initial implementation. Same-
   buffer contractions are rejected until aliasing is handled.

This is deliberately cheap: it rejects pointwise, softmax-only, and ordinary
reduction tiles before any affine analysis.

### 4.2 Dependence and escape checks

For each candidate group, require:

- all members fold into the same reduction loop or nested reduction loops;
- the fold op is `add` (or the canonical `sum` alias);
- every product operand is a load from an external buffer, not an accumulator
  or an arbitrary computed expression;
- the accumulator group has no reads other than the final epilogue/write;
- the only writes of the group are the intended output writes;
- no write targets an input buffer;
- the region does not contain a second unrelated contraction or a reduction
  that consumes the same accumulator names in a conflicting way;
- all collectives are at region boundaries or are part of a recognized staging
  pattern; and
- the entire body to be replaced is covered by an explicit allowlist of
  statements: loads, pure assigns, accumulator folds, shared declarations,
  staging copies, barriers, and the final writes.

The initial fast path should be whole-tile replacement. Do not rewrite a small
subtree inside a body that also performs unrelated work. `m01_matmul_512` and
`m02_matmul_rect` are dedicated matmul tiles, so they satisfy this constraint.
Attention will need a later segment-level matcher, described in section 10.

`Accum.base` must be `None` for a plain matmul. A non-`None` base means the
fold reads a rescaled or otherwise redirected left operand; that is an online
attention pattern, not a plain contraction, and is rejected here.

### 4.3 Affine coordinate proof

Represent every load index and output index as an affine expression over the
tile axes, block axes, and reduction axes:

```text
index = constant + sum_j coefficient_j * axis_j
```

The initial matcher accepts only expressions that simplify to this form with
static integer coefficients. It may accept an exact constant division or modulo
when the typed expression prover can show the divisor is exact; otherwise the
candidate is rejected. Arbitrary gathers, data-dependent indices, and untrusted
`TernaryExpr`/`CastExpr` maps are rejected.

For a candidate with output axes `m` and `n`, require the two operand index maps
to have the forms:

```text
A[m_axis, k_axis]       or       A[k_axis, m_axis]       (transposed A)
B[k_axis, n_axis]       or       B[n_axis, k_axis]       (transposed B)
```

The same `k_axis` must appear in both operands. The output write coordinates
must be affine in the same `m` and `n` axes. Strides come from the declared
buffer shapes, not from guessed row lengths.

The reducer must be a sum over one contraction axis (or a proved split of that
axis). A contraction with two independent reduction axes is rejected until a
plan exists for it.

### 4.4 Tile and launch factorization

Derive the CTA output tile from the `Tile.axes`, not from fixture-specific
constants. The initial implementation supports:

- a static 1-D, 2-D, or 3-D launch whose block size is a multiple of 32;
- a CTA output extent factorable into `16x8` atoms;
- no auxiliary producer/consumer threads in the first version;
- no non-empty `serial` axes;
- full CTA participation in every collective; and
- a complete output tile, or an explicitly proved tail-mask plan.

The first version may reject a tile that is not an exact multiple of the atom
shape. That is a deliberate fail-closed choice. Tail support is required before
claiming general matmul coverage on hidden shapes, but it is a separate,
testable increment.

For an output tile of `(BM, BN)` and `W = blockDim.x / 32` warps:

```text
warps_n = BN / 8
warps_m = W / warps_n
warp_id = threadIdx.x >> 5
warp_m  = warp_id / warps_n
warp_n  = warp_id % warps_n
row0    = cta_m_base + warp_m * 16
col0    = cta_n_base + warp_n * 8
```

Require both divisions to be exact and require `warps_m * warps_n == W`. Reject
anything else rather than silently dropping output cells.

For `m01`: `BM=32`, `BN=64`, `W=16`, `warps_n=8`, `warps_m=2`.
For `m02`: `BM=16`, `BN=32`, `W=8`, `warps_n=4`, `warps_m=2`.

## 5. MatmulPlan

Add an internal plan object, for example:

```python
@dataclass(frozen=True)
class MatmulPlan:
    region: int                 # identified Tile/body region
    m_extent: int
    n_extent: int
    k_extent: int
    block_m: int
    block_n: int
    warps_m: int
    warps_n: int
    atom: tuple[int, int, int]  # (16, 8, 8) for TF32
    a_layout: AffineLayout
    b_layout: AffineLayout
    c_layout: AffineLayout
    k_splits: int
    atomic_output: bool
    precision: str              # "tf32x3" initially
    epilogue: EpiloguePlan      # initially only identity/copy
```

The plan must be validated before any PTX is emitted. Validation checks the
number of C cells, the lane partition, the output coverage, the K coverage, and
that each output cell is written exactly once (or atomically once per split).

Keep the recognizer and emitter separate. The recognizer should be testable
without ptxas or CUDA. The emitter should be testable with a synthetic plan and
a one-atom oracle before it is connected to real fixtures.

## 6. MMA atom and lane maps

### 6.1 Preferred instruction for f32 inputs on sm_89

Use:

```ptx
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
```

with four `.b32` A registers, two `.b32` B registers, and four `.f32` C/D
registers. `sm_89` supports this. The same PTX was accepted by `ptxas` for
`sm_120`; the hosted runner may re-target it for its own GPU. Do not encode any
sm_89-only side effects in the PTX.

This is not a replacement for the existing target/assembly tests. Both targets
must continue to assemble, and the final submitted PTX must remain valid for the
runner's chosen architecture.

### 6.2 Lane map for `m16n8k8` TF32

Let:

```text
lane = threadIdx.x & 31
g    = lane >> 2
t    = lane & 3
```

The canonical fragment maps are:

```text
A (row-major, 16 x 8):
  a0 = A[g     ][t]
  a1 = A[g + 8 ][t]
  a2 = A[g     ][t + 4]
  a3 = A[g + 8 ][t + 4]

B (column-major operand, K x N = 8 x 8):
  b0 = B[t     ][g]
  b1 = B[t + 4 ][g]

C (16 x 8):
  c0 = C[g     ][2t]
  c1 = C[g     ][2t + 1]
  c2 = C[g + 8 ][2t]
  c3 = C[g + 8 ][2t + 1]
```

All `A` and `B` values are converted to TF32 before the MMA. The C registers
are f32 accumulators.

This table must be checked against a one-atom numerical oracle before it is
used in a fixture-level transformation. A lane-map error can look plausible on
symmetric test data, so the oracle must use asymmetric values and validate every
output cell, not just a sum or a norm.

### 6.3 Optional f16 atom for later work

For a future f16 fast path, `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
uses packed half pairs. The standard `m16n8k16` fragment maps are:

```text
a0 = { A[g][2t],     A[g][2t+1]     }
a1 = { A[g+8][2t],   A[g+8][2t+1]   }
a2 = { A[g][2t+8],   A[g][2t+9]     }
a3 = { A[g+8][2t+8], A[g+8][2t+9]   }

b0 = { B[2t][g],     B[2t+1][g]     }
b1 = { B[2t+8][g],   B[2t+9][g]     }

C: the same four-cell map as above.
```

`ldmatrix` can feed a 16-bit path, but it is not a TF32 loader; a TF32 path
must use 32-bit shared-memory loads (scalar or vectorized) or another explicit
layout. Do not reuse an f16 `ldmatrix` assumption for the TF32 plan.

## 7. Precision: one-pass TF32 is not correct enough

The open matmul tolerance is `rtol=1e-3, atol=1e-3`. A single TF32 product
does not meet it on the open `m01_matmul_512` data.

Local numerical simulation on the exact open inputs:

| scheme | max abs error | failing elements | worst error / tolerance |
| --- | ---: | ---: | ---: |
| one TF32 pass | 3.25e-2 | 14.8% | 21.4x |
| three-term TF32 | 9.16e-5 | 0 | 0.025x |

The three-term scheme is:

```text
ah = tf32(a)              al = tf32(a - ah)
bh = tf32(b)              bl = tf32(b - bh)

C = ah*bh + ah*bl + al*bh
```

Use three separate f32 C accumulator sets so the terms are accumulated
independently across K, then combine them in the order `hh + hl + lh`. This
matches the tested simulation and avoids relying on the MMA instruction's
inter-pass rounding. The `al*bl` term is omitted initially; it passed the open
case with substantial margin, but generated cases must confirm the margin
before this precision policy is frozen.

The TF32 conversion must be deterministic and must match the independent
oracle. `cvt.rna.tf32.f32` assembles for `sm_89`; `cvt.rn.tf32.f32` and
`cvt.rz.tf32.f32` do not (they require `sm_90` or newer). The high/low split
therefore uses the `.rna` conversion, or an explicit bit-exact helper if the
oracle needs a different rounding. Do not mix rounding modes between the
high/low split and the reference.

Precision is part of the plan identity. Changing the number of terms, the
rounding mode, or the accumulation order creates a new evaluation version.

A later bf16-three-term path is possible, but it is not the default:

| scheme | max abs error on open `m01` | failing elements |
| --- | ---: | ---: |
| one bf16 pass | 0.268 | 72.9% |
| three-term bf16 | 5.52e-4 | 0 |

The bf16 margin is materially smaller than TF32's. Do not promote it without
fresh generated cases and a documented tolerance budget.

## 8. PTX emission plan

### 8.1 Stage 1: one atom through global memory

Build the first emitter as a small isolated kernel generator:

1. one CTA, one warp, one `16x8x8` atom;
2. scalar global loads using the lane map;
3. TF32 conversion and three-term MMA;
4. scalar f32 stores; and
5. a CPU oracle that validates all 128 output cells.

This stage exists to prove the lane map, register ordering, and rounding. It is
not a performance candidate. It should emit `<40` registers and be simple
enough to inspect in SASS.

### 8.2 Stage 2: integrate the open matmul tiles

Replace a recognized whole tile body with:

```text
prologue: parameter loads, thread/warp ids, block-coordinate decode
  for each k8 step in the CTA's K range:
    load A/B operands for this warp atom
    split each operand into high and low TF32 values
    issue three MMAs into separate C fragments
  combine the three C fragments in order
  epilogue: write the lane's four cells (or atomic-add for split-K)
```

The first version may read the already-staged shared-memory tiles. The current
`CpAsyncCopy` lowering emits `cp.async` and honors the IR's commit/wait groups,
with a runtime alignment check and scalar fallback for proven-wide copies. There
is no claim that the resulting pipeline matches a tensor-core schedule's desired
prefetch depth. That allows the MMA work to be validated independently of a
further copy-pipeline redesign. Do not change copy semantics in the same change
as the lane map.

For `m01`, the CTA K range is 512. For `m02`, the recognized split-K axis gives
each CTA one 8-element K slice; the split is proved from the affine coefficient
of the partition axis and must cover the reduction exactly once across CTAs.

### 8.3 Stage 3: copy and epilogue efficiency

Only after stages 1-2 pass correctness:

- tune the existing `cp.async` path (4/8/16-byte copies with commit/wait groups)
  for the recognized tile; unsupported swizzles remain an explicit gap;
- choose a bank-conflict-aware shared layout for the recognized tile;
- use vectorized shared loads where the lane map permits them;
- vectorize the two contiguous C stores per row pair with `st.global.v2.f32`;
- hoist affine address arithmetic out of the K loop; and
- consider double buffering or a two-stage pipeline.

For TF32, `ldmatrix` is not available as a 32-bit loader. Do not plan a
performance path around it. The f16/bf16 path can use `ldmatrix` later.

## 9. Split-K and atomics

`m02_matmul_rect` is the required split-K pattern:

- the CTA's reduction coordinate includes a block axis;
- that axis changes K but not the output M/N coordinates;
- the split ranges are disjoint and cover the reduction extent;
- the output write is atomic; and
- the host resets the output according to `zero_outputs` before each launch.

The matcher must prove all of these properties from typed statements and affine
maps. It must not infer split-K merely from the word `atomic` or from a fixture
name.

For the tensor-core path:

- each CTA computes its split's partial C fragments;
- the three precision terms are summed in registers first;
- exactly one `red.global.add.f32` (or the already-supported atomic store) is
  issued per output cell; and
- no CTA writes a cell twice.

Before timing or promotion, split-K requires:

- isolated first correctness;
- the prescribed reset before every launch;
- repeated runs to expose ordering-sensitive errors; and
- Compute Sanitizer memcheck with zero errors.

## 10. Attention after matmul

Attention is not a second matmul fast path. It is a fused pattern with a
row-wise softmax between two contractions. The safe order is:

1. Finish and freeze the dense-matmul recognizer and emitter.
2. Add a segment matcher for a contraction whose C fragment is consumed by a
   row reduction and then by a second contraction.
3. Keep the existing SIMT path for every attention form that is not proved.

### 10.1 Matcher shape

Recognize the following typed dataflow, within one Tile:

```text
S = Q @ K^T                    (or the documented equivalent layout)
row_max, row_sum = stable softmax reductions over S's N axis
P = exp(S - row_max) / row_sum
O = P @ V
```

Requirements for the first attention plan:

- both contractions are affine and use the same 16x8 atom geometry;
- the softmax reduction axis is the N axis of the first contraction;
- the maximum is row-local, not a block-wide arbitrary reduction;
- the exponentiation and normalization use the IR's own scalar operations;
- the output write is a complete affine tile (or has a proved mask);
- all intermediate score and probability tiles fit within the metadata shared
  memory budget; and
- no unrelated statement is interleaved between the two contractions.

The initial implementation should materialize the score tile in shared memory,
perform the existing stable row-softmax scalar lowering on that tile, write the
probability tile back to shared memory, and then run the second MMA. This is
less aggressive than keeping scores entirely in fragments, but it is much easier
to audit and reuses proven scalar softmax semantics.

### 10.2 Attention-specific precision and reduction risks

- Use the three-term TF32 path for both contractions initially.
- Preserve the IR's stable-softmax form. Do not replace a non-stable form with a
  mathematically equivalent but numerically different formula.
- Validate the score matrix, the normalized probabilities, and the final output
  against an independent NumPy float64 oracle, not just the final norm.
- Check rows with all-equal values and rows with a large dynamic range.
- Check the no-mask case first; then add the mask forms that the typed IR
  actually contains. Do not guess a causal-mask convention from names.
- The first contraction's N axis and the second contraction's K axis must be
  proved to be the same logical axis after layout normalization.

`a01_attention` and `a02_attention_128` currently use different launch shapes
and different fused schedules. The matcher must derive each tile from typed
axes and metadata; it must not assume the open shapes.

### 10.3 Attention acceptance

Attention is complete only when:

- all 15 published cases still pass;
- the fresh generated attention cases pass at their fixed tolerances;
- the fused open cases pass through the unchanged upstream client;
- any atomic or inter-warp synchronization path has sanitizer evidence; and
- performance is measured separately from dense matmul, with the same
  baseline/candidate and noise controls.

## 11. Fallbacks and failure policy

The SIMT emitter remains the correctness oracle for unsupported forms. A
recognizer must reject, not approximate, in these cases:

- non-affine or data-dependent indices;
- ambiguous axis roles;
- same-buffer aliasing;
- multiple output writes per cell;
- non-contractile epilogues;
- unsupported dtypes;
- tails without a proved mask;
- partial CTA participation in collectives;
- non-power-of-two warp participation;
- dynamic shapes;
- unknown `Accum` bases or reductions;
- split-K without an atomic output; and
- any tile whose complete replacement cannot be proven.

An `UnsupportedMatch` may print a concise diagnostic to stderr when debugging is
enabled, but diagnostics must never appear on stdout. If the fast path cannot be
completed after matching, discard it and emit the ordinary SIMT path. A silent
partial fast path is forbidden.

## 12. Baseline prerequisites before implementation

The tensor-core work should not be used to hide general emitter gaps. Before or
alongside it, tighten the correctness baseline in separate changes:

- support all non-starred launch geometries, including non-[1,1,1] block/grid
  dimensions where the metadata supplies them;
- support scalar-by-value arguments when metadata marks them as scalars;
- support the documented f16/f64/i32/i64 buffer and arithmetic cases, or fail
  closed with a clear diagnostic rather than misreading them as f32;
- canonicalize reduction aliases such as `sum`/`prod` and the corresponding
  combine spellings;
- handle nested `Accum` initialization and `Loop.seed=False` from typed
  semantics, not by position alone;
- support the non-starred `Select`, `Init`, barrier, and loop end forms; and
- keep the 15-problem gate and generated-case suite green after each change.

These are not optional if the hidden set uses the unstarred vocabulary. They are
also prerequisites for trusting a tensor-core path that consumes the same
metadata and ABI.

## 13. Implementation phases and exit criteria

### Phase 0: freeze the baseline

- Re-run all 15 published cases on the target GPU.
- Re-run the reordered-ABI probe and atomic sanitizer check.
- Record source hash, evaluator hash, compiler commands, GPU identity, driver,
  clocks, and complete outputs.
- Do not start a discovery/sweep campaign merely to implement this plan.

Exit: the baseline gate is reproducible from the current source.

### Phase 1: recognizer and plan objects

- Add typed candidate collection and affine analysis.
- Add positive and negative unit tests for every recognition rule.
- Add a synthetic one-atom plan and a CPU oracle.
- Prove that the recognizer rejects all non-matmul statements.

Exit: no PTX changes yet; recognizer tests pass.

### Phase 2: one-atom emitter

- Emit `m16n8k8` TF32 PTX with the exact lane map.
- Validate all 128 outputs against an independent oracle.
- Assemble for sm_89 and sm_120.
- Inspect SASS for the expected MMA instruction.

Exit: one-atom numerical and assembly evidence is retained.

### Phase 3: open matmul integration

- Lower `m01_matmul_512` and `m02_matmul_rect` through the fast path.
- Preserve the existing SIMT fallback for all other cases.
- Run all 15 published cases; no tolerances may change.
- Run fresh generated matmul variants covering shapes, strides, transposes, and
  tails that the matcher claims to support.
- Run split-K in isolation, then Compute Sanitizer.

Exit: matmul candidates pass the complete gate; no timing is accepted yet.

### Phase 4: matmul performance

- Validate and tune the already-emitted asynchronous staging for the recognized
  tile; do not reimplement its basic semantics.
- Improve shared layouts and vectorize the epilogue.
- Measure paired baseline/candidate blocks with per-case noise calibration.
- Confirm a selected winner in a fresh isolated run.

Exit: performance evidence is complete, reproducible, and does not hide
regressions in any published case.

### Phase 5: attention

- Add the segment matcher and staged score/softmax/probability/value plan.
- Start with the open attention shapes, then generate variants.
- Keep the scalar fallback for unsupported fused forms.

Exit: attention passes the full correctness gate and its own isolated
performance and sanitizer checks.

### Phase 6: optional dtype and instruction variants

- Evaluate f16/bf16 contraction precision only after the f32 path is frozen.
- Evaluate `ldmatrix` and vectorized 16-bit loads only for the 16-bit path.
- Do not use `wgmma` for this target: it is Hopper-specific, and the runner is
  not guaranteed to be Hopper. The target metadata and actual GPU determine the
  instruction set.

## 14. Test matrix

Every implementation phase should exercise:

| dimension | cases |
| --- | --- |
| shapes | exact 16x8 multiples, non-multiples, small 1/7/16/17/31/33, rectangular |
| K | 8, 16, 17, 31, 32, 33, 64, 512 |
| layouts | A row/transposed, B row/transposed, broadcast-free |
| launch | 1-D, 2-D, 3-D grid/block where metadata permits |
| split | 1 CTA, 2 CTA, 32 CTA, nonuniform K tails |
| precision | one-pass and three-pass simulations; exact tolerance comparison |
| output | f32 copy, f32 atomic, later narrow dtypes |
| safety | bounds guards, input integrity, output reset, sanitizer for atomics |

The test oracle must use plain NumPy float64 for the mathematical result and
then apply the published dtype/tolerance contract. Do not use the compiler's own
emitted operations or Emmy's renderer as the oracle.

Negative tests must include:

- an affine-looking reduction that is not a matmul;
- a multiply whose operands share one buffer;
- a contraction with a non-affine gather;
- a contraction with an unsupported epilogue;
- a partial CTA collective;
- a split-K output without atomic semantics; and
- a tile whose output cells are not covered exactly once.

## 15. Measurement and promotion rules

Indicative smoke numbers are not promotion evidence. For each candidate:

1. bind the gate to the exact compiler source hash and evaluator identity;
2. require all 15 published cases to pass at unchanged tolerances;
3. require fresh generated cases and independent references;
4. run correctness in isolation before any timing;
5. time only in an exclusive GPU window, with baseline and candidate paired;
6. calibrate noise per case and size regime;
7. retain complete samples, not just a median;
8. confirm a winner in a fresh isolated run; and
9. record both 4090/5090 evidence separately if either is ever used.

The 4060 numbers in this document are local indicative baseline data. They are
not a 5090 claim and cannot authorize a speedup statement.

## 16. Open questions to resolve during implementation

1. Does the evaluation copy the whole repository or only `compile.py`? This
   decides whether the tensor-core code can live in a separate module or must
   remain inside the single submission file.
2. Which hidden matmul dtypes occur? The open matmuls are f32; f16/bf16 support
   is a separate precision decision.
3. Does any hidden schedule use a non-1-D grid/block or a non-empty serial axis?
   Those must be supported by the general emitter before a fast path can claim
   coverage.
4. Are hidden contractions always dense, or do they include same-buffer,
   broadcast, or gather forms? The initial recognizer rejects them deliberately.
5. Does any hidden attention schedule already materialize probabilities, or is
   the fused online form the only representation? The first attention matcher
   should handle the materialized form and fall back otherwise.
6. What shared-memory budget does each hidden schedule provide? The plan must
   never exceed metadata `smem_bytes`; an internal layout change must be checked
   against the launch contract.

## 17. Immediate next actions after baseline tightening

1. Implement the typed candidate collector and affine expression helper.
2. Add `MatmulPlan` validation plus recognizer tests.
3. Implement the one-warp, one-atom TF32 emitter and CPU oracle.
4. Wire it to `m01_matmul_512` only; keep all other cases on SIMT.
5. Add `m02_matmul_rect` split-K only after the m01 path is green.
6. Run the full correctness gate and sanitizer before changing copy semantics.
7. Revisit attention only after the matmul path and its performance evidence are
   stable.

Do not combine the recognizer, lane-map, precision, copy-pipeline, and attention
changes into one patch. Each increment needs its own source identity, oracle,
regression result, and retained evidence.
