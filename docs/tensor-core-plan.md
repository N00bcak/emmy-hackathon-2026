# Tensor-core lowering: implemented matmul path and follow-on plan

Status: the initial one-atom-per-warp TF32 matmul path is implemented and
correctness-validated. The correctness baseline is frozen and passing, and the
preserved SIMT fallback is byte-identical. The completed preflight and evidence
are recorded below; TMA, `cp.async.bulk`, `ldmatrix`, `stmatrix`, `wgmma`, and
performance promotion remain out of scope.

Date: 2026-10-23

## 1. Objective and constraints

The frozen `6db2dad4...` baseline compiler emits a correct scalar SIMT
implementation. The current tree adds a strict, typed recognition and lowering
path that:

1. identifies matmul-like dataflow in Kernel IR;
2. synthesizes a warp-level `mma.sync` plan;
3. emits direct PTX for that plan;
4. preserves the existing SIMT path as a correctness fallback; and
5. leaves extension of the same machinery to attention-like kernels for later;
   no attention lowering is implemented in this increment.

The lowering remains ours. We use Emmy's typed deserialization to read the IR,
but we do not call Emmy's renderer, `emmy compile`, nvcc, or another compiler to
produce PTX. `ptxas` is used only to assemble the PTX we emit.

Correctness is the first gate. No tensor-core candidate may be timed for
performance until it passes all 15 published problems at their unchanged
tolerances, plus fresh generated variants and the required memory-safety checks.
Passing the open fixtures alone is not evidence of hidden-set correctness.

### 1.1 Legality of tensor-core instruction selection

It is legal for ordinary Kernel IR operations to lower to tensor-core PTX. The
contract specifies a Kernel IR to PTX translation, an unchanged ABI/launch, and
correct results at the metadata tolerance; it does not prescribe a one-to-one
PTX opcode mapping. `Load`, `multiply`, `Accum(add)`, and `Write` may therefore
become an MMA plan when the compiler proves the equivalent contraction,
coordinate maps, output coverage, synchronization, and numerical contract.

Tensor-core work is optional for the contest, not prohibited:
`emmy-hackathon-2026/TASK.md` lists TMA and tensor-core instructions under
"What is explicitly NOT required." Emmy's own `*` tensor-core IR forms are
documented vocabulary that may be absent from open and hidden problems; support
for those explicit nodes and recognition of ordinary SIMT dataflow are separate
features. Neither path may delegate lowering to Emmy's renderer or another
compiler.

Legality does not make every replacement valid. The selected instruction must
preserve the operation, signed-zero/NaN behavior where the tolerance depends on
it, the output mapping, atomics and `zero_outputs`, collectives, shared-memory
use, and the prescribed launch geometry. For the open f32 matmuls, one-pass TF32
does not meet the unchanged tolerance; the three-term TF32 plan in section 7 is
required. Any plan that cannot prove these properties must fall back to SIMT.

## 2. Baseline state and measured debt

The frozen pre-tensor-core source (SHA-256
`6db2dad4944d87ae46194fd4deaf9331327cda21b21adaa434a1db41f515e6ba`)
reproduced all 15 published cases on the local RTX 4060 (sm_89) through the
unchanged upstream client, including the reordered scale/shift ABI probe. Atomic
matmul also passed an isolated Compute Sanitizer memcheck with zero errors. The
bounded evidence for that hash is in
[experiment 015](../experiments/015-rtx4060-f16-scalar-math/README.md).

The current tensor-core source has SHA-256 values
`996cf5fc...` for `compile.py`, `ca5af9c9...` for `tensor_core.py`, and
`6a9e1af9...` for `tf32_emitter.py`; the full values, corpus hash, PTX/cubin
hashes, logs, and classification rows are in
[the tensor-core evidence bundle](../experiments/tensor-core-v1/README.md). It
passes all 15 published cases through the unchanged client. No tensor-core
performance result has been measured or claimed.

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

The frozen baseline's emitted matmul PTX, before this increment:

- lowers a proven-aligned 16-byte `CpAsyncCopy` to one `cp.async.cg` with a
  runtime alignment check and scalar fallback; other copies retain the typed
  scalar path;
- honors `CpAsyncCommit` / `CpAsyncWait` ordering for the emitted copies;
- has no TMA, `ldmatrix`, or `stmatrix` lowering;
- uses scalar per-element loads and stores for C;
- recomputes address expressions for each element; and
- uses scalar FMA instructions rather than tensor-core instructions.

`m01_matmul_512` has a 32x64 block output tile and 512 threads. That is 16
warps, exactly 2x8 `m16n8` atoms. `m02_matmul_rect` has a 16x32 block output
tile and 256 threads: it has four `m16n8` atoms, not eight. Its 16x32 tile is
exactly covered by a 1x4 warp arrangement using four warps, or by another
explicitly proved decomposition that assigns useful work to every warp. The
prior claim that m02 uses eight warps as 2x4 atoms was arithmetically wrong
(a 2x4 atom grid covers 32x32) and is retained here only as a corrected
preflight finding.

The implemented m01 change is a lane-map and instruction change under its
current launch. m02 remains `UnsupportedMatch`: its four atoms do not match its
eight launched warps, and no idle-warp policy is claimed. A global-load schedule
does not adopt m01's 2x2 per-thread C template; the recognizer derives the
available atom cells and lane map independently.

## 2.1 Implemented increment and evidence map

The current increment provides:

- name-independent recognition from typed `Load`/`multiply`/`Accum(add)`/`Write`
  dataflow, including affine coordinate proofs and dependence/escape checks;
- a validated `MatmulPlan` whose launch factorization, CTA/output coverage, K
  coverage, lane partition, collectives, and non-atomic reduction semantics are
  proved before emission;
- one `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` atom per active warp,
  with three TF32 terms, separate `hh`/`hl`/`lh` C fragments, and deterministic
  `cvt.rna.tf32.f32` conversion;
- 15 `Plan` executions: m01 plus 14 independently generated corpus cases across
  direct-global, shared/`cp.async`, and m01's rotating pipeline staging, with
  several atom-grid factorizations and A/B layouts; and
- byte-identical SIMT fallback for every `NoMatch` and `UnsupportedMatch` case,
  including m02.

The implementation is split between `compile.py`, `tensor_core.py`, and
`tf32_emitter.py`. The evidence bundle records the frozen baseline and corpus at
`experiments/tensor-core-v1/`, and the two adversarial audits at
`experiments/tensor-core-audit/` and `experiments/tensor-core-audit-agent2/`.
The audits are historical pre-repair records; their disclosed false accepts were
fixed and replayed, while the malformed P08 nested-K probe remains fail-closed.

## 3. Design decision: match dataflow, not names

Do not recognize fixtures by entry name, buffer name, accumulator name, or
statement spelling such as `acc0__c0_0`. The hidden set varies names and shapes.
The recognizer must operate on typed Emmy nodes and prove the algebra and
coordinate maps.

The implementation has four layers:

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

The first pass over each `Tile` body:

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

The implemented checker additionally proves that direct operand coordinates
match the output cell, that no output aliases an operand, and that direct
operand K indices are exact affine reduction coordinates (staging-slot terms
are stripped only for shared operands with a proved copy producer). It replays
typed rotating pipelines in statement order and checks every compute read
against the filled and committed source cell. Endpoint-CTA sampling is valid
only after rejecting non-affine CTA-dependent coordinates; all ordering
barriers must be full CTA barriers (`warp=False`, `count=None`).

The independent audits under `experiments/tensor-core-audit/` and
`experiments/tensor-core-audit-agent2/` record the adversarial probes that
found and closed CTA-drift, alias, hidden-K-phase, interior-CTA-copy, and
partial-barrier holes. Those probes are reproduced as permanent regressions in
`tests/test_tensor_core.py`.

The initial fast path is whole-tile replacement. Do not rewrite a small
subtree inside a body that also performs unrelated work. `m01_matmul_512` and
`m02_matmul_rect` are dedicated matmul tiles, so they satisfy this constraint;
m02 still fails the launch-factorization rule below. Attention needs a later
segment-level matcher, described in section 10.

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

### 4.4 Typed-IR corpus gate

The recognizer must not be validated primarily by `m01_matmul_512` and
`m02_matmul_rect`. `tools/research_cases.py` can construct typed `matmul` and
`atomic_matmul` cases, but its `Builder.matmul` uses one direct-load
accumulation template; shape changes alone would let a recognizer overfit to
that template. The corpus must also include transport variants once TMA,
`ldmatrix`, or `stmatrix` are proposed, rather than treating a staged
`cp.async` sample as representative of every shared-memory layout.

Before the recognizer is considered complete, freeze a corpus with at least:

- 12 independently constructed positive typed Kernel IR samples, varying direct
  global loads versus shared staging, operand transpose/layout, M/N/K extents,
  tile decomposition, uniform versus nonuniform reduction splits, tails, and
  complete epilogues;
- 6 near-miss negative samples that must be rejected despite affine-looking
  dataflow, including same-buffer products, non-affine gathers, unrelated
  reductions, unsupported epilogues, partial collectives, and split-K without
  atomics; and
- an explicit expected result (`Plan`, `UnsupportedMatch`, or `NoMatch`) for
  every sample, with an independent plain NumPy float64 reference for positive
  cases and unchanged metadata tolerances.

At least two samples must be written directly against Emmy's typed node classes
rather than through `tools/research_cases.py`. All other samples may reuse
generator machinery only after their dataflow has been reviewed as distinct.
Freeze the corpus bytes and hashes before recognizer work begins so
classification results can be replayed against the same evaluation version. A
frozen corpus proves coverage only of the dataflows it contains; it does not
prove general contraction recognition or hidden-set correctness.

### 4.5 Tile and launch factorization

Derive the CTA output tile from the `Tile.axes`, not from fixture-specific
constants. The initial implementation supports:

- a static 1-D, 2-D, or 3-D launch whose block size is a multiple of 32;
- a CTA output extent factorable into `16x8` atoms;
- no auxiliary producer/consumer threads in the first version;
- no non-empty `serial` axes;
- full CTA participation in every collective; and
- a complete output tile, or an explicitly proved tail-mask plan.

The first version rejects a tile that is not an exact multiple of the atom
shape. That is a deliberate fail-closed choice. Tail support is required before
claiming general matmul coverage on hidden shapes and remains a separate,
testable increment.

For an output tile of `(BM, BN)` with `atoms_m = BM / 16` and
`atoms_n = BN / 8`:

```text
required_warps = atoms_m * atoms_n
required_live_warps <= W = blockDim.x / 32
```

The initial matcher requires `required_warps == W` so every launched warp has
proved work. A tile that needs fewer warps is rejected until the compiler can
mask idle warps without skipping collectives. Multi-atom-per-warp plans are a
separate extension.

For a one-atom-per-warp plan:

```text
warps_n = atoms_n
warps_m = atoms_m
warp_id = threadIdx.x >> 5
warp_m  = warp_id / warps_n
warp_n  = warp_id % warps_n
row0    = cta_m_base + warp_m * 16
col0    = cta_n_base + warp_n * 8
```

Require both divisions to be exact and require `warps_m * warps_n == W`.
Reject anything else rather than silently dropping output cells or leaving
launched warps without a defined role.

For `m01`: `BM=32`, `BN=64`, `W=16`, `warps_n=8`, `warps_m=2`.
For `m02`: `BM=16`, `BN=32`, `required_warps=4`, but `W=8`. The current
open-fixture launch is therefore not eligible for the initial one-atom-per-warp
plan without a proved idle-warp or multi-atom-per-warp policy. Do not claim
`warps_n=4, warps_m=2`; that covers 32x32 outputs and silently loses or
duplicates work when mapped onto a 16x32 tile.

## 5. MatmulPlan

The implemented internal plan object is:

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
    active_warps: int           # must equal blockDim.x / 32 initially
    atom: tuple[int, int, int]  # (16, 8, 8) for TF32
    a_layout: AffineLayout
    b_layout: AffineLayout
    c_layout: AffineLayout
    k_splits: int
    atomic_output: bool
    output_reset: str           # "none" or "host-zero-outputs"
    precision: str              # "tf32x3" initially
    epilogue: EpiloguePlan      # initially only identity/copy
```

The plan must be validated before any PTX is emitted. Validation checks the
number of C cells, the lane partition, the active-warps/launch relationship, the
output coverage, the K coverage, the split-K reset contract, and that each
output cell is written exactly once (or atomically once per split). A plan whose
active warps differ from \`blockDim.x / 32\` is rejected until an idle-warp or
multi-atom-per-warp policy is separately designed, validated, and tested.

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

The implemented first emitter is a small isolated kernel generator:

1. one CTA, one warp, one `16x8x8` atom;
2. scalar global loads using the lane map;
3. TF32 conversion and three-term MMA;
4. scalar f32 stores; and
5. a CPU oracle that validates all 128 output cells.

This stage exists to prove the lane map, register ordering, and rounding. It is
not a performance candidate. It should emit `<40` registers and be simple
enough to inspect in SASS.

### 8.2 Stage 2: integrate the open matmul tiles

The integration replaces a recognized whole tile body with:

```text
prologue: parameter loads, thread/warp ids, block-coordinate decode
  for each k8 step in the CTA's K range:
    load A/B operands for this warp atom
    split each operand into high and low TF32 values
    issue three MMAs into separate C fragments
  combine the three C fragments in order
  epilogue: write the lane's four cells (or atomic-add for split-K)
```

The first version reads the already-staged shared-memory tiles. The current
`CpAsyncCopy` lowering emits `cp.async` and honors the IR's commit/wait groups,
with a runtime alignment check and scalar fallback for proven-wide copies. There
is no claim that the resulting pipeline matches a tensor-core schedule's desired
prefetch depth. That allows the MMA work to be validated independently of a
further copy-pipeline redesign. Do not change copy semantics in the same change
as the lane map.

For `m01`, the CTA K range is 512. For `m02`, the recognized split-K axis gives
each CTA one 8-element K slice; the split is proved from the affine coefficient
of the partition axis and must cover the reduction exactly once across CTAs.
That split-K analysis does not make m02 eligible for the initial emitter: its
16x32 tile has four atoms and its launch has eight warps, so the launch
factorization rule rejects it until a separate warp policy is proved.

### 8.3 Stage 3: copy, epilogue, and TMA efficiency

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

Stage 3 also includes a later, target-gated TMA and matrix-transport track. TMA
is not part of the implemented correctness path; keep it visible in the design
so the copy interface does not foreclose it, but do not emit it until the MMA
path is correct and measured. No copy-interface work beyond the existing
`cp.async` lowering was done in this increment.

#### TMA and matrix transport

- `cp.async.bulk.tensor` (TMA tensor copies) requires `sm_90` or newer, a
  host-built tensor-map descriptor, an mbarrier completion protocol, and a
  proved box/swizzle/OOB policy. The current compiler emits PTX targeting
  `sm_89`, and the open argument ABI carries buffers and scalars, not a 128-byte
  tensor-map descriptor. A generic TMA tensor path therefore needs an explicit
  target-aware mode and a descriptor source; it cannot be inferred from the
  existing `meta.json` alone.
- `cp.async.bulk` without a tensor map can move contiguous blocks between
  global and shared memory and also requires `sm_90` or newer plus completion
  synchronization. It is the lower-risk TMA experiment for contiguous staging
  or epilogue slabs, but it must not be substituted for an arbitrary affine
  copy without proving that the region is contiguous.
- TMA stores are relevant to the epilogue only after C has been staged in
  shared memory. They are descriptor-bound for tensor stores, so the same ABI
  limitation applies. Scalar/vector global stores remain the fallback.
- `ldmatrix` and `stmatrix` are 16-bit matrix-fragment transports; they are not
  TF32 loaders or f32-C epilogue stores. They belong to a later f16/bf16 path.
  Use `ldmatrix` for packed 16-bit A/B fragments and `stmatrix` only where a
  16-bit shared-memory fragment is actually materialized. Do not reuse either
  assumption for TF32.
- `stmatrix` is also `sm_90` or newer. The local RTX 4060 is `sm_89`, so TMA
  and `stmatrix` cannot be validated on the local execution device; `ptxas`
  rejects both for `sm_89`. Assemble them for the hosted target and measure
  there, keeping `sm_89`, `sm_120`, and any other target records separate.

Keep TMA work behind a separate source identity and a target/ABI feasibility
probe. A TMA path that cannot obtain a descriptor, or whose target cannot execute
the instruction, must fall back to `cp.async`/scalar transport rather than
changing the launch contract.

TMA and `ldmatrix`/`stmatrix` are not prerequisites for promoting the initial
MMA path.

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

## 12. Baseline prerequisites (completed)

Phase 0 was completed for the frozen source
`6db2dad4...`: all 15 published cases pass, the reordered-ABI probe passes, and
the published and generated atomic-matmul sanitizer checks report zero errors.
The current compiler also implements scalar-by-value arguments, non-unit 1-D/2-D/
3-D launch flattening, carried state, reduction aliases, `Select`, `Init`, and
typed f16/f64/integer paths as described in
[the baseline report](baseline-correctness.md). Those capabilities do not prove
hidden-set coverage, and they are not a substitute for the tensor-core sample
corpus.

Remaining contract-relevant gaps are recorded in the baseline report:
`bitwise_not`/boolean masks lack a durable variant; `maximum`/`minimum` keep
their documented NaN and signed-zero limit; and the scalar `erf`/`gelu`/
`gelu_tanh` approximations are validated only on the generated variant domain.
They remain general-emitter work, not prerequisites for the implemented
recognizer stage.

The sample corpus and pre-change PTX were frozen before implementation. The
completed recognizer changes PTX selection only for proved plans and shows every
`NoMatch`/`UnsupportedMatch` sample retains byte-identical SIMT output to that
frozen baseline.

## 13. Implementation phases and exit criteria

### Phase 0: freeze the baseline (completed)

- Source `6db2dad4...` passed all 15 published cases, the reordered-ABI probe,
  and the atomic sanitizer checks on the local RTX 4060.
- The tensor-core source was separately re-gated with all 15 published cases
  passing; it did not inherit the earlier source's qualification.
- The frozen source, exact metadata, references, PTX/cubin hashes, commands,
  and experiment `015` links are retained.
- No discovery or sweep campaign was started for this implementation.

Exit met: the baseline gate is reproducible from the frozen source. The current
tensor-core source was separately re-gated with all 15 published cases passing.

### Phase 1: recognizer and plan objects (completed)

- The typed-IR corpus from section 4.4 was frozen with classification
  expectations, independent references, and corpus hashes.
- Typed candidate collection, affine analysis, and plan validation were added.
- Positive and negative unit tests cover the recognition rules and adversarial
  mutations from both audits.
- A synthetic one-atom plan and CPU oracle validate register ordering and lane
  maps independently of the fixture path.
- Non-matmul statements are rejected by the recognizer.
- The open-fixture classification is correct: m01 is eligible for the initial
  one-atom-per-warp plan; m02 is `UnsupportedMatch` under the initial
  `active_warps == blockDim.x / 32` rule until a proved idle-warp or
  multi-atom-per-warp policy exists.

Exit met: the frozen corpus and recognizer tests pass, and the classification
report is retained.

### Phase 2: one-atom emitter (completed)

- The emitter produces `m16n8k8` TF32 PTX with the exact lane map.
- All 128 outputs pass the independent asymmetric CPU oracle.
- The PTX assembles for `sm_89` and `sm_120`; both SASS inspections contain the
  expected TF32 `HMMA` instruction.
- The emitter was validated synthetically before it was connected to published
  fixtures.

Exit met: one-atom numerical, `sm_89`/`sm_120` assembly, and SASS evidence is
retained. Execution evidence exists only for the local `sm_89` device.

### Phase 3: open matmul integration (completed for eligible plans)

- Integration covers m01 and 14 corpus `Plan` cases spanning two staging layouts
  and several atom-grid factorizations; m02 remains outside the `Plan` set.
- The existing SIMT fallback is preserved for all other cases, with frozen PTX
  byte identity checked by the corpus tests.
- All 15 published cases ran at unchanged tolerances and passed.
- Fresh generated cases cover direct loads, shared staging, transposes, tails,
  and complete epilogues; split-K without atomics and atomic split-K remain
  `UnsupportedMatch`.
- No atomic tensor-core path was implemented, so no new reset or sanitizer claim
  is made for MMA. The pre-existing SIMT atomic sanitizer evidence is unchanged.

Exit met: m01 and 14 independently generated corpus `Plan` cases pass; all other
cases retain the frozen SIMT fallback. No tensor-core timing was performed.

### Phase 4: matmul performance (not started; out of scope here)

- Validate and tune the already-emitted asynchronous staging for the recognized
  tile; do not reimplement its basic semantics.
- Improve shared layouts and vectorize the epilogue.
- Evaluate `cp.async.bulk` and TMA tensor copies only after a target/ABI
  feasibility probe proves the instruction set, descriptor source, mbarrier
  protocol, box/swizzle/OOB policy, and shared-memory budget. Keep TMA behind
  its own source identity and fallback; it is not required for an MMA promotion.
- Keep `ldmatrix`/`stmatrix` in the later 16-bit transport track; do not apply
  them to TF32 operands or f32-C epilogues.
- Measure paired baseline/candidate blocks with per-case noise calibration.
- Confirm a selected winner in a fresh isolated run.

Exit: performance evidence is complete, reproducible, and does not hide
regressions in any published case.

### Phase 5: attention (not started)

- Add the segment matcher and staged score/softmax/probability/value plan.
- Start with the open attention shapes, then generate variants.
- Keep the scalar fallback for unsupported fused forms.

Exit: attention passes the full correctness gate and its own isolated
performance and sanitizer checks.

### Phase 6: optional dtype and instruction variants (not started)

- Evaluate f16/bf16 contraction precision only after the f32 path is frozen.
- Evaluate `ldmatrix`, `stmatrix`, and vectorized 16-bit loads only for the
  16-bit path, on a target that supports them. `stmatrix` requires `sm_90` or
  newer and cannot be validated on the current local `sm_89` device.
- Evaluate `cp.async.bulk`/TMA as a separate `sm_90+` transport track; do not
  infer a tensor-map descriptor from `meta.json` or change the launch ABI.
- Do not use `wgmma` for this target: it is Hopper-specific, and the runner is
  not guaranteed to be Hopper. The target metadata and actual GPU determine the
  instruction set.

### Phase 7: close the campaign backlog (not started)

This phase begins only after dense matmul correctness and performance evidence
are stable, and it is not part of the first recognizer stage. It tracks the five
coverage items in `AGENTS.md`: `m01_matmul_512`, `m02_matmul_rect`,
`f02_row_softmax`, `a01_attention`, and `a02_attention_128`.

- Mark an item covered only by frozen source, exact-case correctness, retained
  evidence, and its required isolated confirmation.
- For m02, include the warp-policy decision, atomics reset contract, repeated
  isolated checks, and Compute Sanitizer evidence.
- An item may remain `UnsupportedMatch` with SIMT fallback only when its
  limitations are recorded; do not relabel it covered.

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
| transport | `cp.async`, `cp.async.bulk`, TMA tensor copies, `ldmatrix`, `stmatrix`, per-target assembly |

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

## 16. Open questions outside this increment

1. How does the evaluation package the submission? The contract guarantees only
   the `compile.py` invocation and sibling metadata. Until packaging is known,
   keep the submission path self-contained or make any extra module importable
   from the same delivery unit; do not assume the whole repository is copied.
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
7. Is any launched warp allowed to do no atom work? The initial policy is
   `active_warps == blockDim.x / 32`; an idle-warp policy must prove collective
   participation, output coverage, and synchronization before m02 or similar
   shapes become eligible.
8. What is the complete evaluation unit and tolerance version for each hidden
   matmul? Corpus changes, precision terms, rounding, and accumulation order each
   create a new evaluation version that must be frozen before comparing results.
9. Is there a target/ABI path for TMA tensor copies? TMA needs a host-built
   tensor-map descriptor and `sm_90` or newer; the current `sm_89` PTX target and
   buffer/scalar ABI do not provide either. Resolve this before designing a
   tensor-map path, and prefer `cp.async.bulk` only for proved contiguous
   regions.
10. Which target executes each performance candidate? `ldmatrix`, `stmatrix`,
    TMA, and `cp.async.bulk` have different minimum targets; do not infer
    support from the hosted GPU or reuse an `sm_89` result for an `sm_90+`
    instruction.

## 17. Current status and remaining work

Completed in this increment:

1. The independent typed-IR corpus, hashes, classifications, and float64
   references are frozen.
2. Typed candidate collection, affine analysis, `MatmulPlan` validation, and the
   adversarial regression tests are implemented.
3. The one-atom three-term TF32 emitter passes its 128-cell CPU oracle and the
   `sm_89`/`sm_120` assembly and SASS checks.
4. m01 and 14 corpus `Plan` cases are integrated and pass independent CPU and
   `sm_89` execution references; m02 and every non-`Plan` case retain frozen SIMT
   bytes.
5. The current source passes all 15 published cases at unchanged tolerances.

Not implemented, and not claimed here:

- m02 or any launch where atom count differs from `blockDim.x / 32`;
- atomic split-K MMA, including its reset and isolated sanitizer checks;
- attention segment lowering;
- TMA, `cp.async.bulk`, `ldmatrix`, `stmatrix`, and `wgmma`;
- `sm_120` execution (assembly only, because no such device is available here);
- replay of `cp.async` stages in the rotating-pipeline emitter, which reads the
  proved global cells rather than replaying each staged transport step;
- nested reduction spellings such as the malformed P08 probe, which remains
  fail-closed;
- hidden-set correctness, timing, speedup, or promotion evidence; and
- elimination of the remaining general-emitter gaps listed in the baseline
  report.

The corpus deliberately models only the dataflows it contains. Generalization
comes from the typed proofs, not from fixture names or the published shapes.
