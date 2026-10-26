# Scaled performance suite

Status: implemented; local evidence recorded. The suite contains 22 frozen
typed-IR cases in two tiers, a guarded full-output correctness path, an
independent NumPy float64 reference, and paired A/A timing against a measured
launch floor. This suite is not in the routine 65-case correctness policy and
must not be used as a replacement for it.

The preflight contract check is yes: the published runner permits at most
`2^26` buffer elements across all arguments of one launch
(`emmy-hackathon-2026/docs/API.md`), so 4M-and-larger inputs are admissible.
The separate `2^22` cap applies to *returned output values*; cases above it are
marked `local_only` and checked with the guarded driver rather than being
treated as hosted correctness passes.

The goal is to measure real kernel performance outside the launch-bound regime
of several published fixtures. The published cases remain the authoritative
correctness contracts; scaled cases add performance evidence for the same
families and mechanisms.

## Published anchors

The published inputs are small enough that many pointwise and reduction
launches are dominated by dispatch and fixed overhead:

| Family | Published anchor | Shape |
| --- | --- | --- |
| Pointwise | `p00_scalar_add` | `N = 128` |
| Pointwise | `p01_vector_add` | `N = 1024` |
| Pointwise | `p02_scale_shift` | `N = 4096` |
| Pointwise | `p03_relu_mul` | `N = 2048` |
| Reduction | `r01_array_sum` | `N = 1024` |
| Reduction | `r02_big_sum` | `N = 65536` |
| Reduction | `r03_mean` | `N = 32768` |
| Reduction | `r04_row_sum` | `64 x 512` |
| Matmul | `m01_matmul_512` | `M = N = K = 512` |
| Matmul | `m02_matmul_rect` | `M = 128, N = 64, K = 256` |
| Fused | `f01_rmsnorm` | `64 x 1024` |
| Fused | `f02_row_softmax` | `64 x 256` |
| Fused | `f03_l2norm` | `64 x 512` |
| Attention | `a01_attention` | sequence `64`, head width `64` |
| Attention | `a02_attention_128` | sequence `128`, head width `32` |

These shapes are correctness and regression anchors. They are not sufficient
performance evidence unless a paired measurement shows a resolved effect.

Published fixtures and the small `A<n>` audit cases are correctness-only. They
must not be used as performance cases, A/A noise-calibration cases, or timing
exit conditions. The sole exception is `p00_scalar_add`, which is measured as
the explicit launch-floor reference so scaled cases can be required to sit a
fixed multiple above measured dispatch noise.

## Scale design

### Frozen tiers

The cases are generated in named tiers so a shape change or a new draw creates
a new evaluation version:

| Tier | Purpose | Frozen target |
| --- | --- | --- |
| `perf-a` | Observable work for ordinary paired timing | pointwise `N = 4,000,000`, except `p02` at `N = 8,000,000`; scalar reductions `N = 4,000,000`; row work `3072 x 4096`; matmul `1024 x 1024 x 512`; norms and softmax `3072 x 2048`; attention sequence `256`, head width `64` |
| `perf-b` | Stress the same mechanisms without changing semantics | pointwise `N = 16,000,000`; scalar reductions `N = 16,000,000`; row work `4096 x 4096`; matmul `2048 x 2048 x 512`; norms `2048 x 4096`; softmax `4096 x 4096`; attention sequence `512`, head width `64` |

These are frozen witnesses, not claims that every current lowering benefits
from every shape. Each case must pass its own resource, correctness, and timing
checks before it is admitted. A tier may be reduced only with a recorded
reason, such as the runner's output-return limit or a compiler resource limit;
the published contract must never be changed to make a shape fit.

The hosted runner currently returns at most `2^22` values per output and allows
at most `2^26` elements across one launch. Cases whose full output exceeds that
return limit are local-GPU only unless a runner with full-output checking is
available. Do not use truncated output as a correctness pass. The manifest's
`local_only` flag is derived mechanically from output size; the runner also
records `hosted_eligible` from the same rule.

### Common scaling rules

1. Start from typed IR and change the actual graph tensor shapes, loop/tile
extents, and metadata argument shapes together. Changing only `meta.json` is
not a scale case.
2. Preserve the family's operation, dtype, ABI order, output membership, smem
contract, and ordered kernel sequence unless a separate schedule variant is
being evaluated. Launch geometry must be derived for the new shape, not copied
blindly from the published case.
3. Preserve the published numerical contracts. Use the same rtol/atol policy for
the family; do not relax a tolerance to make a scaled baseline or candidate
pass. Larger reductions can expose accumulation-order error; that is a case
failure to investigate, not permission to change the contract.
4. Compute references independently in plain NumPy, normally in float64 and
then cast to the declared output dtype. Do not use the compiler's PTX, a
rendered reference kernel, or a prior candidate output as the oracle.
5. Keep deterministic input generation, fixed seeds, and stable case identity.
A changed input, reference, tolerance, or shape is a new evaluation version.
6. Do not hardcode published names, values, or shapes in the compiler. Scale
cases should exercise general typed-IR transformations and should be generated
from a family builder rather than copied from a solution-specific fixture.
7. Keep small tail, alignment, cancellation, and boundary variants separate
from the main scale cases. Scaling must not remove those correctness checks.

## Family plan

### Pointwise (`p00`-`p03`)

Scale `N` while preserving the operation and vector layout: scalar add for
`p00`, vector add for `p01`, scale/shift for `p02`, and relu/multiply for
`p03`. Keep the existing per-thread ownership and vector width; do not turn a
scalar case into a vector case merely to make it look faster. Include an odd
tail variant after the main tier so the scalar tail remains covered.

For `p00` and `p01`, the point is to separate launch cost from real work; a
scalar or vector result that remains unresolved at `4M` should be escalated
to `perf-b` rather than relabeled as a win. For `p02` and `p03`, use the same
scale for a direct comparison and retain the published cases as anchors.

### Reductions (`r01`-`r04`)

For `r01`-`r03`, first scale only the reduction axis and keep the existing CTA
participation and launch shape. This tests longer per-CTA work and exposes
whether the reduction itself is still launch-bound. A split-reduction schedule
is a different case, not a transparent scale of the published IR.

For `r04`, scale rows and columns independently: row count changes the number of
row owners/CTAs, while column count changes per-row work. Preserve the row
mapping, output membership, and any masking. Check the row-cache and wide-row
limits explicitly; do not assume a shape that passes a small test has enough
registers or shared memory at `perf-b`.

### Matmul (`m01`, `m02`)

For `m01`, scale `M` and `N` first with `K` fixed at `512`, because `K` controls
the staged shared-memory pipeline and the published case already uses
`73,728` bytes of shared memory. A `1024 x 1024 x 512` case should test more
output tiles and occupancy without silently changing the staging proof. Any
`K` increase or a `2048 x 2048 x 512` stress case needs a fresh shared-memory
and register check first.

For `m02`, preserve a rectangular problem and the existing ABI/ordering. Scale
`M` and `N` before `K`; `512 x 256 x 256` is the first candidate, with a later
rectangular stress variant only after the staged path is qualified. Do not
introduce an atomic matmul merely to create a larger case unless the atomic
reset and Compute Sanitizer gate is included.

### Fused row work (`f01`-`f03`)

For RMS norm (`f01`) and L2 norm (`f03`), scale rows first to fill the GPU and
then columns to increase per-row work. Update every shape-dependent constant,
including the RMS mean count and any additive stabilizer. Keep the row-cache
reuse and map ownership semantics unchanged; if a wider row exceeds the cache
or register budget, record that as a scaling limit rather than silently
falling back and claiming the same mechanism.

For row softmax (`f02`), scale rows to increase CTA count and columns to increase
per-row work. Test the serial/fused path and any cooperative/online path as
separate case variants; do not compare a candidate using one schedule against a
baseline using another. Use the stable f64 softmax reference for each row.

### Attention (`a01`, `a02`)

Scale sequence length before head width: attention work is quadratic in the
sequence dimension and the published cases are especially launch-bound. For
`a01`, candidates are `256 x 64` and `512 x 64` (sequence by head width). For
`a02`, candidates are `512 x 32` and `1024 x 32`, preserving the Q/Kt/V/output
shapes and ordered kernel structure.

Use a stable float64 softmax reference and retain the exact scaled-softmax
constant, output membership, and any intermediate order. Do not change the
attention schedule or add fusion as part of a scale-only variant. If a larger
sequence requires a different kernel decomposition, record it as a new
mechanism case and re-run both correctness and timing gates.

## Case and runner contract

A materialized case contains:

- a typed Kernel IR file for every kernel in the ordered pipeline;
- `meta.json` with the exact entry, positional `arg_order`, shapes, dtypes,
  grid, block, shared memory, zero-output list, and tolerances;
- deterministic `inputs.npz` and an independently generated `expected.npz`;
- a case record with a stable ID, family, tier, generator/reference identity,
  and hashes of all payload files; and
- a manifest entry binding the case to the frozen evaluator and compiler source
  identities.

Materialize large inputs and binaries in an ignored work or evidence directory;
keep the small generator, manifest, and provenance in the repository. A case
is not identified by a filename alone: its payload hashes, shape, tolerance,
and reference identity must all be checked before use.

Layout:

```text
 tests/performance/
   README.md                 status and operating instructions
   manifest.json            suite version and case identities
   generators/              typed-IR builders and independent references
   runner.py                prepare, correctness, and paired timing driver
   results/                 local receipts, not committed by default
```

## Correctness and timing workflow

1. Build a fresh scale case from typed IR and generate its independent
   reference. Validate the ABI, shapes, launch geometry, and every output.
2. Run the complete published 15-case gate plus the routine regression policy
   for both the exact candidate and baseline sources. A failure or a source
   change invalidates the scale result.
3. Run the scaled correctness checks with full outputs at the fixed family
   tolerances. A hosted result is not a pass if the output was truncated.
4. Establish per-case A/A noise on the same device and protocol before
   interpreting a candidate comparison. Use alternating paired blocks, fixed
   warmup and iterations, GPU-exclusive execution, and a case-specific
   deadband.
5. Compare baseline and candidate on the same frozen inputs and launch
   contract. Report one verdict per case: `faster`, `slower`, or `unresolved`.
   A family claim requires no resolved regression in the measured family.
6. Confirm a resolved win in a fresh isolated run. Keep unresolved cases
   unresolved rather than pooling favorable blocks or changing the deadband
   after seeing the result.
7. Record the host/device, source and evaluator hashes, PTX/cubin identities,
   launch descriptors, warmup/iterations, raw samples, per-block ratios,
   output checks, and failures. Static SASS or PTX counts may explain the
   mechanism, but wall-clock timing decides the performance verdict.
8. Apply the role policy mechanically: every timed performance case must come
   from the scaled manifest; published launches and small `A<n>` fixtures are
   correctness checks only. The launch-floor measurement is calibration, not a
   tested performance case, and cannot satisfy a speed-promotion criterion.

## Current evidence and commands

Evidence is kept under the ignored research root, not committed:

```text
.emmy/research/performance-v1/published-gate-20261027/results.json
.emmy/research/performance-v1/perf-a-final-20261027/results.json
.emmy/research/performance-v1/perf-b-final-20261027/results.json
.emmy/research/performance-v1/cases-v3/
```

The recorded runs used compiler SHA-256
`067ef9c4ce31a4bf5ccda90a8d13e9bd1b71b89aff19abe7598cdba4c77e2921`,
RTX 4060 Laptop GPU (`sm_89`), 9 alternating pairs, 5 warmup launches, and 50
timed iterations per request.

Results:

- The all-15 published gate passed 15/15 with an unchanged compiler hash.
- All 11 `perf-a` and all 11 `perf-b` cases passed full-output correctness;
  `sm_89` and `sm_120` assembly passed. The stress case
  `a01_attention_512x64` reported 12-byte spill stores and 20-byte spill loads
  on `sm_89`, and 16-byte spill stores and 12-byte spill loads on `sm_120`;
  these are retained as resource warnings rather than reported as zero.
- The measured published `p00` launch floor was `4.062 us` in `perf-a` and
  `4.074 us` in `perf-b`; the thresholds were `40.624 us` and `40.736 us`
  (`10x` each floor's p90).
- `perf-a` admitted 9/11 cases, with all medians at `47.6x-95.8x` the floor.
  `r04` and `f02` exceeded the fixed `5%` A/A deadband (`14.0%` and `12.1%`)
  and remain unresolved despite being `50.7x` and `60.5x` the floor.
- `perf-b` admitted 8/11 cases, with all medians at `71.4x-371.9x` the floor.
  `p02`, `r04` and `m01` exceeded the fixed `5%` A/A deadband (`14.1%`,
  `14.3%` and `12.7%`) and remain unresolved.

These are local diagnostic receipts, not hosted benchmark results. Unresolved
cases remain in the manifest and evidence rather than being relabelled.

Reproduce the gate and suite with the shared GPU lock held by the runner:

```bash
.venv/bin/python tools/evaluate.py \
  --compiler compile.py --server http://127.0.0.1:48702 \
  --out .emmy/research/performance-v1/published-gate-$(date +%Y%m%d)

.venv/bin/python -m tests.performance.runner \
  --tier perf-a --case-root .emmy/research/performance-v1/cases-v3 \
  --published-gate .emmy/research/performance-v1/published-gate-20261027/results.json \
  --out .emmy/research/performance-v1/perf-a-$(date +%Y%m%d-%H%M%S) \
  --pairs 9 --warmup 5 --timing-iters 50
```

The suite runner acquires `.emmy/research/.locks/gpu.lock` itself, starts the
loopback timing server, and refuses to time anything until every selected case
has passed full-output correctness. Use the same command with `--tier perf-b`
for the stress tier. The case generator is CPU-only and can be tested with:

```bash
.venv/bin/python -m unittest tests.performance.test_suite -v
```

## Admission checklist

- [x] Freeze `perf-a` and `perf-b` case shapes and tolerances in the manifest.
- [x] Add pointwise, reduction, matmul, fused, and attention builders with
  independent NumPy references.
- [x] Prove every generated IR/metadata pair passes deserialization and both
  `sm_89`/`sm_120` assembly; record nonzero spills as resource warnings.
- [x] Verify full-output correctness on the local GPU before timing; all
  selected cases use the guarded driver, including outputs above `2^22`.
- [x] Run the all-15 published correctness gate.
- [x] Measure a launch floor and record per-case A/A samples and deadbands.
- [x] Keep unresolved cases explicit; do not promote them.
- [ ] Run a fresh isolated confirmation before using a case for a speed claim.
- [ ] Re-run the all-15 gate after any compiler or suite source change.

## Non-goals

This directory does not replace the published cases, the audited 65-case
correctness policy, or the evaluator's contract. It does not authorize
hardcoded shapes or values, a lower tolerance, a different ABI, or a speed
claim based on a single smoke latency. It also does not require every family to
produce a win: an unresolved or slower scaled case is valid evidence and must
be retained.
