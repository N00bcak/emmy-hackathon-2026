# Compiler-design optimization plan: access structure and reuse

Status: P0 complete; P1 accepted on fresh paired wall-clock timing; P2 accepted
on scaled f02 with a confirmed hosted win; P3 accepted and integrated locally
on a confirmed scaled-attention win with no resolved regression; P4 SASS-active
on the pointwise primaries but timing-unresolved and unmerged; P5 deferred with
no compiler change after a complete post-P2/P3 final-SASS inventory of all 14
scaled plan shapes, including both f02 tiers. Static counts are diagnostic. The
wall-clock versus SASS issue-count documentation and judging change is CLOSED.

Date: 2026-10-27

Repository baseline before this work: `33ffc0e` (`compile.py` SHA-256
`f5f895b50f9a7642574d7a125c2be33c5a4ea8e978ad78d9936ec0bf1cd9cecf`).
Measured P1 candidate, retained at repository root over `33ffc0e` (`compile.py` SHA-256
`067ef9c4ce31a4bf5ccda90a8d13e9bd1b71b89aff19abe7598cdba4c77e2921`).

## 1. Objective

Improve generated PTX through compiler reasoning that `ptxas` cannot recover on
its own. Keep the existing IR-to-PTX lowering as the correctness fallback, and
fail closed to it whenever a proof is incomplete.

The design premise is:

- generic local CSE, constant folding, dead-code removal and address
  simplification are largely handled by `ptxas`;
- the durable losses are access structure (scalarized contiguous memory
  operations), values recomputed across phases or CTA threads, and loop/nest
  organization; and
- a transformation earns its place only when fresh paired wall-clock timing
  shows that it helps; removing memory traffic, barriers, or live registers is
  the mechanism, not the result.

The scaled performance suite in `tests/performance/manifest.json` provides
the timing witnesses for P1-P5. Its `perf-a` tier contains 4M-element
pointwise and scalar-reduction cases, `3072 x 4096` row sums,
`3072 x 2048` RMSNorm/L2Norm/softmax cases, `1024 x 1024 x 512` matmul and
`256 x 64` attention; `perf-b` is the stress tier at 16M pointwise/reduction,
`4096 x 4096` row sums and softmax, `2048 x 4096` norms,
`2048 x 2048 x 512` matmul and `512 x 64` attention. Both tiers use frozen
typed-IR shapes, independent NumPy float64 references, and the published
family tolerances.

Current local status: all 22 cases pass full-output correctness and exceed the
10x measured-floor threshold. Local guarded-driver A/A receipts admit 9/11
`perf-a` and 8/11 `perf-b` cases under the fixed 5% deadband. The remaining
cases (`r04` and `f02` in `perf-a`; `p02`, `r04` and `m01` in `perf-b`) are
explicitly noise-unresolved in `tests/performance/README.md`.
These are local diagnostics, not hosted benchmark receipts or a
candidate-versus-baseline speed claim.

Use only the cases named in the priority sections below. The published cases
remain correctness anchors but are excluded from scaled timing lists because
their launch times are below the noise floor. The `perf-b` cases are stress
witnesses and do not replace their `perf-a` analogue for a mechanism verdict.
`r01_array_sum_4m` and `r01_array_sum_16m` are excluded from P1-P5
timing because none of those actions targets scalar-reduction work; they remain
correctness evidence. `m01_matmul_1024x1024x512` has an admitted local A/A
receipt but remains diagnostic for mechanism work until the staged pipeline has
its own legality and timing study.

The small `A<n>` audit fixtures and the published cases are correctness-only
for every priority below. Their grids and element counts are launch-bound:
for example, `A86` is `3 x 127` with three CTAs, and `A95` has one element and
one CTA. Do not use an `A<n>` fixture as a speed witness, an A/A calibration
case, or a timing exit condition. A timing claim must name a scaled suite case
and satisfy that case's output-correctness, A/A deadband, launch-floor ratio
and fresh confirmation gates.

The plan therefore targets typed, whole-region analyses. It does not add a
second general-purpose optimizer over PTX text, and it does not replace the
specialized paths that already work. Those paths remain correctness fallbacks
while their proofs are generalized.

## 2. Current implementation state

The compiler already contains substantial region-level optimization:

- versioned expression and address CSE for pure index trees;
- loop-local optimistic value numbering, including carried accumulators;
- loop-invariant address hoisting with a repeated-use threshold;
- affine index flattening and constant div/rem strength reduction;
- vectorized readonly dot and shared-memory loads;
- CTA-level hoisting of reductions that several threads would recompute;
- a specialized fused softmax path;
- an unrolled per-K-group SIMT matmul pipeline with named barriers and
  `cp.async` staging.

The current source also has a focused cross-phase reuse path for row
reductions:

- `merged_vector_reduction` recognizes a thread-strided reduction whose
  partials merge to a CTA total;
- it can retain the reduction's loaded vectors in registers within
  `ROW_CACHE_VALUES = 32`;
- `row_cached_map` reuses those registers in a later map with the same strided
  header;
- the map emits 128-bit stores and can issue its other row loads alongside the
  reduction; and
- a runtime pointer-alignment check falls back to scalar lowering.

`prefetch_scalar_loads` issues eligible Tile-level scalar loads before a
reduction so their latency overlaps the merge. `cta_total_trees` reuses a full
CTA total instead of re-reading shared slot 0. These are local proofs, not yet
instances of a shared analysis.

For matmul, `plan_simt_ptx` stages A in chunk-major order with a row-group
swizzle, partitions K across CTA thread groups, synchronizes each group with a
named barrier, and uses `cp.async.wait_group stages-2`. This is a schedule with
its own correctness and timing obligations, not a generic loop transformation.

The current lowering is not uniform across families:

- the P1 candidate lowers p02/p03's contiguous tuple groups as 128-bit
  transactions, with scalar address fallback;
- f01 and f03 use the row-cache path, now sharing the strided access proof;
- f02 uses its existing fused-softmax path and still re-reads scores for the
  numerator pass;
- a01 and a02 use the existing KEY/hoisting paths and retain their current
  memory behavior;
- m01 uses the staged SIMT plan; m02 uses its existing atomic path.

## 3. Static evidence for the frozen baseline

All 15 published IRs lower successfully. The following measurements use local
`ptxas -arch=sm_120 -O3` and `nvdisasm`. They are static observations, not GPU
correctness, timing, or performance claims.

PTX global-memory operations:

| case | scalar loads | vector loads | scalar stores | vector stores |
| --- | ---: | ---: | ---: | ---: |
| `p02_scale_shift` | 12 | 0 | 4 | 0 |
| `p03_relu_mul` | 8 | 0 | 4 | 0 |
| `f01_rmsnorm` | 45 | 4 | 0 | 2 |
| `f02_row_softmax` | 42 | 0 | 1 | 0 |
| `f03_l2norm` | 42 | 1 | 0 | 1 |
| `r04_row_sum` | 41 | 1 | 1 | 0 |
| `a02_attention_128` | 132 | 1 | 1 | 0 |
| `m01_matmul_512` | 0 | 0 | 4 | 1 |

Static SASS size and memory instructions:

| case | instructions | LDG | STG | LDS |
| --- | ---: | ---: | ---: | ---: |
| `p02_scale_shift` | 48 | 6 | 4 | 0 |
| `p03_relu_mul` | 64 | 8 | 4 | 0 |
| `f01_rmsnorm` | 528 | 55 | 3 | 5 |
| `f02_row_softmax` | 1160 | 48 | 1 | 10 |
| `f03_l2norm` | 472 | 49 | 2 | 5 |
| `r04_row_sum` | 352 | 48 | 1 | 5 |
| `a01_attention` | 3720 | 77 | 8 | 288 |
| `a02_attention_128` | 1456 | 257 | 1 | 36 |
| `m01_matmul_512` | 2688 | 178 | 5 | 245 |

Registers after `ptxas`: f01 40, f02 40, f03 40, m01 98, a01 146 and a02 164.
No published case spills.

The m01 instruction and memory counts describe a staged pipeline with more
copy and synchronization instructions than a simple loop. They do not establish
whether the overlap wins; only paired GPU timing can decide that.

## 4. Baseline defect repaired first

At the start of this work, the CPU suite ran 32 tests with 1 failure and 6
errors. All 15 published IRs still lowered and assembled for sm_89 and sm_120,
so this was a test-fixture mismatch rather than a demonstrated lowering defect.
It blocked a green baseline and was repaired before P1 work:

1. `test_vector_copy_has_runtime_fallback_and_scalar_negative` now builds the
   complete emitter state, including `unary_cache`, through one shared
   `emitter_fixture()` constructor.
2. `test_m01_inner_loop_hoists_invariant_index_division` now checks the selected
   `plan_simt_route` invariants; `test_generic_loop_hoists_invariant_index_division`
   retains independent generic-loop coverage.
3. Focused tests cover `prefetch_scalar_loads`, `cta_total_trees`, populated row
   caches, and shared strided-access legality, including fail-closed cases.

Current CPU result: 44 compiler tests pass. The 65-case policy passes 65/65
with 16/16 sanitizer checks. The frozen candidate is unchanged throughout these
gates.

Fresh hosted RTX 5090 pairs were attempted for f01, f03, r04, m01 and pointwise
controls (`.../hosted-5090/p0-*`). Those published launches run at the 3-11 us
floor, so the table is retained only as a launch-bound rejection record. It is
not a performance case list and cannot support a timing verdict:

| case | ratio | A/A median (range) | verdict |
| --- | ---: | ---: | --- |
| `f01_rmsnorm` | 1.0000 | 1.0000 (0.9851-1.0303) | unresolved |
| `f03_l2norm` | 1.0106 | 1.0000 (0.9794-1.0159) | unresolved (0.03 us) |
| `r04_row_sum` | 1.0000 | 0.9944 (0.8918-1.0412) | unresolved; 9-block confirmation 1.0000 |
| `m01_matmul_512` | 1.0000 | 1.0000 (0.9913-1.0058) | unresolved; 9-block confirmation 1.0000 |
| `p00_scalar_add` | 1.0105 | 0.9366 (0.8824-1.0471) | unresolved |
| `p01_vector_add` | 1.0169 | 1.0460 (0.9944-1.1477) | unresolved |
| `p02_scale_shift` | A/A | 0.9375 (0.8654-1.1667) | unresolved |
| `p03_relu_mul` | A/A | 1.0112 (0.8627-1.0920) | unresolved |

Identical-code A/A controls can cross the harness's fixed 1% label by several
percent on these launch-bound cases. Those labels are therefore not a valid
noise calibration there, and neither arm is promoted or rejected from them.
The published fixtures stay correctness and regression anchors; scaled cases
in `tests/performance/manifest.json` are the only timing witnesses.

Larger independently referenced witnesses separate mechanism from launch floor:
f01 and f03 at 1024x4096 remain neutral. Their candidate ratios were 1.0002 and
0.9991 primary, and 1.0049 and 1.0059 on fresh nine-block confirmation, against
A/A 0.9997 and 1.0085 (`.../hosted-5090/p0-f0{1,3}-large-4m*`). The two p02
variants also remain unresolved. Thus P0's correctness and infrastructure
acceptance is complete, but its row-cache, prefetch and CTA-total mechanisms
are not credited with a measured speed gain. No P0 speed promotion is claimed.

## 5. Priority P1: general contiguous access vectorization

This is the largest concrete gap. `Load` and `Write` lower their tuple
elements independently in the general path. A body that the IR already exposes
as four consecutive elements therefore reaches SASS as four scalar
transactions, even when the addresses are affine, contiguous and aligned.

### 5.1 Scope

Build one access-plan analysis shared by `Load`, `Write` and the row-map path.
The implemented first version covers f32 four-element groups in:

- straight-line Tile bodies, including p02 and p03;
- map loops whose accesses are contiguous and whose element ownership is
  already established by a preceding merged reduction; and
- epilogues with a contiguous tuple store and scalar producers.

A plain thread-strided map with no preceding merged reduction remains a
documented limit. It stays scalar until a separate ownership proof establishes
that a thread may take a four-element group rather than the reduction's
thread-strided element mapping.

Do not begin with cross-lane packing, reductions, or dtype conversions. Those
change numerical or ownership semantics and belong behind separate proofs.

### 5.2 Proof obligations

A candidate group is legal only when all of the following hold:

- every element is in the same buffer and has a static affine flat index;
- consecutive elements differ by one, in source order;
- the first element is 16-byte aligned when the base pointer is, or a runtime
  alignment check selects a scalar fallback;
- no element index depends on a name defined by the group;
- destination bindings do not shadow a live index operand; and
- every element is read-only, or written exactly once with no intervening
  possibly-aliasing write.

Use the existing `written_buffers` summary for the conservative proof. Extend it
to an alias-aware effect check before allowing any motion across a statement.
Do not vectorize a load whose lanes are only partially consumed without a
register-cost estimate.

### 5.3 Lowering

Represent the result as a typed access descriptor containing:

- buffer, space and element dtype;
- affine index and element offset;
- access width and guard;
- alignment fact and runtime fallback predicate; and
- per-element read/write counts.

Lower a proven group to `ld.global.v4.f32`, `ld.shared.v4.f32`,
`st.global.v4.f32` or `st.shared.v4.f32`, with the scalar path retained for a
misaligned base. Reuse the existing runtime-check pattern from the cp.async
path. Keep vector loads at their original statement position unless the
alias-aware effect check proves earlier issue is safe.

### 5.4 Exit criteria

- Generated contiguous, non-contiguous, unaligned and reordered-ABI pointwise
  variants preserve outputs exactly where required.
- All 15 published cases pass at their exact metadata tolerances.
- Primary timing witness: `p03_relu_mul_4m` exercises four-element loads and a
  straight-line vector store. `p01_vector_add_4m` is a width-four ABI control;
  `p02_scale_shift_8m` is a single-input control. `p00_scalar_add_4m` is
  width-two, outside the first P1 four-element group, so it is a transport
  control rather than a primary. Do not count the 128-element published `p00`,
  which is the measured launch-floor witness. Use the 16M `perf-b` variants
  only after `perf-a` admits them.
- Map witnesses: `r04_row_sum_3072x4096` exercises affine row addressing and
  `f01_rmsnorm_3072x2048` and `f03_l2norm_3072x2048` exercise the row-cache
  map. These are follow-on P1 evidence only where the row-map planner is in
  scope; they do not replace the pointwise primary witnesses.
- A fresh, noise-calibrated paired timing run resolves at least one primary
  pointwise case as faster, with no resolved regression in the measured
  pointwise or admitted map cases. A separate isolated run confirms the win.
- SASS instruction and issue counts are recorded to explain the result, but they
  neither authorize nor veto promotion. Wall-clock time is the ultimate arbiter.

P1 evidence is bound to candidate `067ef9c4...` and baseline `f5f895b5...`:

- all 15 published contracts plus reordered ABI passed on RTX 4060/sm_89
  (`.emmy/evidence/p1-vector-access/final-4060-gate/results.json`);
- the audited 65-case policy passed 65/65 with 16/16 sanitizer checks
  (`.emmy/evidence/p1-vector-access/final-core65-gate/results.json`);
- five generated access variants passed with five sanitizer checks
  (`.emmy/evidence/p1-vector-access/final-access-variant-gate-v6/results.json`);
- isolated width-1 m01 passed three repetitions and Compute Sanitizer
  (`.emmy/evidence/p1-vector-access/final-m01-isolated/`); and
- fresh 4M-element p02 and p03 witnesses with independent NumPy references
  passed on RTX 4060/sm_89 (`.emmy/evidence/p1-vector-access/final-large-local-gate-v3/results.json`).

Hosted RTX 5090/sm_120 paired timing of the 2M-element `p03` variant measured a
0.8805 candidate/baseline median over seven fresh blocks, 10 warmup and 50 timed
iterations (`.../hosted-5090/final-p03-large-2m/result.json`). The same-case A/A
control was 1.0021 (`.../final-aa-p03-large-2m/result.json`). A separate fresh
nine-block confirmation measured 0.8768 (range 0.8745-0.9792;
`.../final-p03-large-2m-confirm/result.json`). The 2M p02 variant was unresolved
at 1.0000 (`.../final-p02-large-2m/result.json`), not a resolved regression.
A fresh 4M-element p03 witness strengthened that result: seven blocks measured a
0.8166 median (range 0.8136-0.8251), same-case A/A was 1.0000, and a separate
nine-block confirmation measured 0.8190 (range 0.8136-0.8648;
`.../final-p03-large-4m/`, `.../final-aa-p03-large-4m/`, and
`.../final-p03-large-4m-confirm/`). The 4M p02 witness was again unresolved at
0.9982, not a resolved regression (`.../final-p02-large-4m/`). Published p02 and
p03 remain correctness and regression anchors because their launch times are
below the measurement floor.

Mechanism diagnostic, not the verdict: on sm_120 the candidate's aligned fast
path issues two 128-bit LDG instead of eight scalar LDG; the misalignment
fallback remains in the binary. Its contiguous store likewise has one vector
fast path plus four scalar fallback stores. The candidate drops from 63 to 43
instructions and from 24 to 20 registers, with no spills (`.../final-sass/`).

Scope limitation: a plain thread-strided map with no preceding merged reduction
still takes the scalar map fallback. Vectorizing that shape would change the
thread-to-element ownership that the row-map path relies on; it is deliberately
left for a separate ownership proof rather than claimed as covered here.

These are local-harness results, not evaluator-issued receipts. The published
launch-bound cases remain correctness and regression anchors rather than speed
evidence.

## 6. Priority P2: shared value-reuse analysis

The row cache proves one textual pattern. Replace the family-specific checks
with an analysis that answers four questions for every pure value:

- does it depend only on literals and CTA-uniform names?
- is it recomputed by multiple threads from identical inputs?
- does every load read a buffer the kernel never writes?
- if materialized, is the register or shared-memory cost lower than
  recomputation?

The analysis should provide a summary that `plan_hoists`, `row_map_plan`,
`prefetch_scalar_loads` and the CTA-total rule can share. Keep each existing
path until the general analysis has equivalent tests and evidence.

### 6.1 Phase reuse

For reductions followed by a map, the desired invariant is that a row element
is loaded once and stays live only as long as the register budget allows. The
budget must include the reduction's accumulators, the map's other live values,
and the address registers. A cache that raises register use enough to reduce
occupancy or spill is not a win.

### 6.2 Second-pass softmax

The softmax maximum and denominator are already available before the numerator
pass. The remaining candidate is to reuse score values across the two passes
when the per-thread value budget permits. Preserve the existing maximum/sum
regrouping and do not force a serial summation order.

### 6.3 Attention

The attention path already hoists some redundant dot products. Generalize the
admission rule to any pure reduction whose free names are Tile axes plus one
serial axis, independent of statement names or the exact attention spelling.
Materialize only values that multiple threads would otherwise recompute, and
compare the shared-memory round trip against the recomputation cost.

### 6.4 Exit criteria

- f01 and f03 retain their current row-reuse behavior through the general
  analysis.
- `A79` proves that independent normalizers stay separate, and `A83` permits
  reuse only for the duplicate graph.
- All 15 published cases pass at their exact metadata tolerances, with fresh
  isolated confirmation before timing.
- Primary row-reuse witnesses: `f01_rmsnorm_3072x2048` and
  `f03_l2norm_3072x2048` exercise cross-phase row reuse at 6M elements;
  `f02_row_softmax_3072x2048` exercises the two-pass softmax reuse candidate.
  `r04_row_sum_3072x4096` is the reduction-only control for the row path.
  The 16M `perf-b` rows are stress witnesses, not substitutes for first
  admission.
- Attention reuse witnesses: `a01_attention_256x64` is the smallest scaled
  attention case and remains secondary to the row-reuse primaries; use it only
  when measuring the CTA-uniform hoisting sub-mechanism. Do not use the
  published 64x64 `a01`, whose launch dominates.
- A fresh, noise-calibrated paired timing run resolves one row-reuse primary
  (`f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048` or
  `f02_row_softmax_3072x2048`) as faster, with no resolved regression in the
  measured reuse family. A separate isolated run confirms the win. `A86` and
  the `A57`, `A79` and `A83` attention guards are correctness-only; they cannot
  satisfy this timing exit.
- Load, barrier, issue and register counts are recorded as mechanism evidence.
  They do not decide acceptance; wall-clock time is the ultimate arbiter.

## 7. Priority P3: sibling-loop fusion

Adjacent loops over the same iteration space can share loads without changing
any accumulator's fold order. This is fusion, not reassociation: each fold keeps
its original operand sequence.

Target a map loop following a reduction over the same row, and several sibling
dots sharing one operand. The existing KEY batching is a narrow form of the
same idea.

Admission requires:

- statically identical start, step and bound;
- no cross-iteration carried state except independent accumulators;
- no write that could alias a fused read;
- preserved per-element order for every accumulator; and
- live-register growth within the measured budget.

Do not fuse until P1 and P2 are in place. Without vectorized accesses and a
reuse budget, fusion often removes only a load that the hardware already had in
flight.

### 7.1 Expected test cases

Case labels below use `A<n>` for row `n` in
`docs/test-relevance-audit/cases.json`; that audit index is not a position in
the routine 65-case array. Repeated witness names make the audit index the
unambiguous reference.

- Primary scaled gain targets: `a01_attention_256x64` and
  `a01_attention_512x64`. These contain the repeated QK/PV dot and map
  structure that the existing KEY batching only partly captures. Do not time
  the published 64x64 `a01` or 128x32 `a02` as P3 evidence; both are launch
  anchors and the suite has no `a02` scale case.
- Correctness generalization cases: `A57`
  (`nested-key-score-shadow`, width 16) and `A84` (the width-8 variant) test
  child score scopes; `A71` (`shared-dot-seed-and-tail`) and `A72`
  (`shared-dot-misaligned-row-stride`) test fusion order and access fallback;
  `A85` (`nested-seeded-fold-scopes`) tests accumulator ownership across nested
  seeded folds. These fixtures have at most 280 output elements and are not P3
  timing witnesses.
- Negative guards: `A62` (`attention-0`, `ordered-three-kernel-attention`)
  requires three ordered kernels and intermediate buffers, so fusion must not
  cross that boundary. `A79` (`independent-query-normalizers`) must keep
  distinct queries distinct; `A83` (`duplicate-vs-distinct-normalization`) may
  share only the structurally equivalent graph. `A75`, `A76` and `A80`-`A82`
  cover the attention full, partial-fallback and tail geometries that fusion
  must not collapse.
- Generated variants must include sibling dots with equal and unequal bounds,
  distinct operand identities, an aliasing write between candidate loops, and
  independent carried accumulators. Existing published cases do not cover every
  rejection condition.
- `f01_rmsnorm`, `f03_l2norm` and `r04_row_sum` are not P3 gain targets. They
  are P2 row-cache clients and remain regression guards against changed
  ownership or accumulator order.

### 7.2 Exit criteria

- Generated sibling variants and `A71`, `A72` and `A85` preserve each
  accumulator's operand order and output ownership. `A79` does not share
  distinct normalizers, and `A83` shares only equivalent normalization values.
- All 15 published contracts pass at their exact metadata tolerances, and the
  complete routine 65-case policy passes, including `A57`, `A62`,
  `A71`-`A72`, `A75`-`A76`, `A79`-`A85` and the three row-cache clients above.
- A fresh noise-calibrated paired timing run on the target device resolves
  `a01_attention_256x64` or `a01_attention_512x64` as faster, with no resolved
  regression in either scaled attention case. A separate isolated run confirms
  the win. The published attention cases remain correctness anchors only.
- Final-SASS load, barrier and issue counts, spills and occupancy are recorded
  as mechanism evidence. They neither substitute for a timing win nor reject a
  measured win by themselves; wall-clock time is the ultimate arbiter.
- If resolved timing shows no win, record the rejection with case names, SASS
  counts and paired timings, and retain the current KEY batching. If timing is
  unresolved, record it as unresolved rather than rejected. Do not merge fusion
  on structural grounds alone.

## 8. Priority P4: address strength reduction

The general address path emits an index, a byte offset and a 64-bit base
addition per access. `ptxas` folds much of this in straight-line code, so this
ranks below P1-P3. Apply it only to a loop that already has a proven access
plan.

Prefer a base pointer per buffer plus immediate offsets for a proven-affine
unrolled group. Keep the runtime alignment fallback and the existing signed32
overflow bounds. Record register count and SASS issue count as diagnostics.
Wall-clock latency is the ultimate arbiter: neither a smaller PTX file nor a
lower static count is a performance verdict, and neither can reject a measured win.

### 8.1 Expected test cases

- Primary gain targets: `p02_scale_shift_8m` and `p03_relu_mul_4m` exercise the
  straight-line address chain with the least unrelated control flow. Use the
  16M `perf-b` variants only after `perf-a` admission. The map loops in
  `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048` and
  `r04_row_sum_3072x4096` exercise affine row addressing once P1 and P2 supply
  a proven access plan; they are follow-on, not first-P4 acceptance cases.
- Tail and index guards (correctness-only): `A90`
  (`scalar-pointwise-odd-tail`) and `A91`
  (`width-two-pointwise-odd-tail`) require a scalar tail after a proven group;
  `A24` (`signed-index-arithmetic`) requires exact signed division and remainder
  in addresses; `A25` (`vector-alignment-and-tail`) requires the alignment and
  partial-group fallback. All are limited to at most 4,099 outputs and must not
  be used as P4 timing references.
- Scope and snapshot guards: `A30` (`nested-tile-binding-restoration`), `A31`
  (`loop-axis-shadow-scope`), `A50` (`load-name-shadow`) and `A51`
  (`grouped-load-snapshot`) catch a cached base or offset that survives the
  wrong binding. `A58` and `A59` cover the alternate RMSNorm schedule and the
  short L2Norm row.
- Targeted extension: `A95` (`axis-rebound-before-load`) is outside the routine
  65-case policy but must run for P4 because it directly tests a stale loop-axis
  assumption in the vector planner.
- `m01_matmul_1024x1024x512` is a follow-on address-pressure probe, not an
  acceptance case for the first P4. Its larger staged pipeline needs a separate
  legality and timing study; the published 512-cube is below the launch floor.

### 8.2 Exit criteria

- Register count, spilling, occupancy and final-SASS address (`IMAD`, `LEA`,
  `SHF`, address-only `LOP3`) and memory counts are recorded for the affected
  loop body. A change in these counts is a diagnostic explanation, not an
  acceptance condition. SASS issue count is likewise diagnostic; wall-clock
  latency is the ultimate arbiter.
- All 15 published contracts pass at their exact metadata tolerances, the
  routine 65-case policy passes, and the named P4 guards `A24`, `A25`, `A30`,
  `A31`, `A50`, `A51`, `A58`, `A59`, `A90` and `A91` pass. Targeted `A95` also
  passes.
- A fresh noise-calibrated paired timing run resolves at least one primary case
  (`p02_scale_shift_8m` or `p03_relu_mul_4m`) as faster, with no resolved
  regression in either primary. Admitted map cases may strengthen the result
  but cannot substitute for a primary pointwise win. A separate isolated run
  confirms the result.
- If resolved timing does not improve, retain the general address lowering and
  record the rejection; if timing remains unresolved, record that result
  instead. Do not promote on PTX size, a lower SASS issue count, or any other
  static instruction count, and do not reject a repeatable timing win merely
  because a static count rose.

## 9. Priority P5: generic CSE, LICM and DCE

Keep the existing scoped CSE, GVN and invariant hoisting. Do not add a
whole-function SSA optimizer as a standalone project. Current SASS shows that
redundant PTX arithmetic is largely removed downstream, while the valuable
misses are gated on memory-layout or inter-thread proofs.

If a general pass is added, operate on a typed IR summary rather than PTX text.
Its acceptance test is a fresh paired wall-clock win; SASS counts are retained
only to explain which work disappeared.

### 9.1 Expected test cases

- Scaled opportunity probes: `p02_scale_shift_8m` and `p03_relu_mul_4m` can expose
  duplicate constants and dead prologue work; `f01_rmsnorm_3072x2048`,
  `f03_l2norm_3072x2048` and `r04_row_sum_3072x4096` can expose loop-invariant
  index work; `m01_matmul_1024x1024x512` is a screening case for invariant
  staged-plan arithmetic. Remove the published small pointwise and reduction
  cases and all `A<n>` fixtures from P5 timing; they are below the noise floor
  and serve as correctness anchors. Any probe may show that `ptxas` already
  removed the candidate, which is a valid negative result.
- CSE cache-key sentinels: `A26`-`A29` (`cast`, `FuncCall`, `FlatIndex` and
  `Builtin` child keys) require distinct children to remain distinct.
- Scope and lifetime sentinels: `A30`-`A32` and `A40` cover nested Tile/loop
  bindings and Init shadowing; `A50`-`A52` and `A55` cover load-name shadows,
  grouped-load snapshots, carried updates and select-local bindings.
- Attention value-numbering sentinels: `A57` and `A84` cover width-16 and
  width-8 KEY child score scopes; `A79` must keep independent normalizers
  separate; `A83` permits reuse of the duplicate normalization and must still
  distinguish the non-equivalent query graph; `A85` covers nested seeded fold
  ownership.
- DCE and visibility: `A68` (`observable-padding`) forbids deleting a lookahead
  copy or shared-memory write whose bytes are read later even when the
  arithmetic iteration appears dead.
- Targeted `A95` must run because LICM or address CSE can otherwise hoist an
  address across a loop-axis rebinding.

### 9.2 Exit criteria

- Build a no-op SASS inventory before implementing the pass. It identifies
  candidate work that survives the existing scoped CSE, GVN and `ptxas` and
  helps predict which probe can exhibit a timing change. A probe with no
  surviving static opportunity is still timed when the mechanism predicts a
  dynamic effect that static counts cannot expose.
- Final-SASS instruction, memory, spill and occupancy counts for accepted work
  are recorded as mechanism evidence. They do not determine acceptance.
- All 15 published contracts pass at their exact metadata tolerances, and the
  routine 65-case policy passes. The named CSE and lifetime sentinels
  `A26`-`A32`, `A40`, `A50`-`A52`, `A55`, `A57`, `A68`, `A79`, `A83`-`A85`
  pass; targeted `A95` also passes.
- The pass preserves floating-point operand order, exact index and transport
  semantics, output membership and all load-to-store visibility. It must not
  expose reassociation merely to create a common subexpression.
- A fresh noise-calibrated paired timing run resolves a win on at least one
  scaled probe (`p02_scale_shift_8m`, `p03_relu_mul_4m`,
  `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048` or
  `r04_row_sum_3072x4096`), with no resolved regression in the admitted
  probes. `m01_matmul_1024x1024x512` is a diagnostic screen only. A separate
  isolated run confirms the result. A static reduction without a measured win
  does not qualify, and static counts cannot overturn a measured win.
- If resolved timing shows no win, record P5 as deferred, retain the existing
  scoped CSE, GVN and invariant hoisting, and do not merge a whole-function
  pass. If timing is unresolved, keep the result unresolved rather than
  attributing rejection to static counts.

## 10. Non-goals and hazards

- Do not reassociate floating-point additions or multiply/add sequences to
  expose CSE. Reuse only structurally identical values.
- Do not vectorize across lanes or change output ownership. The repository's
  historical vector-reduction candidate was rejected for changing ownership.
- Do not hard-code fixture names, shapes or constant values. Hidden metadata
  may vary.
- Do not extend live ranges without a register budget. a01 and a02 already use
  146 and 164 registers.
- Do not treat the m01 static instruction increase as either a regression or a
  win. Its named-barrier pipeline needs paired GPU measurement.
- Do not time a candidate before it passes all 15 published contracts at their
  exact tolerances from the frozen source identity.
- Do not replace a working specialized path until the general path has
  equivalent correctness and performance evidence.

## 11. Implementation sequence

### Step 0: restore a green baseline

- Repair the two stale CPU tests and add focused tests for the row cache,
  scalar prefetch and CTA-total reuse.
- Compile all 15 published IRs for sm_89 and sm_120.
- Run the authorized GPU correctness gate on the routine 65-case policy, plus
  required sanitizer checks for atomics.
- Record fresh paired timings for the scaled `f01_rmsnorm_3072x2048`,
  `f03_l2norm_3072x2048`, `r04_row_sum_3072x4096` and
  `m01_matmul_1024x1024x512` cases and their scaled controls. The published
  f01, f03, r04 and m01 fixtures are correctness anchors only.
- When a published launch is below the measurement floor, add a larger typed
  witness and calibrate A/A before classifying its result.

Exit: CPU suite green, both-target assembly green, no unexplained GPU failure,
and m01 has isolated first-run plus reset/sanitizer evidence. All of these hold;
the P0 timing record is neutral and makes no speed-promotion claim.

### Step 1: access-plan builder and P1

- Add the shared access descriptor and legality analysis.
- Lower proven f32 four-element groups with a scalar alignment fallback.
- Time the four-wide `p03_relu_mul_4m` primary, with `p01_vector_add_4m` and
  `p02_scale_shift_8m` as controls, before touching the `r04`, `f01` or `f03`
  map cases. Keep width-two `p00_scalar_add_4m` as a transport control because
  it is not a four-element P1 group.

Exit: the P1 criteria in section 5.4.

### Step 2: generalized reuse and P2

- Factor uniformity, escape and cost into one analysis.
- Re-express the row cache, hoisting and CTA-total rules as clients.
- Extend reuse only where the cost model permits.
- Use `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048` and
  `f02_row_softmax_3072x2048` for the row-reuse verdict, with
  `r04_row_sum_3072x4096` as the control.

Exit: the P2 criteria in section 6.4.

### Step 3: sibling-loop fusion and P3

- Implement only with preserved fold order and an alias-safe effect proof.
- Validate correctness on `a01_attention`, `a02_attention_128`, `A62`,
  `A71`-`A72` and `A75`-`A85`, plus generated sibling-dot variants. These small
  fixtures gate correctness only; do not time them or validate only on a01.
- Time `a01_attention_256x64` first, then `a01_attention_512x64`; the latter is
  the confirmation-scale witness. The published attention cases remain
  launch-bound correctness anchors.

Exit: the P3 criteria in section 7.2.

### Step 4: address strength reduction and P4

- Apply only to access-plan loops.
- Record SASS issue count and register pressure as diagnostics.
- Run correctness guards `A24`-`A25`, `A30`-`A31`, `A50`-`A51`, `A58`-`A59`,
  `A90`-`A91` and targeted `A95` before timing. Do not time these small
  fixtures.
- Time `p02_scale_shift_8m` and `p03_relu_mul_4m` first; treat
  `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048` and
  `r04_row_sum_3072x4096` as admitted-map follow-ons.
- Gate promotion on a fresh paired wall-clock win with no resolved regression;
  static counts explain a result but cannot replace it.

Exit: the P4 criteria in section 8.2.

### Step 5: P5 screening and disposition

- Build a no-op SASS inventory for the P5 probes before changing the compiler.
- Implement only if repeated or dead work survives both existing passes and
  `ptxas`, or another falsifiable mechanism predicts a timing gain; otherwise
  close P5 as deferred.
- Run the cache-key, scope, normalization and visibility sentinels from section
  9.2 as correctness checks before any timing. They are not performance cases.
- Screen with `p02_scale_shift_8m`, `p03_relu_mul_4m`,
  `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048` and
  `r04_row_sum_3072x4096`; keep `m01_matmul_1024x1024x512` diagnostic until its
  staged-plan timing study is complete.

Exit: the P5 criteria in section 9.2. A deferred P5 does not block P3 or P4.

## 12. Test matrix

| mechanism | correctness cases | diagnostic checks | wall-clock verdict |
| --- | --- | --- | --- |
| P1 vector access | published p02, p03, f01/f03 maps, generated contiguous/non-contiguous/unaligned pointwise IRs | both-target assembly, SASS LDG/STG counts, spills | `p03_relu_mul_4m` (primary), `p01_vector_add_4m` (width-four control), `p02_scale_shift_8m` (single-input control), `p00_scalar_add_4m` (width-two transport control); fresh paired wall-clock win and no resolved regression |
| P2 reuse | published f01, f02, f03, a01, a02; correctness-only `A57`, `A79`, `A83`, `A86` plus typed uniformity variants | SASS LDG/LDS/barrier counts, registers | `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048`, `f02_row_softmax_3072x2048` (row primaries), `r04_row_sum_3072x4096` (control), `a01_attention_256x64` (secondary); fresh paired wall-clock win and no resolved regression |
| P3 loop fusion | published a01/a02; correctness-only `A62`, `A71`-`A72`, `A75`-`A76`, `A79`, `A83`-`A85`, `A57`, `A84` plus fresh sibling-dot variants | SASS LDG/LDS counts, operand-order and ownership checks, spills | `a01_attention_256x64` and `a01_attention_512x64`; fresh paired wall-clock win and no resolved regression |
| P4 address strength | published p02, p03, f01, f03, r04; correctness-only `A24`-`A25`, `A30`-`A31`, `A50`-`A51`, `A58`-`A59`, `A90`-`A91`; targeted `A95` | SASS issue and IMAD/LEA/SHF/LOP3 counts, registers, spill and occupancy | `p02_scale_shift_8m`, `p03_relu_mul_4m` (primaries), `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048`, `r04_row_sum_3072x4096` (maps), `m01_matmul_1024x1024x512` (diagnostic); fresh paired wall-clock win and no resolved regression |
| P5 generic CSE/LICM/DCE | published p02, p03, f01, f03, r04, m01; correctness-only `A26`-`A32`, `A40`, `A50`-`A52`, `A55`, `A57`, `A68`, `A79`, `A83`-`A85`; targeted `A95` | no-op SASS inventory and final counts | `p02_scale_shift_8m`, `p03_relu_mul_4m`, `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048`, `r04_row_sum_3072x4096`; `m01_matmul_1024x1024x512` diagnostic; fresh paired wall-clock win, else defer |

Every candidate also runs the routine audited 65-case policy. Correctness-only
screening is not a timing promotion. All timing candidates require the
source-bound all-15 gate and fresh isolated confirmation. Static counts may
prioritize which candidates to time first, but they cannot screen one out of a
speed verdict; only a resolved paired wall-clock result can promote or reject one
on performance grounds.

### 12.1 Judging and promotion workflow

Judge correctness and performance in this order. A failure at any gate stops
promotion, and an unresolved speed result cannot promote; a later gate does not
waive an earlier one.

1. Run the CPU suite and both-target assembly. Record failures as diagnostics,
   not as performance evidence.
2. From frozen source, run all 15 published contracts at exact metadata
   tolerances, then the routine 65-case policy and required sanitizer checks.
   Bind every result to source and evaluator identities; any source change
   invalidates the gate.
3. Identify a workload regime where the mechanism can affect wall-clock time.
   A launch-bound published case is a valid correctness and regression anchor,
   but not sufficient evidence of a speed change. Add a fresh typed-IR case
   with an independent reference and unchanged ABI rules, or use an existing
   larger witness, when the published case is below the measurement floor.
4. Establish per-case A/A noise in that regime. Use the repository's paired,
   alternating, GPU-exclusive protocol with case-specific warmup, iterations,
   deadband and bounded block count. Neither static counts nor one smoke
   latency may select a winner.
5. Compare baseline and candidate over fresh paired blocks. Accept only a
   candidate-faster result outside the calibrated deadband with no resolved
   regression in the measured family. Keep unresolved cases unresolved; do not
   turn them into wins by pooling blocks or changing the deadband afterward.
6. Confirm the selected win in a fresh isolated run. Record host/device,
   buffers, launch geometry, source and PTX identities, SASS counts, registers,
   spills, samples, failures and exact commands alongside the timing receipt.
7. Classify the result by confirmation. A confirmed win promotes; a resolved
   slower or non-win result is rejected or deferred as the priority specifies;
   an incomplete, unresolved or contradictory result remains unresolved. State
   which gate failed, retain the mechanism and measurement artifacts, and do not
   substitute a lower static instruction or SASS issue count, a smaller PTX
   file, or a historical benchmark for the missing wall-clock result.

SASS issue count is a proxy, not a performance criterion. SASS issue counts,
LDG/LDS/STG counts, spill counts and occupancy are retained as mechanism
evidence and falsification aids. They may guide which candidates are worth
timing, but they are neither acceptance criteria nor a veto over a reproducible
measured improvement. Wall-clock latency is the ultimate arbiter.

Mechanism activation, separately from the performance verdict, is judged on the
measured shape in final post-`ptxas` SASS. A change that exists only in PTX is
not activation: if `ptxas` removes, rebuilds or cancels it, re-implement the
lowering so the intended final-SASS effect survives, or fail closed and record
that the mechanism is inactive. Static SASS counts still cannot award or reject
a performance result; this rule only prevents an inactive mechanism from being
measured or described as if it had fired.

Report exactly one performance verdict per compared case: faster, slower or
unresolved. Only a complete paired result outside the case-specific A/A deadband
can yield faster or slower, and confirmation must preserve that verdict. If the
timing is incomplete, statistically unresolved, or contradicted by fresh
confirmation, record `unresolved`. Do not relabel it as rejected on the basis of
SASS counts, nor promote it on mechanism plausibility alone.

### 12.2 Case expectations at a glance

`A<n>` is the row index in `docs/test-relevance-audit/cases.json`; do not
confuse it with a position in the routine 65-case array, where several names
repeat.

| priority | expected to help | expected to stay unchanged |
| --- | --- | --- |
| P1 | `p03_relu_mul_4m`; generalization through generated contiguous/non-contiguous/unaligned pointwise IRs | `p01_vector_add_4m` and `p02_scale_shift_8m` controls; `p00_scalar_add_4m` width-two transport control; `r04_row_sum_3072x4096`, `f01_rmsnorm_3072x2048` and `f03_l2norm_3072x2048` map follow-ons |
| P2 | `f01_rmsnorm_3072x2048`, `f03_l2norm_3072x2048`, `f02_row_softmax_3072x2048` | `r04_row_sum_3072x4096` control; `a01_attention_256x64` secondary; `A86` correctness-only; `A79`/`A83` reuse boundaries |
| P3 | `a01_attention_256x64`, `a01_attention_512x64` | correctness-only `A57`, `A71`-`A72`, `A84`-`A85` generalization; `A62` (three-kernel boundary), `A79` (distinct normalizers), `A83` (equivalent-only reuse), `f01`/`f03`/`r04` ownership |
| P4 | `p02_scale_shift_8m`, `p03_relu_mul_4m`; map follow-ons `f01`/`f03`/`r04` | correctness-only tails `A90`-`A91`; `A24`-`A25`, `A30`-`A31`, `A50`-`A51`, `A58`-`A59`, targeted `A95`; `m01_matmul_1024x1024x512` diagnostic |
| P5 | `p02_scale_shift_8m`, `p03_relu_mul_4m`; invariant index work in `f01_rmsnorm_3072x2048`/`f03_l2norm_3072x2048`/`r04_row_sum_3072x4096`; `m01_matmul_1024x1024x512` screen | correctness-only cache keys `A26`-`A29`, scopes `A30`-`A32`/`A40`/`A50`-`A52`/`A55`, attention `A57`/`A79`/`A83`-`A85`, visibility `A68` |

“Expected to help” means the case has a plausible structural opportunity; it
does not assert a gain. Each priority still has to demonstrate a fresh paired
wall-clock win, else be recorded as unresolved or rejected/deferred. Static
counts are retained to explain the mechanism and to catch an unexpected resource
or codegen change.

## 13. Evidence boundaries

This plan is based on:

- source inspection of `33ffc0e`;
- successful lowering and both-target assembly of all 15 published IRs;
- local sm_120 assembly and static SASS counts, used only as non-authoritative
  mechanism proxies; wall-clock timing remains the performance arbiter; and
- existing repository records covering the gate policy and the rejected
  ownership-changing vector reduction.

P0 and P1 have frozen-source correctness evidence. P0's hosted timing matrix is
complete but neutral: the published cases are launch-bound and the larger f01,
f03 and p02 witnesses are unresolved. P0 claims no measured speed gain. The P1
candidate source is `067ef9c4...`; any compiler edit creates a new identity and
invalidates those gates. Hosted RTX 5090 results resolve a P1 p03-family win at
0.8805 primary and 0.8768 confirmation against A/A 1.0021; the 4M witness
resolves at 0.8166 with 0.8190 confirmation against 1.0000 A/A. These are
local-harness measurements rather than evaluator-issued receipts. This plan does
not claim a hidden-set result. The row cache and m01 pipeline retain their own
isolated gates above; P1 does not reopen or re-measure them as a speed claim.
