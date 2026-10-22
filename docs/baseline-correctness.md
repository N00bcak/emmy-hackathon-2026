# Correctness-first baseline

The pre-tensor-core compiler snapshot with SHA-256
`6db2dad4944d87ae46194fd4deaf9331327cda21b21adaa434a1db41f515e6ba`
passes all 15 published problems on the local RTX 4060 Laptop GPU (sm_89) at
their original metadata tolerances. [Per-case results](baseline-correctness.json)
record the observed errors from that complete gate, and
[the gate report](../experiments/015-rtx4060-f16-scalar-math/gate/results.json)
binds the pass to the frozen source hash. This is published-case correctness,
not a claim about the hidden set. The tensor-core increment is a separate source
identity, gated independently and summarized in
[the tensor-core evidence bundle](../experiments/tensor-core-v1/README.md).

## What changed

The emitter walks typed Kernel IR rather than recognizing fixture names. It
supports scalar-by-value ABI parameters, non-unit 2-D/3-D launch flattening,
carried state, partial-CTA collective reductions, non-sum reduction identities,
natural `log`, explicit dtype conversion at SSA and store boundaries, and
`f16` storage with promoted arithmetic. Index expressions preserve Emmy's
floor-division/floor-remainder semantics, including negative operands. Shared
memory respects each declaration's natural dtype alignment as well as its
requested alignment.

Address and integer-expression CSE is versioned and invalidated at joins,
backedges, and mutable-state updates. Loop-invariant scalar index expressions
used more than once are computed in the loop preheader. A regression test guards
the m01 staging quotient/remainder; an attempted extension that hoisted whole
address chains was rejected because it regressed attention. Shared transport
has a vectorized f32 path only when the slab is 16-byte aligned and the index
residue is proven; it performs a runtime address check and falls back to scalar
copies otherwise. All publish-facing arithmetic and copies retain a generic
typed fallback. `cp.async` recombines the slab's 32-bit shared-window position
with each destination's offset; a dynamic slab's window position is nonzero on
sm_120, so an offset-only destination silently addresses the wrong storage on
the hosted GPU.

The `015` increment also adds typed `f16` loads/stores and shared-memory
carriers, `f16` to/from `f32`, `f64`, and integer conversion, and the scalar
math ops `sin`, `cos`, `tan`, `pow`, `sign`, `erf`, `gelu`, and `gelu_tanh`.
The correctness gate and generated variants cover those paths at the hash above.

## Verification

- The complete 15-case gate ran with the unchanged upstream `submit.py` client
  against the admitted local runner (`sm_89`, RTX 4060). All 15 cases exited
  zero with `all15_passed: true`, `missing: []`, and `failed: []`. A separate
  reordered-ABI probe also passed.
- The current non-root runnable suite passes 157 tests, including two-target
  assembly, partial-CTA masking, scalar ABI, integer copy, natural log, reduction
  aliases, vector-copy emission/runtime fallback, `f16` storage/promotion, the
  scalar-math lowering, the m01 invariant-index CSE regression, and 13
  tensor-core tests. The tests assert emitted instructions and typed proofs, not
  bit-exact numerical equivalence for every approximation or NaN/signed-zero
  edge case.
- Full `unittest discover` finds 171 tests; nine preparation tests error in this
  non-root environment because the restricted compile boundary requires
  effective UID 0 and libseccomp. Those errors occur before any compiler
  invocation. The pre-increment `015` snapshot's 158-test boundary record stays
  unchanged in [the machine-readable baseline](baseline-correctness.json).
- The hosted runner (RTX 5090, `sm_120`) reproduced the reported
  `m01_matmul_512` failure on the pre-fix `main` source (`max abs err
  1.454e+02`). With the shared-window correction, the same source passed m01 on
  that runner (`max abs err 8.392e-05`) and on the local sm_89 runner
  (`8.392e-05`). This is one-case hosted execution evidence; the hidden set is
  still unverified.
- [Typed variants](../experiments/015-rtx4060-f16-scalar-math/variants2/gpu-results.json)
  ran 18 generated cases in isolated batches with five repetitions each; all
  checks and guard buffers passed.
- Paired CUDA-event measurements against the frozen pre-CSE compiler
  (`8ebd86a...`) used 16 pairs per case at unchanged launch metadata. The `015`
  medians were: `m01_matmul_512` 517.120 vs 1056.896 us (ratio 0.4893),
  `m02_matmul_rect` 10.240 vs 10.384 us (0.9831), `a01_attention` 236.544 vs
  235.520 us (1.0043), `a02_attention_128` 112.704 vs 112.640 us (1.0000), and
  `f02_row_softmax` 4.096 us for both (1.0000). Only `m01` is a clear
  improvement; `m02` moved from 0.9453 in `014` to 0.9831 here for the same
  cubin pair, so it is not claimed as a stable win. The remaining cases are
  neutral within measurement resolution. These are indicative local paired
  diagnostics, not an evaluator-issued performance qualification. Matching
  16-pair A/A controls give `1.0000` for `m01`, `m02`, and `f02`; the attention
  controls measure `1.0041` and `1.0045`, matching their candidate ratios. The
  controls and exact per-pair artifacts are in
  [the `015` experiment directory](../experiments/015-rtx4060-f16-scalar-math/).
- The rejected full-address-hoist prototype regressed `a01_attention` to
  `1.474x` the baseline; it was not promoted and is retained separately under
  `experiments/006-optimizer-hoist/`.
- Compute Sanitizer memcheck passed `m02_matmul_rect` and an independently
  generated typed atomic-matmul case (`7x5x9`) for the `015` hash, with five
  isolated repetitions each, zero reported sanitizer errors, and all output and
  guard checks passing. Both cubins are byte-identical to the earlier `014`
  sanitizer artifacts, so this is a fresh receipt for the same machine code.

The `015` manifests and `sha256.txt` retain cubin hashes, but compiled cubins
are intentionally ignored artifacts and are not committed. They are
reproducible from the recorded source, exact PTX/cubin/`ptxas` toolchain, and
commands; a fresh checkout does not contain the binary bytes.

## Remaining correctness work

`TASK.md` marks `*` forms as absent from both open and hidden problems. The
following are explicit coverage gaps, not verified failures. Only the first two
groups are within the challenge-contract work queue; starred forms are deferred
unless a concrete contract-relevant failure requires them.

- Contract-relevant gaps: `bitwise_not` and boolean mask forms lack a durable
  variant; `maximum`/`minimum` NaN and signed-zero semantics need an independent
  typed reference before changing the current `max`/`min` lowering; and the
  scalar `erf`/`gelu`/`gelu_tanh` approximations are validated only on the
  generated variant domain.
- Performance-transport limits: the current compiler emits `cp.async` only for
  proven-wide, aligned copies and has no TMA, `ldmatrix`, or `stmatrix` path.
  TMA, `cp.async.bulk`, and `stmatrix` require `sm_90` or newer; the local
  RTX 4060 is `sm_89`. PTX targeting `sm_89` cannot be used to claim their
  execution evidence.
- Capability limits: non-warp `TreeHalve`, general named-barrier forms, shuffle
  widths above 32 bits, and non-add/integer atomics are unverified or rejected.
  `TASK.md` does not require every constructible form, so these need a concrete
  contract-relevant failure before they enter the work queue.
- Starred constructs, explicitly outside the challenge contract: `bf16`,
  `CpAsyncCopy` swizzles `B32`/`B64`/`B128`, `raster_group` application,
  shared atomics, `ZeroPrologue`, TMA, Emmy's explicit tensor-core fragment
  nodes, and `wgmma`. The non-starred SIMT vocabulary is not a restriction on
  legal instruction selection: the current compiler lowers a proved contraction
  from ordinary `Load`/`multiply`/`Accum`/`Write` dataflow to `mma.sync` when all
  plan invariants hold.

## Limits

This evidence does not establish hidden-set correctness, general support for
all legal schedules, full RTX 5090 behavior, or a comparative speedup on another
machine. The hosted 5090 evidence covers m01 correctness, not the remaining
published cases. The CSE and `015` results are local RTX 4060 diagnostics rather
than an evaluator-issued performance qualification. The local runner is an admitted
research tool, not a hostile kernel sandbox. The current source emits a
three-term TF32 `mma.sync` path for proved one-atom-per-warp matmul plans; the
generic typed scalar path remains the byte-identical correctness fallback for
every unmatched or unsupported case. The implemented path, corpus gate, and
remaining limits are documented in [tensor-core-plan.md](tensor-core-plan.md)
and [the tensor-core evidence bundle](../experiments/tensor-core-v1/README.md).
