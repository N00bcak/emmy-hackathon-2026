# RTX 4060 `f16` and scalar-math breadth candidate

This directory is the durable record for the `f16` storage plus scalar-math
breadth candidate. It is local RTX 4060 (sm_89) evidence. It does not establish
hidden-set correctness, RTX 5090 parity, or evaluator-issued performance
qualification.

## Frozen identities

- Compiler source: [source/compile.py](source/compile.py), SHA-256
  `6db2dad4944d87ae46194fd4deaf9331327cda21b21adaa434a1db41f515e6ba`
- Compiler tests: [source/test_compiler.py](source/test_compiler.py), SHA-256
  `d165117170502010b6f3c56b16c9c20c188c6b54a8e730d9ab41c837a87183aa`
- Variant generator: [source/variant_suite.py](source/variant_suite.py),
  SHA-256 `7aabc32724c97d66d4cb34ea6e5800b562474cda4f5e3e82d6b3326d0c76b0c6`
- Frozen timing baseline: [timing/baseline-compile.py](timing/baseline-compile.py),
  SHA-256 `8ebd86a7822c503c14767f46cc93b18590a87fc31195dae7ba9be5e0c82abf78`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9

Every artifact below records its own source identity. The gate, variant suite,
sanitizer, and timing all name compiler hash `6db2dad4...` above. The baseline
is the same frozen baseline used by experiment `014`, so the timing columns are
directly comparable.

## Correctness evidence

- [Gate](gate/results.json): all 15 published problems pass through the
  unchanged upstream `submit.py` client, plus a reordered-ABI probe. It reports
  `all15_passed: true`, `missing: []`, `failed: []`, and `source_unchanged: true`.
  This is published-case correctness only; hidden-set correctness is unverified.
- [Full suite log](logs/full-suite.log): the focused compiler suite is 31/31
  passing. The full repository discovery run is 158 tests with 149 passing and
  nine expected non-root preparation errors requiring Linux effective UID 0 and
  libseccomp; those errors occur before any compiler invocation.
- [Typed variants](variants2/gpu-results.json): 18 generated cases in three
  isolated batches, five repetitions each, all checks and guard buffers passing.
  They cover the frozen ABI, integer addressing and 64-bit division, f64, f16
  storage and promotion, flattening, partial reductions and partial warps,
  transcendental and scalar math (`sin`, `cos`, `erf`, `gelu`, `gelu_tanh`,
  `pow`, `sign`), casts, selection, bitcast, and vector/scalar copies.
- [Sanitizer](sanitizer/): Compute Sanitizer memcheck reports zero errors for
  published `m02_matmul_rect` and a generated atomic matmul, five isolated
  repetitions each. See [sha256.txt](sanitizer/sha256.txt) for artifact hashes.
  The two cubins are byte-identical to the `014` sanitizer cubins, so this is a
  fresh receipt for the same machine code, not new code-path coverage.

## Paired timing

The protocol is `cuda-graph-paired-v2`: 16 alternating candidate/baseline pairs
per case, each kernel measured by CUDA events, with reset/copy work excluded.
All paired blocks completed; medians below are medians of the 16 per-pair
ratios. Every case has a matching 16-pair A/A control. Three controls are exactly
`1.0000`; the two attention controls measure `1.0041` and `1.0045`, which is the
observed noise floor for those kernels.

| Case | Candidate us | Baseline us | Median ratio | A/A median |
|---|---:|---:|---:|---:|
| `m01_matmul_512` | 517.120 | 1056.896 | 0.4893 | 1.0000 |
| `m02_matmul_rect` | 10.240 | 10.384 | 0.9831 | 1.0000 |
| `f02_row_softmax` | 4.096 | 4.096 | 1.0000 | 1.0000 |
| `a01_attention` | 236.544 | 235.520 | 1.0043 | 1.0041 |
| `a02_attention_128` | 112.704 | 112.640 | 1.0000 | 1.0045 |

The m01 win is large and consistent. The m02 result is small and noisy; the
earlier `014` block measured `0.9453` for the same cubin pair, so this report
does not claim a stable m02 improvement. Softmax is timer-quantized and the
attention cases are within the A/A noise floor. These are indicative local
paired diagnostics with `benchmark_qualified: false`, not an evaluator speed
qualifier.

## Reproduction

Run from the repository root with the local runner admitted on `sm_89`. The
gate must pass before any timing replay, and any source edit invalidates this
entire record.

```sh
.venv/bin/python -m unittest tests.test_compiler -v
python3 -m unittest discover -s tests -v
PYTHONPATH=. .venv/bin/python tools/variant_suite.py experiments/015-rtx4060-f16-scalar-math/variants-replay --run-gpu --repetitions 5
compute-sanitizer --tool memcheck --error-exitcode 86 \
  .venv/bin/python -m tools.research_gpu \
  experiments/015-rtx4060-f16-scalar-math/sanitizer/results/m02-job.json
.venv/bin/python -m tools.research_gpu \
  experiments/015-rtx4060-f16-scalar-math/timing/pair-m01_matmul_512.json.job.json
```

Exact job inputs are retained in the per-job JSON files under `timing/` and
`sanitizer/results/`, and the frozen PTX/cubin pair for each timing anchor is in
`timing/prepared/prepared/`.

The timing protocol alternates the two arms inside one captured CUDA graph, but
it does not lock clocks or claim parity with the hosted RTX 5090 runner. Each job
JSON names the exact case directory and absolute cubin paths; use those files
rather than reconstructing the launch metadata.

## Limits

- Published fixtures and generated typed variants do not prove hidden-set
  correctness or arbitrary legal schedules.
- `bf16`, swizzled `cp.async` (`B32`/`B64`/`B128`), wider-than-32-bit shuffle
  forms, general named-barrier and non-warp tree forms, shared/integer/non-add
  atomics, and `ZeroPrologue` remain outside the verified surface.
- The scalar approximations (`erf`, `gelu`, `gelu_tanh`) are checked against a
  NumPy/SciPy independent reference only on the generated variant domain.
- The local runner is a trusted research tool, not a hostile-kernel sandbox.
- Tensor-core lowering is not implemented. The recognizer/emitter boundary and
  planned precision/split-K work are described in
  [tensor-core-plan.md](../../docs/tensor-core-plan.md).
