# RTX 4060 `cp.async` candidate

> **Superseded by `015-rtx4060-f16-scalar-math`.** This record is valid only for
> compiler hash `b3b0a3ac...`. Use `015` for the current compiler, gate, and
> reproduction commands; this directory remains the durable prior-stage evidence.

This directory is the durable record for the `cp.async` candidate. It is local
RTX 4060 (sm_89) evidence. It does not establish hidden-set correctness, RTX
5090 parity, or evaluator-issued performance qualification.

## Frozen identities

- Compiler source: [source/compile.py](source/compile.py), SHA-256
  `b3b0a3acd6b7373cb3707ea50bcf99c621fff865acbc92db650f6c753c4ee045`
- Compiler tests: [source/test_compiler.py](source/test_compiler.py), SHA-256
  `096c7590080b84ca48d877100ef9c4ba291f8bb43083d1ffd34e0234e2c730d2`
- Variant generator: [source/variant_suite.py](source/variant_suite.py),
  SHA-256 `cbb090954f3985defbffb01fc3fd3228955112af02a0ea5b53df2be1ffdb7b77`
- Frozen timing baseline: [timing/baseline-compile.py](timing/baseline-compile.py),
  SHA-256 `8ebd86a7822c503c14767f46cc93b18590a87fc31195dae7ba9be5e0c82abf78`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9

Each artifact records its own source identity. The gate, variants, sanitizer,
and timing all name compiler hash `b3b0a3ac...` above.

## Results

- [Gate](gate/results.json): all 15 published problems pass through the
  unchanged upstream `submit.py` client, plus the reordered-ABI case. The gate
  reports `all15_passed: true`, no missing or failed published cases, and
  `source_unchanged: true`.
- [Compiler tests](logs/full-suite.log): all 29 compiler tests pass. The full
  repository suite is 156 tests with 147 passing and nine preparation tests
  erroring because this non-root session lacks the required libseccomp compile
  boundary; those tests do not reach the compiler.
- [Typed variants](variants/gpu-results.json): 16 generated cases, one isolated
  GPU batch per case, five repetitions each, all checks and guard buffers
  passing. They cover scalar ABI, integer addressing and 64-bit division, f64,
  flattening, partial reductions, transcendental operations, casts, selection,
  bitcast, partial warps, and vector/scalar copies.
- [Sanitizer](sanitizer/): Compute Sanitizer memcheck reports zero errors for
  published `m02_matmul_rect` and a generated atomic matmul, five repetitions
  each. See [sha256.txt](sanitizer/sha256.txt) for the exact compiled artifacts.

## Paired timing

The protocol is `cuda-graph-paired-v2`: 16 alternating candidate/baseline pairs
per case, with each kernel measured by CUDA events and reset/copy work excluded.
Medians are computed as the mean of samples 8 and 9 of the sorted per-arm
values. A/A controls use the same candidate for both arms.

| Case | Candidate us | Baseline us | Median ratio | A/A median |
|---|---:|---:|---:|---:|
| `m01_matmul_512` | 517.264 | 1056.768 | 0.4895 | 1.0004 |
| `m02_matmul_rect` | 10.240 | 10.832 | 0.9453 | 1.0000 |
| `f02_row_softmax` | 4.096 | 4.096 | 1.0000 | 1.0000 |
| `a01_attention` | 245.440 | 243.712 | 1.0071 | 1.0080 |
| `a02_attention_128` | 117.424 | 116.736 | 1.0059 | 1.0019 |

The matmul cases improve; softmax is timer-quantized; attention is within the
A/A noise floor. These are indicative local paired diagnostics with
`benchmark_qualified: false`, not an evaluator speed qualifier.

## Reproduction

Run the gate with the unchanged upstream client against an admitted local
runner, then run the isolated variant suite, sanitizer checks, and finally
paired timing only after every correctness check passes. Exact commands and
inputs are retained in the per-job JSON files. Any source edit invalidates this
entire record.

```sh
.venv/bin/python -m unittest tests.test_compiler -v
.venv/bin/python tools/variant_suite.py --help
```

## Limits

- Published fixtures and generated typed variants do not prove hidden-set
  correctness or arbitrary legal schedules.
- `cp.async` currently covers nonswizzled 4/8/16-byte global-to-shared copies.
  Swizzled copies, wider shuffle forms, named-barrier variants, some atomic
  forms, and `f16`/`bf16` storage remain outside the verified surface.
- Tensor-core lowering is not implemented. The recognizer/emitter boundary and
  planned precision/split-K work are described in
  [tensor-core-plan.md](../../docs/tensor-core-plan.md).
- The local runner is a trusted research tool, not a hostile-kernel sandbox.

