# Historical RTX 4060 scalar-CSE evidence

> **Historical, superseded record.** This file retains the original measurements
> for compiler hash `1c300b0...`. It is not current evidence and its "final"
> title refers only to that earlier CSE candidate. For the current compiler use
> `014-rtx4060-cpasync-static` as the prior stage and
> `015-rtx4060-f16-scalar-math` for the verified baseline.

This directory contains the local sm_89 evidence for the final scalar-CSE
candidate. It is **not** RTX 5090 evidence and does not establish hidden-set
correctness.

## Frozen identities

- Candidate source: `compile.py`,
  SHA-256 `1c300b0e1f3c340389bfff5f26a8ff532bba434e22fdad8391694eef52fd0df7`
- Paired timing baseline: SHA-256
  `8ebd86a7822c503c14767f46cc93b18590a87fc31195dae7ba9be5e0c82abf78`.
  The durable copies are
  [`014/timing/baseline-compile.py`](../014-rtx4060-cpasync-static/timing/baseline-compile.py)
  and
  [`015/timing/baseline-compile.py`](../015-rtx4060-f16-scalar-math/timing/baseline-compile.py);
  the original `/tmp` path was a transient workspace and is not durable.
- CUDA architecture: `sm_89`, NVIDIA GeForce RTX 4060 Laptop GPU
- Exact per-kernel IR/PTX/cubin identities:
  [`prepared/manifest.json`](prepared/manifest.json)

## Contents

- `gate/results.json` and `gate/published.log`: all 15 published cases pass
  through the unchanged upstream `submit.py` client against the admitted local
  runner, bound to the candidate hash above.
- `gate/compiler-tests.log`: compiler unit-test output.
- `variants/gpu-results.json`: ten independently generated typed variants passed
  on GPU. These are coverage diagnostics, not a promotion gate.
- `timing-*.json`: 16 paired CUDA-event samples per case against the frozen
  baseline. Per-kernel event intervals exclude resets and copies.
- `timing-*-job.json`: exact worker inputs.
- `aa-*.json`: paired A/A controls. Medians are essentially 1.0; rare large
  m01/f02 ratios are timer-quantization or system-noise outliers.
- `sanitizer/`: historical Compute Sanitizer evidence from an earlier suite. It
  predates the final source hash and is **not** a final-candidate sanitizer
  qualification.

## Result

| Case | Median candidate / baseline | Observed range | Reading |
|---|---:|---:|---|
| `m01_matmul_512` | 0.5077 | 0.2717–0.5086 | clear improvement |
| `m02_matmul_rect` | 0.9412 | 0.9412–0.9706 | small improvement |
| `f02_row_softmax` | 1.0000 | 0.875–1.0 | neutral; timer quantization |
| `a01_attention` | 1.0061 | 1.0–1.0091 | neutral |
| `a02_attention_128` | 1.0026 | 0.9922–1.0130 | neutral |

The low `m01` ratio of `0.2717` is a single noisy outlier; the other fifteen
pairs cluster tightly around `0.507–0.509`. Only `m01` is a clear improvement.

## Reproduction and limits

The gate was produced with the repository's unchanged upstream client and
admitted local runner (`http://127.0.0.1:48707`), with an exclusive GPU
admission. Every source change invalidates the gate and requires rerunning all 15
cases before timing. Rejected full-address hoisting remains in
`experiments/006-optimizer-hoist/` and is not part of this candidate.

The local runner is a trusted research tool, not a hostile-kernel sandbox.
Hosted/5090 parity, hidden-set correctness, and a final-hash sanitizer run remain
unverified.
