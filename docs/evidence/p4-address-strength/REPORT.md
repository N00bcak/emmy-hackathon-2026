# P4 address strength reduction: experimental, neutral

## Outcome

The bounded P4 candidate is correct on every required local gate but shows no
resolved hosted performance win. It remains on branch
`compiler-design/p4-address-strength` and is **not merged**. Baseline
`compile.py` remains `067ef9c4ce31a4bf5ccda90a8d13e9bd1b71b89aff19abe7598cdba4c77e2921`.

Candidate source: [compile.py](compile.py), SHA-256
`a057a3bdc7e0f55c7959960d9ceeb4b370c0f4b3a126fbec46b2b5674f4357c1`.

## Mechanism

The pass adds a direct address path for an index with one variable term, zero
constant, a positive coefficient, a current i32 binding, a proven nonnegative
range, and a complete byte scale fitting signed32. It emits `mul.wide.s32` from
the coordinate directly instead of flattening an element offset and then
widening. Everything else, including offsets, sums, negative scales, element
offsets, shared dynamic slabs and unproven ranges, keeps the generic path.

This is a lowering simplification, not a new memory-access order or ownership
rule. It does not alter ABI, output membership, store visibility or alignment
fallbacks.

## Correctness

- CPU suite: 47/47, including the new positive/fallback tests.
- Published gate: 15/15 at exact metadata tolerances.
- Routine audited core65: 65/65, 16/16 sanitizer checks, source unchanged.
- Eight scaled full-output witnesses: 8/8.
- Local proxy: RTX 4060 Laptop GPU, sm_89.

Receipts are in [published.json](published.json), [core65.json](core65.json) and
[scale_gate.json](scale_gate.json). The copied test file is
[test_compiler.py](test_compiler.py).

## Hosted RTX 5090 timing

Seven alternating pairs, warmup 10, 50 iterations, frozen scaled cases. The
A/A controls for the corresponding pointwise mechanism were median 1.0000, with
observed absolute maxima 6.8% (p03) and 5.3% (f01). Those bounds are conservative
noise estimates for these short kernels; the verdicts remain unresolved.

| Case | Ratio median | A/A deadband | Verdict |
| --- | ---: | ---: | --- |
| `p03_relu_mul_4m` | 1.0018 | 0.0684 | unresolved |
| `p02_scale_shift_8m` | 1.0000 | 0.0525 | unresolved |

The focused p02 mechanism removed a `SHF.L.U32` in local SASS, but that did not
produce a resolved wall-clock effect. Raw results are [p02.json](p02.json) and
[cells.json](cells.json).

## Disposition and limits

Keep the candidate experimental; do not merge it without a fresh resolved win.
The local SASS observation is mechanism evidence only. No hidden-set, hosted
correctness or streaming-throughput claim follows from this receipt. Re-open P4
only with a dynamic mechanism that predicts a larger effect at the same
measurable scale.
