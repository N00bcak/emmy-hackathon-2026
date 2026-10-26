# P2 shared value reuse: experimental, no resolved timing win

## Outcome

P2 implements the shared typed value-reuse analysis and activates f02's scalar
second-pass row cache, but the hosted RTX 5090 row-reuse primaries are all
unresolved near 1.00. The candidate therefore does **not** meet the P2 timing
exit and is retained on branch `compiler-design/p2-shared-reuse`, not merged.

- Base: `d8f65f0`
- Candidate `compile.py` SHA-256: `4479b8523799050521469b475f4826c658b0bdde11f60a08e087150d1b63d79f`
- Branch: `compiler-design/p2-shared-reuse`
- Merged baseline stays at `067ef9c4ce31a4bf5ccda90a8d13e9bd1b71b89aff19abe7598cdba4c77e2921`

## Mechanism

`ReuseValue`, `reuse_expr_key`, and `ReuseAnalysis` provide typed structural
identity, transitive readonly-load summaries, CTA-uniformity facts, and
fail-closed shared-hoist admission. The scalar phase path caches the exact
per-thread row values between a reduction and its following map, with strict
header, affine-load, alias, register-budget and fold-order checks. The existing
four-wide f01/f03 row cache remains ahead of the scalar path.

## Correctness

All checks used the candidate hash above:

- CPU suite: 49/49 pass, including scalar-cache activation and fail-closed
  tests plus structural A79/A83 coverage.
- Published contracts: 15/15 pass at exact metadata tolerances.
- Routine audited policy: 65/65 pass, including A79/A83; 16/16 sanitizer checks;
  source unchanged.
- Scaled full-output witnesses: 8/8 pass.

Receipts: [published.json](published.json), [core65.json](core65.json),
[scale_gate.json](scale_gate.json). The candidate and test files are preserved
as [compile.py](compile.py) and [test_compiler.py](test_compiler.py).

## Hosted RTX 5090 timing

Seven alternating pairs, warmup 10, 50 iterations. A/A for f02 was median
1.0028 with a 1.69% observed absolute maximum, so its calibrated deadband is
1.69%. The f01 control's A/A maximum was 5.25%; no P2 candidate result exceeds
that conservative band.

| Case | Candidate ratio median | Deadband | Verdict |
| --- | ---: | ---: | --- |
| `f01_rmsnorm_3072x2048` | 1.0000 | 1.0% | unresolved |
| `f02_row_softmax_3072x2048` | 1.0019 | 1.69% | unresolved |
| `f03_l2norm_3072x2048` | 1.0025 | 1.0% | unresolved |

Raw results: [f01_primary.json](f01_primary.json),
[f02_primary.json](f02_primary.json), [f03_primary.json](f03_primary.json), and
the [f02 A/A control](aa_f02.json). No resolved regression or win is claimed.

## Disposition and limits

The general analysis and f02 activation are correct and are useful experimental
work, but the plan requires a resolved row-reuse timing win plus fresh
confirmation. That evidence does not exist, so P2 remains unresolved/deferred
and is not merged. P3 was not started because the plan explicitly makes it
dependent on P1/P2 being in place and because no P2 win establishes the reuse
budget P3 needs.

This is local full-output correctness plus hosted timing on a frozen source. No
hidden-set or streaming-throughput claim follows.
