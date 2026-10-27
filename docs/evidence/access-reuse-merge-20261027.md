# P1 and P3 integration on `n00bcak`

The compiler incorporates two independent mechanisms from the access-reuse work:

- **P1:** prove aligned, contiguous four-lane `f32` accesses within a straight-line block. Global accesses use a runtime 16-byte alignment guard and retain scalar fallback. The descriptor rejects cross-write motion, index rebinding, and unsafe alias patterns.
- **P3:** batch complete sibling attention dot work when the serial B operand is shared across configuration items and the live-value budget is at most 64. Each accumulator retains its original fold order, followed by the existing score prescale.

P2 cross-phase row caching is not included. Its reuse-analysis and hoisting dependencies were not safely ported into this compiler.

## Exact source and tests

Combined `compile.py` SHA256: `713fa03fd2d348aa63e9b5cbc5c222e9fbf1b8a3ff0eb92a4b9351d3b64a7379`. The earlier individual P1 and P3 source SHA256 values were `5da1941927c226d2c53dd274b9749cd5b03ff3261cfe5444fb16fa12066fb281` and `1412ce6a5f0d29594c00ee0e7201c62782160c7686b2146a6c2f92f3295`. The combined source emits PTX byte-for-byte identical to those individually evaluated ports for `p03_relu_mul_4m` (`497f90d9ae17ebcef817f1eda8426cfd8161c5e3da3d5906df6e8b3376ab616d`) and `a01_attention_512x64` (`a9f8a6f5fb9f4bb6080192578d97a30b82f672b45d8b45b0a167874d9855b402`), respectively.

The unchanged [100-check plan](../../tests/experiment-gate.json) selected 66 GPU cases and 34 CPU leaf checks; plan ID `7c908b3e8218e285572c6435ce274b29d7a6eb90595b0f18bf0746eb3bdf320d`. Both sm89 and sm120 assembly passed. On the RTX 4090, the merged source passed all 66 GPU cases with width-one guarded correctness and three repetitions, including all 15 published contracts at their exact metadata tolerances. The 16 existing atomic cases passed separate Compute Sanitizer memcheck jobs with zero errors. The 34 CPU leaf checks passed: 29 methods, six parameter subchecks, zero failures, errors, or skips. No case, reference, tolerance, fixture, or test code was changed or added.

The [source-bound receipt](access-reuse-combined-100-check-20261027.json) records the plan identity, source and evaluator inputs, exact case IDs, preparation, and retained GPU job hashes. Raw IR, inputs, references, PTX, cubins, SASS, result JSON, and host admission records are in campaign `speculative-access-reuse-20261027` under the configured durable research root. All eleven regression and sanitizer jobs exited 0 with `observed_no_contention=true`. The receipt is a low-level correctness record; it is not a new Store promotion.

## Performance interpretation

The completed official RTX 5090 batch measured the individual ports. P1's p03 4M PTX was about 18% faster than the current compiler across three pairs. P3's a01 512x64 PTX was about 1.7% faster, below the batch's 2% decision margin. The combined compiler emits exactly those same PTX bytes for those cases, so the case-specific official execution applies to those generated kernels. This does not establish a general 5090 speedup or hidden-set correctness. On the 4090, prior paired timing for P1 p03 was unresolved because the same-source control varied sharply; P3 a01 showed a small gain within the control variation. Other per-case latency regressions were not measured in this integration check. See [the speculative comparison](access-reuse-speculative-20261027.md) for raw paired values and the official request record.
