---
name: kernel-experiment
description: Implement and evaluate a falsifiable Emmy compiler hypothesis with reproducible correctness and timing evidence.
---

Read root AGENTS.md. Before edits write experiments/<id>/plan.md: strategy,
hypothesis, predicted observation, implementation idea, baseline, affected semantics,
correctness cases, performance protocol, budget and rejection criteria.
Preserve unrelated changes. Implement typed IR lowering, never parse body strings.
Fail unsupported operations explicitly with nonzero exit and stderr diagnostics.

Check the metadata ABI, tail guards without divergent barriers, index width,
signedness, vector alignment, reduction identities/order, coupled softmax state,
shared-memory visibility, async-copy waits, atomic resets and ordered multi-kernel
execution as applicable. Emit ASCII PTX with target-compatible version.
Run focused tests, then tools/evaluate.py using upstream submit.py with our compiler.
Keep reproducing IR, generated PTX and exact diagnostics for failures.
Assembly alone, output sums and truncated outputs cannot establish correctness.
Add legal non-fixture shapes/ops, tails, constants and numerical edge cases as
coverage grows. Generate IR through supported tooling or justified typed fixtures;
never arbitrarily patch serialized bodies. Emmy frontend generation is allowed,
but delegating submitted lowering to an existing compiler is not.

Benchmark only correct candidates after regression checks. submit.py uses
warmup=0/iters=1; its reported latency is smoke feedback, not speedup evidence.
Implement a separate helper using documented /compile and /run when benchmarking
is needed; do not alter the upstream harness. Preserve arg_order, metadata launch,
zero_outputs before EVERY launch, full tolerance checks and intermediate chaining.
Warm baseline and candidate, alternate paired runs on identical hardware, retain
raw samples, and report medians and spread. Respect warmup<=20, iters<=200,
20-second launch limit, sequential shared-runner use and bounded busy retries.
Record GPU, arch, tool versions, compile time, register count and timing settings.
Reject gains within observed noise. Derive throughput only from explicit work/byte
counts; check correctness under repeated launches, especially atomic outputs.
