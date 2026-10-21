---
name: kernel-strategy
description: Choose strategies for general Emmy IR to PTX lowering, correctness coverage and measured efficiency.
---

Read root AGENTS.md and the challenge documents it names. Inspect current code,
prior experiment records and unsupported statement/expression kinds and dtypes.
Start with typed Graph.from_dict loading, KernelOp discovery, metadata ABI,
expression/register emission and pointwise lowering if no compiler exists.
Pin emmy-ml==0.3.11 and Python >=3.12. Read reference CUDA for semantics only;
never use its renderer for lowering or recognize fixtures to emit canned outputs.

Write experiments/strategy.md: current category coverage, evidence paths,
bottleneck, candidate strategies, chosen strategy, reason, hypothesis queue,
time/request budget and stop condition. Prioritize the 8 MVP pointwise/reduction
problems before matmul, fused and attention. Retain rejected ideas and reasons.
Distinguish compiler time, assembly time, launch latency and GPU execution time.
For correct kernels, investigate coalescing, vector alignment, register pressure,
shared-memory traffic, synchronization and redundant work where IR semantics and
fixed launch permit. Choose one falsifiable change rather than an unbounded search.
