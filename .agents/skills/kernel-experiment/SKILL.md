---
name: kernel-experiment
description: Execute a kernel compiler hypothesis or bounded sweep with cheap filters, isolated correctness, paired timing and replayable provenance.
---

Use this skill for an explicitly requested discovery campaign or its named stage.
It does not start a campaign, background service, workers, sweeps, or hosted calls
by being read. For ordinary bug repair, follow AGENTS.md and run relevant tests;
resume discovery only when the user asks.


For cross-device lowering, validation, submissions or performance claims, read
[compute-capability and submission guidance](../../../docs/compute-capability.md).
Keep RTX 4090/SM8.9 and RTX 5090/SM12.0 execution evidence separate.

Read root AGENTS.md and [the research guide](../../../docs/research-loop.md).
Use an isolated evaluation workspace derived from a verified baseline, in the
user-configured runtime. Shared instructions prescribe neither Docker nor NFS;
connection details, paths, mounts and launch commands belong in private runtime
configuration. The supplied restricted evaluator requires Linux/CUDA; other
adapters must preserve its isolation and evidence guarantees, not bypass them.
Record a brief and handoff in the durable research store before substantial work;
search prior evidence first. Make
research choices freely within the authorized question and budget. Explain pivots.

Implement direct typed IR lowering. Preserve ABI/launch, numerical semantics,
collective participation, memory visibility, atomic resets and pipeline order.
Do not edit evaluator code, generator policies, tolerances, or expected outputs to
make your candidate pass. Report suspected evaluator defects for independent repair
under a new evaluation version. Assembly and output sums never prove correctness.

Freeze actual source before confirmation cases are drawn. CPU preparation runs the
compiler under its restricted identity and caches cases/references and assembly.
Inspect both-target SASS activation and resources before GPU timing. A mechanism
that did not activate needs explanation/revision; more registers alone is not a
reason to reject it. Preserve diagnostics, failed artifacts and exact reproducers.

All GPU work goes through host admission. Let the shared scheduler own dispatch.
Concurrent correctness screens guarded private buffers; corruption quarantines
all implicated batchmates. A familiar mechanism with a new bad parameter is still
a possible memory bug. m02 atomics always need isolated-first and Compute Sanitizer.
Before timing or treating any GPU benchmark as valid, both frozen baseline and
candidate must pass ALL 15 published cases at unchanged metadata tolerances under
the current evaluator identity, including isolated atomic/reset and sanitizer
checks. Partial correctness probes are debugging only. Source or evaluator
changes invalidate this gate. Every survivor still needs fresh isolated
confirmation before shortlist/promotion.

Use exclusive paired timing with per-case A/A noise, bounded adaptive blocks,
persisted burst identities, full-value checks and prescribed resets. Finish a
started paired block before yielding; incomplete hard-timeout blocks are not
comparison evidence. A sweep winner still needs fresh cases and new measurements.
Label warm-cache versus rotating-buffer behavior; never claim streaming throughput
from warm-cache timings. Keep compiler/assembly/runtime costs separate.

Persist source, IR/meta, realized inputs, references/tolerances, PTX/cubin/SASS,
commands, environment, admission, output checks and raw samples in that store. Update the
handoff with live jobs/dependencies/next executable action before yielding. Only
GPU-evaluated shortlists may return to the owner's repository; no Git on a remote
GPU host or Git remote work.
