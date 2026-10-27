---
name: kernel-reflection
description: Explain kernel compiler experiment results, retain mechanisms and failures, and choose the next research question without losing provenance.
---

Use this skill for an explicitly requested discovery campaign or its named stage.
It does not start a campaign, background service, workers, sweeps, or hosted calls
by being read. For ordinary bug repair, follow AGENTS.md and run relevant tests;
resume discovery only when the user asks.


For cross-device lowering, validation, submissions or performance claims, read
[compute-capability and submission guidance](../../../docs/compute-capability.md).
Keep RTX 4090/SM8.9 and RTX 5090/SM12.0 execution evidence separate.

Read root AGENTS.md and [the research guide](../../../docs/research-loop.md).
Inspect actual frozen source, generated code and measurements. Distinguish:
what changed in lowering; whether the mechanism activated; measured behavior;
and whether the evidence attributes that behavior to the mechanism. Use ablations
when they resolve a material alternative explanation, not as a ritual.

Classify compiler, assembly, ABI, numerical, memory/synchronization, infrastructure,
and performance failures. Infrastructure errors are unverified, not wrong answers.
Keep failed evidence and counterexamples; a useful null result is valid progress.
Never promote incorrect code or an exploratory sweep winner without fresh isolated
confirmation. Treat ambiguous gains as inconclusive and show per-case regressions.
Wall-clock latency is the ultimate arbiter; SASS issue/instruction/memory counts,
spills and occupancy explain a mechanism but cannot turn an unresolved or
negative timing result into a win, or a confirmed win into a loss.

Write a concise mechanism finding: applicable IR conditions, legality assumptions,
PTX/SASS effect, supporting experiments, contradictions, prerequisites and next
uncertainty. Update the experiment handoff and coordinator strategy. Subagent
returns contain outcome, key finding, evidence links, blocker and next action;
raw logs stay in the configured durable research store. Storage and execution
details come from private runtime configuration, not this skill.
Do not emit hidden reasoning transcripts.

An optimization is activated only when its intended effect appears in final
post-`ptxas` SASS on every timing target, not merely in PTX. If `ptxas` removed,
rebuilt or cancelled it, the worker must re-implement the lowering to preserve
the mechanism or fail closed and report it inactive; do not attribute timing to
a non-surviving change.

Only the coordinator records shortlist/promotion with evaluator-issued evidence.
Require the source/evaluator-bound all-15 published correctness gate for baseline
and candidate before accepting any performance evidence; no partial-family or
performance-debt exception can waive correctness.
Preserve the previous evaluated baseline and retest stale-parent candidates against
the current one before advancement. Coverage records need supplied anchors,
generated cases and existing regressions; atomic coverage also needs sanitizer.
Keep both coverage responsibilities until ALL five gaps close, then release them
to speed. No hidden-set or 5090 claim follows automatically from local success.

Budgeted hosted calibration is for evaluated shortlists only. Accumulate evidence
per mechanism and regime, record uncertainty/rank reversals, and leave local
performance advisory where cross-device rankings disagree. Do not spend unused
request headroom without a useful reason. Finish within the run budget with a
replayable checkpoint; compaction does not erase unfinished research.
