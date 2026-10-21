---
name: kernel-reflection
description: Reflect on compiler experiment evidence, retain failures and select the next hypothesis.
---

Read the plan, logs and actual diff. Separate observations from interpretations.
Classify failures: deserialization, compiler, assembly, ABI/launch, numerical,
infrastructure or performance. Infrastructure failure leaves the hypothesis
unverified; it is not a failed correctness test.

Write experiments/<id>/reflection.md: prediction versus observation, passed/failed/
unverified cases, timing distribution and hardware, supported cause, uncertainty,
keep/reject/inconclusive decision and next falsifiable hypothesis. Never promote
incorrect code because it is fast or claim speedup from one smoke timing.
Keep correctness fixes even when slower and record optimization debt. Report
per-case/category coverage rather than hiding regressions in an aggregate score.

Promote only after required regressions. Preserve failed evidence and reproducers.
Restore only experiment-owned edits, never reset user changes. Update strategy.md
with durable findings and the next action. Repeat within the authorized budget;
stop on completion, budget exhaustion or a genuine user-owned dependency.
Report only performed verification and explicitly name unverified requirements.
