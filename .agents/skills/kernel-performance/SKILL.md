---
name: kernel-performance
description: Measure and diagnose Emmy IR-to-PTX kernel performance using latency, useful FLOPs, MFU, memory traffic, hardware counters, and comparable references. Use for a performance audit or explaining slow kernels; does not start a discovery campaign or change the compiler.
---

Produce per-case evidence explaining where execution time goes, which limits come
from the prescribed schedule, and which compiler changes could plausibly help.
Do not label a latency good or bad merely because it is a few microseconds.

## Establish the measurement

Read repository AGENTS.md, the pinned challenge contract, and
[docs/research-loop.md](../../../docs/research-loop.md). Resolve execution,
admission, storage, and tooling from private runtime configuration. Inspect live
jobs before dispatch; coordinate with an active campaign rather than starting
another run. Follow its remaining budget and generated behavior-type cap.
Outside a campaign, use a bounded diagnostic/smoke run. Reading this skill starts
no workers, sweeps, hosted grading, profiler, or service. Do not interact with
another chat's workers without authorization.

Freeze compiler, evaluator, case/input/reference, and launch identities. Use
existing exact-identity evidence where valid. All 15 published cases must pass
at unchanged tolerances, with required isolation, resets and sanitizer checks,
before performance evaluation is valid. Both arms need gates for a comparison.
Keep invalid/unsupported cases visible; a compilation or assembly pass is not a
GPU correctness pass. Do not repair a broken compiler as part of an audit unless
implementation is also authorized.

Use the existing admitted evaluator for unprofiled GPU timings. Record per-kernel
and ordered-pipeline latency separately from compile, assembly, host submission,
copies, synchronization and resets; state precisely which intervals are included.
Retain complete raw samples, medians, dispersion, pair/block identities and
case-specific A/A noise. Compare paired arms in the same cache/input regime;
an overlapping noise band supports an inconclusive result. Preserve independent
post-selection confirmation requirements from kernel-experiment/reflection.

For tiny kernels, check timer quantization and submission gaps against an
appropriate empty/minimal control under the same timing protocol. Do not subtract
a guessed fixed launch overhead. Repeated-launch amplification or CUDA Graph
changes require a separately validated protocol; preserve every launch's atomic
resets and correctness protections. A profiler replay is not a replacement for
unprofiled latency. Report deferred admission as unmeasured.

## Account for work and traffic

Read [references/metrics-and-diagnosis.md](references/metrics-and-diagnosis.md)
before calculating FLOPs, MFU, bandwidth, rooflines, or bottleneck classifications.
Derive useful work from typed IR and actual iteration domains, not fixture names
or output dimensions alone. Independently check contractions, reductions,
predication, split reductions and multi-kernel pipelines. Count FMA as two FLOPs;
report exp/div/other special operations separately rather than assigning invented
equivalent FLOPs. Keep estimated useful work separate from measured executed
instructions/work, including padded cells and redundant computation.

Obtain the actual GPU model/SM count, relevant precision/instruction mode,
observed clocks and ceiling provenance. SM89 alone does not define throughput.
Use a matching SIMT/tensor/dtype/dense ceiling; unavailable or mismatched ceilings
make MFU unavailable, not zero. Separate theoretical specifications from measured
sustained ceilings. Do not change clocks or shared-host settings.

Calculate logical traffic and cache assumptions from the algorithm. Collect DRAM,
L2 and shared traffic independently where profiling permits. Logical bytes/time
is effective bandwidth, not measured DRAM bandwidth. Warm-cache effective
bandwidth can exceed device DRAM bandwidth without implying a hardware error.

For repeatable arithmetic, use `scripts/metrics.py INPUT.json > OUTPUT.json`.
Its schema and limitations are in the reference. It derives numbers from supplied
measurements; it does not measure hardware or issue correctness receipts.

## Attribute the bottleneck

Start from the per-case table and inspect only the counters needed to distinguish
competing explanations. Query the installed Nsight Compute version and supported
sections/metrics first. Common starting sections are SpeedOfLight,
LaunchStats, Occupancy, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis and
SchedulerStats; add source/SASS counters only for the remaining uncertainty.
Use narrow entry/launch filters and bounded replay. Retain commands, report bytes,
metric units, replay/cache/clock behavior and tool version. Profiling goes through
normal GPU admission and isolation. If counters are unavailable, report that
limit and use PTX/SASS evidence plus controlled comparisons; do not relax host
permissions or install/reconfigure shared services as an inferred next step.

Distinguish insufficient grid parallelism, dependency/issue latency, memory
throughput, excess traffic, shared bank conflicts, barriers, spills and compute
throughput using the evidence ladder in the reference. Low occupancy, a stall
percentage, a register increase, or low MFU alone does not establish causation.
Map hot SASS/instruction patterns back to lowering and typed IR when possible;
label correlation and hypotheses separately from demonstrated causes.

Where available, compare against (1) straightforward reference code preserving
the exact launch/ABI/numerics and (2) an optimized implementation as an explicitly
different scheduling/algorithm ceiling. Keep reference code isolated from compiler
lowering; never delegate our IR-to-PTX translation to another compiler. References
must pass the same outputs/tolerances. Do not present a library speedup using
reduced precision or a changed launch as an achievable compiler-only improvement.

## Deliver the audit

Persist a machine-readable per-case table and concise report in the configured
research store, including source/suite IDs and evidence links. Include latency
and noise; useful FLOPs and special-op counts; useful throughput and defined MFU;
logical and measured traffic/bandwidth; arithmetic intensity; grid/block/active
SM limits; resources; reference latency/ratio when measured; bottleneck evidence,
confidence, alternatives and next discriminating check. Null means unavailable.
Give every case a row, including regressions and inconclusive comparisons.

Rank actionable opportunities by recoverable absolute time and evidence, keeping
schedule-imposed limits separate from compiler defects. Give one next experiment
per unresolved cause, with a predicted instruction/counter/timing change and
bounded cost. Save a resumable handoff with owned live jobs and exact next command.
An audit can finish with a well-supported limitation or null finding; it must not
claim to know exactly what is wrong when the measurements cannot distinguish it.
