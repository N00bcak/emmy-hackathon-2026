# Accounting and interpretation

## Units and scope

Let `t = latency_us * 1e-6`, `F` be useful floating-point operations, `H` measured
executed FLOPs, `B_logical` logical bytes, and `B_dram` measured DRAM bytes.
Use decimal GB/s and TFLOP/s; label KiB/MiB when using binary byte counts.

| Quantity | Calculation | Interpretation |
|---|---|---|
| Useful FLOP/s | `F / t` | Useful algorithmic work per second |
| Executed FLOP/s | `H / t` | Hardware work, only when counted with correct predication/instruction semantics |
| MFU | `100 * F / (t * matching_peak_flops_s)` | Useful compute utilization; name the work convention and precision/mode |
| Hardware compute utilization | `100 * H / (t * matching_peak_flops_s)` | Distinct from MFU and profiler SM busy/throughput percentages |
| Effective bandwidth | `B_logical / t` | Algorithmic byte-accounting convention, not observed bus traffic |
| DRAM bandwidth | `B_dram / t` | Actual measured DRAM traffic; disclose profiler/cache-regime differences |
| L2 bandwidth | `B_l2 / t` | Actual measured L2 traffic; compare only to an L2 ceiling |
| Arithmetic intensity | `F / B` | Specify logical, DRAM or L2 bytes; do not mix memory levels |
| Optimistic resource time | `max(F / P, B / BW)` | Roofline lower-bound estimate with matching work/traffic/ceiling definitions |

For kernel audits MFU means the useful-work convention above; it is not
automatically the training MFU convention for a whole model. Integer/address,
SFU-dominated, and mixed-precision workloads need separate resource accounting.
MFU is unavailable when the FLOP count or matching ceiling is not known. Never
use sparse tensor marketing FLOPs as the denominator for dense FP32 SIMT work.
A hypothetical tensor implementation is a separately labelled opportunity.
Do not clamp utilization above 100%; investigate accounting, clocks, cache regime,
peak provenance and measurement uncertainty. Do not combine different dtype
FLOPs under one ceiling without an explicit resource model.

## Counting work

- GEMM: conventional `2*M*N*K` useful FLOPs, or exact `M*N*(2*K-1)` for unseeded
  mathematical dot products; state the convention. Include batching and epilogue
  work separately. Split-K/atomic accumulation adds executed work and traffic.
- Attention: count `Q*K^T` and `P*V` using their respective dimensions, batch/head
  counts and actual mask domain. Count scaling/softmax elementwise arithmetic and
  reductions separately. Exponentials and divisions are separate operation counts.
  Do not infer attention FLOPs from sequence length without inspecting the IR.
- Sum: `N-1` useful additions for `N>0`; runtime seed additions, padded lanes and
  shuffle trees belong to executed work. Mean also has normalization. RMS/L2 norm
  includes squares, reductions and per-element normalization according to the IR.
- Pointwise: count floating arithmetic per actually valid output. Loads/stores,
  indexing, comparisons and casts are not FLOPs. FMA is two, multiply is one.
- Counts may be analytic estimates. Retain the derivation and distinguish domain
  bounds, conventional FLOPs, exact useful FLOPs, static instruction counts and
  measured executed operations. SASS opcode counts without dynamic execution
  counts cannot establish executed FLOPs or memory traffic.

Logical traffic is normally required distinct input reads plus final output
writes, explicitly stating ideal reuse. For GEMM this lower bound is
`element_bytes*(M*K + K*N + M*N)`, with old-C reads if required. It omits inter-CTA
reloads, cache sectors, spills, atomic read/modify/write and intermediates. Give a
second schedule-aware estimate if known. Fusion can avoid materializing a full
score matrix; do not count an intermediate as compulsory DRAM traffic unless it
is actually materialized. Pipeline bytes/time uses the pipeline time, not one
constituent kernel's latency. Do not average per-kernel MFUs or speedup ratios to
manufacture a pipeline score.

## Measurement checks

Separate bare kernel intervals, ordered GPU pipeline intervals, host launch/API
time and end-to-end time. Timer/control overhead is empirical and protocol-specific.
For microsecond workloads, quantization and jitter may dominate arithmetic costs.
Retain the actual sample resolution and A/A distribution; extra decimal places
do not increase precision. Capture/replay and clock/cache policies must match
between comparisons. Never promote an apparent gain smaller than the accepted
case-specific noise band.

Counters may require multiple profiler replays with different cache or clock
behavior. Use unprofiled timings for latency claims. Combining replay byte counts
with unprofiled latency yields a cross-run bandwidth estimate; identify it as
such and verify the cache/input regime is comparable. If not comparable, keep
profiler traffic and profiler duration together as diagnostic data. Do not
reinterpret instrumented wall/event time as normal execution time.

An empty-kernel control gives context, not a universally subtractable constant.
If a calibrated fixed-cost floor is used in a model, report its provenance and
uncertainty. The compute/memory roofline is optimistic: it omits SFU, issue,
dependencies, synchronization, occupancy and fixed costs. A large gap proposes
questions; it does not identify a defect or guarantee an attainable speedup.

## Evidence ladder

| Possible limit | Supporting evidence | Useful next discriminator |
|---|---|---|
| Too little parallelism | Few CTAs/warps relative to actual SM count, low achieved active SM coverage | Same-launch reference; inspect per-CTA serial work. Grid/SM is an upper bound on simultaneous SM coverage, not occupancy. |
| Memory throughput | High measured bandwidth at the relevant level, traffic and coalescing measurements | Compare bytes with useful/schedule estimates; test reuse/alignment under unchanged semantics |
| Excess memory traffic | Measured sectors/bytes exceed a justified estimate, spills or repeated loads in SASS | Attribute to coalescing, cross-CTA reloads, spills or compiler redundancy |
| Memory latency | Low bandwidth plus dependency stalls, too few eligible warps or insufficient outstanding accesses | Inspect dependency chains and independent loads; bandwidth alone cannot establish this |
| Compute throughput | Relevant pipeline near its applicable ceiling, sufficient issue/parallelism | Count executed versus useful work and inspect redundant instructions |
| Integer/index overhead | Dynamic integer instructions and issue pressure, repeated address expressions in SASS | A legal address simplification must change executed code and pass paired confirmation |
| Dependencies/SFU | Long dependency chains, relevant instruction/counter attribution | Reschedule or remove redundant operations only when numerically legal |
| Synchronization/shared memory | Barrier imbalance, shared conflicts/traffic, matching control-flow evidence | Inspect collective participation and producers/consumers before changing barriers |
| Register/spill pressure | Local-memory traffic and resource-limited residency, not register count alone | Legal lifetime change with measured traffic/occupancy and latency comparison |
| Measurement floor | Tiny work, quantized samples, broad A/A noise, comparable minimal-control duration | Validate a better timing protocol before interpreting small differences |

A demonstrated compiler cause requires a correct, independently confirmed
controlled change with the predicted generated-code/counter effect. With no such
change, say suspected/supported rather than proven. Profiled stall percentages
are conditional statistics, not directly additive fractions of elapsed time.
Do not equate a high occupancy target with maximum performance.

## Calculator input

`scripts/metrics.py` accepts an object with `cases`, each containing:

```json
{
  "cases": [{
    "case": "example",
    "latency_us": 25.6,
    "useful_flops": 268435456,
    "logical_bytes": 3145728,
    "work_mode": "fp32_simt",
    "compute_ceiling": {
      "flops_s": 80000000000000,
      "mode": "fp32_simt",
      "basis": "theoretical",
      "source": "EXAMPLE ONLY: supply actual device evidence"
    }
  }]
}
```

The numbers illustrate units, not a certified device ceiling. Optional fields:
`executed_flops`, `dram_bytes`, `l2_bytes`, `dram_ceiling_bytes_s` and
`l2_ceiling_bytes_s`. Each memory ceiling is an object with `bytes_s`, `basis`
(`theoretical`/`measured`) and `source`. All work/byte counts must be nonnegative;
latency and ceilings must be finite positive numbers. Omit unknown fields rather
than entering zero. Unknown/mismatched compute mode yields null MFU with a note.
Input provenance, correctness, scope, noise and cache regime remain the caller's
responsibility. Input rows are retained in the output. Above-peak values remain
visible. Roofline attribution and confident bottleneck labels are deliberately
not generated by the arithmetic helper.

## Primary references

- [NVIDIA CUDA Best Practices: timing and effective bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Compute Profiling Guide: rooflines, replay and measurement limitations](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Nsight Compute CLI: installed version's sections, metrics and filters](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

Consult the installed tool's help and the corresponding documentation version
before selecting metric names; availability and collection behavior vary.
