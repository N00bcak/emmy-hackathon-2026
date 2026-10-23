# Compute capability and portable submissions

RTX 4090 uses compute capability 8.9 (`sm_89`); RTX 5090 uses compute
capability 12.0 (`sm_120`). Query the actual execution device and capability
when available. A GPU family name is insufficient: Blackwell documentation for
compute capability 10.x does not establish limits or instruction support for 12.0.

Keep four identities separate: execution GPU/capability; emitted PTX version and
`.target`; assembler version and target flags; resulting cubin architecture.
PTX targeting an older supported capability may be assembled for a newer GPU.
Changing `.target` alone does not repair incorrect address or synchronization
semantics. Assembly for `sm_120` establishes acceptance, not execution correctness.

## Lowering and validation

Bind correctness receipts to source, evaluator, case/input/reference, ABI, launch,
execution architecture and toolchain identities. An all-15 pass on `sm_89` does
not establish an all-15 pass on `sm_120`. Keep unsupported and unverified distinct
from passed; preserve any required sanitizer and isolated atomic/reset checks.

For architecture-sensitive changes, inspect both-target PTX/SASS and resources.
Check shared-state versus generic addresses, symbol-relative offsets, address
width and alignment, asynchronous-copy visibility and waits, collective
participation, instruction precision, and launch resource limits. Query resource
limits and dynamic shared-memory opt-in requirements for the actual device;
never substitute another architecture's limits. Direct execution on the target
GPU is required to claim target execution correctness.

The m01 bug provides a counterexample: dropping the dynamic shared slab's
shared-window base happened to work when that base was zero. The tested assembler
placed it at `0x400` on `sm_120`; that value is an allocation observation, not an
architecture constant. Preserve the symbolic shared-window address. A static
shared prefix exposed the same defect on `sm_89`, but that diagnostic does not
replace execution on `sm_120`.

When targets disagree, compare exact source/IR/metadata and prior branch fixes,
then distinguish address, synchronization, numerical and harness hypotheses with
a small counterexample. Retain failures and the architecture/toolchain conditions
that exposed them. Never specialize compiler output to fixture names or values.

## Adapt each authorized submission

Read the pinned runner contract and use the budgeted request helper. Before
sending kernels, preflight the exact frozen compiler/PTX, local correctness
receipt and evaluated shortlist, metadata argument order and scalar types,
ordered pipeline intermediates, grid/block, dynamic shared bytes, per-launch
zeroing, and complete returned outputs at unchanged tolerances. Check endpoint
health within the same request budget and record its actual execution GPU and
advertised target; stop on an unexpected device, unsupported contract, busy,
rate-limit, transport or service failure. Do not silently substitute a cubin,
change launch/precision/tolerances, or retry a failed batch.

Record target execution results separately from local receipts. A correctness
repair can be submitted as a validated correctness shortlist without claiming a
speed improvement. Official published-case success is not hidden-set success.
These instructions do not authorize hosted calls, a discovery campaign, or a
larger request budget by themselves.

## Performance and reflection

Keep per-case latency, regressions and rankings specific to the execution GPU,
toolchain and measurement regime. Revalidate on the destination architecture;
local rankings need not transfer. MFU uses that device's matching dtype,
SIMT/tensor and dense/sparse ceiling with stated provenance. Do not reuse 4090
ceilings or timings to report 5090 MFU or speedups. Label unavailable ceilings
and unmeasured target results explicitly.

References: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
(shared-state addresses, `mov`, `cvta`, `cp.async`) and
[Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
(check the capability-specific sections).
