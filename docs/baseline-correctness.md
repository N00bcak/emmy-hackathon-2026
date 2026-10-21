# Correctness-first baseline

The compiler with SHA-256
`d35da050e41fd3a172e5611b69a48ccabecb2f1d359aa9a4204e5e2685c9bcbf`
passes all 15 published problems on the local RTX 4090 at their original metadata
tolerances. [Per-case results](baseline-correctness.json) record the complete
output counts and maximum observed absolute errors.

Each case used a fresh isolated CUDA worker with guarded buffers, input-integrity
checks, and complete output comparisons. Fourteen cases ran three repetitions;
atomic matmul ran ten, with its prescribed output reset before every launch.
Atomic matmul additionally passed ten repetitions under Compute Sanitizer
memcheck: exit zero, zero reported errors. Both sm_89 and sm_120 assembly and
disassembly passed for all 15. GPU execution was sm_89 only. A separate sweep using the unchanged upstream
submission client also passed all 15 plus a reordered scale/shift ABI check.

## What changed

The emitter walks typed IR rather than recognizing fixture names. It now handles
multidimensional Tile coordinates, explicit carried state, non-seeding inner
loops, and reduction updates whose `base` names replace the usual accumulator
operand. Stable registers preserve carried values across repeated loop bodies.
Online softmax and attention lower the IR's coupled maximum, denominator, and
weighted-value updates; `exp` uses scaled PTX `ex2.approx`.

Shared copies use a conservative synchronous load/store implementation while
retaining the IR's block barriers. Commit/wait markers need no asynchronous
operation under that lowering. Shared memory comes from typed extents and
alignment. When metadata provides the dynamic allocation, one external pool
avoids allocating the same storage twice. This permits the 73,728-byte matmul
scratch allocation. No renderer, nvcc, or existing compiler lowers the IR.

Descriptive windows are accepted; unsupported bounded windows and unproven raster
layouts fail explicitly. Shuffle participation uses the active mask. Scalarized
vector accesses, synchronous transport, and simple code emission favor a readable
correct baseline rather than high throughput. Multi-kernel graphs select entries
using their filename and metadata, preserving the host's prescribed stage order.

## Evidence and limits

The full source, exact inputs/references, IR/meta, both-target PTX/cubin/SASS,
commands, tool versions, admission telemetry, and per-launch checks are retained
on NFS. Paths in the JSON summary are relative to the private artifact root.
Earlier failed compiler versions and diagnostics remain there too.

Published correctness is the minimum gate for future evaluation. It does not
prove hidden-set correctness, general support for every legal schedule, absence
of every race, or RTX 5090 performance. No speedup or discovery campaign was part
of this baseline repair. Research promotion additionally requires frozen-version,
fresh generated cases and source-bound evidence through the research tools.

The final infrastructure qualification passed all-15 correctness and atomic
memcheck before timing, then completed 64 replayed baseline controls, width
comparisons, guard/quarantine controls and paired timing controls. Width two was
selected. An earlier oversized batch request is retained as an incomplete harness
run; its batching was repaired and regression-tested. These controls establish
infrastructure behavior, not 64 novel compilers or a speedup.

The final frozen research suite also passed fresh isolated confirmation on
**15 published + 32 generated cases** (four per generated family), with host
telemetry checked afterward. All five backlog entries are closed for this declared
suite. Generated attention is a three-kernel composition alongside the supplied
fused anchors; arbitrary hidden fused schedules remain unverified. The baseline
advanced for correctness with explicit performance debt. All 132 CPU tests passed.

Setup runs are closed and discovery remains off. The JSON summary links the exact
source, suite, correctness gate, confirmation receipt and qualification receipt.
