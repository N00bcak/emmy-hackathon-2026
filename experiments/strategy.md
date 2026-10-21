# Current strategy

Confirmed target: direct Kernel IR to PTX. Upstream revision remains pinned in
the parent repository. No existing renderer/compiler performs our IR lowering.

Baseline source: compile.py, emitting PTX 8.0/sm_89. All eight MVP cases plus
f01_rmsnorm/f03_l2norm pass local GPU evaluation. Two nonfixture typed IR cases
and a reordered scale/shift ABI pass. Five open cases are explicitly unsupported;
hidden-set correctness remains unverified. See 001-baseline-evaluator for evidence.

Local runner: documented compile/run interface, independent upstream expected
output comparison, per-request GPU-process isolation, output resets and bounded
execution. Sixteen CPU tests pass. A bounded 16-request hosted study confirms the
tested contract, with explicit limits on timing and hidden-evaluation conclusions.
Ordinary evaluation defaults to local loopback and invokes no Git commands.

Next research queue: coupled accumulator states for online softmax; asynchronous
shared transport for matmul; lowering/generalization tests for unseen shapes and
ops. Compare these by coverage gained and information per implementation/eval cost.
Do not infer a runtime bottleneck from emitted PTX length; ptxas may already remove
redundant arithmetic. No optimization or comparative speedup is established yet.

Compiler experiments live on the GPU host as plain source snapshots. Only tested,
shortlisted compilers return to the owner's repository. No Git runs on that host;
local Git network actions and final retention/publication belong to the owner.
The next run states its budget before work (default at most 30 minutes or 10 experiments).
