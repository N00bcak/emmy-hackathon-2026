# Baseline compiler and local evaluator

Budget: one 30-minute implementation/evaluation run, at most 10 candidate revisions,
and at most 20 public API requests (including health/schema, excluding bounded busy retries).

Strategy: build a general typed IR -> PTX baseline, then replicate the documented
compile/run interface locally. Keep expected-output scoring independent from execution.

Hypothesis: a metadata-driven baseline and a local CUDA Driver API runner can
reproduce the public harness's correctness verdicts, leaving the public GPU
necessary for target-specific confirmation rather than ordinary development.

Implementation idea: independent compiler and evaluator work; assembler and CPU
checks first, then the idle local GPU, then a few benign public comparisons.
Compiler prototypes stay on Linux until evaluated and shortlisted. No Git on the
GPU host, no local Git network actions, no publication or commits by the agent.

Tests: pointwise and reduction coverage as implemented; all-output tolerance
comparisons; CLI failure on unsupported nodes; argument order; seeded and exact
inputs; repeated atomic-output reset; truncation behavior; repeat compile IDs;
local/public result comparison. Save exact requests/responses and source hashes.
No scans, load tests, crash probes, hidden cases, server filesystem inspection,
or deliberate invalid GPU accesses. Use documented routes and valid small kernels.

Reject/inconclusive: compilation alone is not correctness; local/public timing
numbers across different GPUs cannot establish timing parity; undocumented server
internals and hidden-set evaluation remain unknown. Missing coverage must be explicit.
