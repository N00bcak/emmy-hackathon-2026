# Emmy compiler workspace

A direct **Kernel IR → PTX** compiler and a local CUDA research system. The
correctness baseline covers all 15 published problems; verification details and
limits are recorded in [the baseline report](docs/baseline-correctness.md).
Hidden-set correctness and RTX 5090 performance remain unverified.

The lowering is ours, using Emmy's typed loader and sibling metadata for ABI and
launch contracts. It emits PTX targeting sm_89; it does not delegate lowering to
Emmy's renderer, nvcc, or another compiler. The baseline prioritizes correctness.
No comparative speedup is claimed.

```sh
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python compile.py emmy-hackathon-2026/problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

Python 3.12+ is required. The challenge is pinned as a public submodule. The owner
performs future networked Git operations. The Mac's requirements.lock records its
installed environment; the private Linux runtime has its own dependency lock.
Torch is unnecessary.

## Working with the agent

Codex and Claude Code share the three skills in `.agents/skills/`:
`kernel-strategy`, `kernel-experiment`, and `kernel-reflection`. Claude Code reads
AGENTS.md through CLAUDE.md and reaches the canonical skills through the tracked
`.claude/skills` directory symlink. The private runtime skill is ignored.

**Discovery starts only when you request it.** For example, tell either agent:

> Start a local discovery run using kernel-strategy, kernel-experiment and
> kernel-reflection. Resume from NFS, verify the all-15 correctness gates, use
> up to three research subagents, and stop after 30 minutes or ten variants.
> Keep hosted calibration off for this run.

The coordinator inspects state, assigns distinct questions, and invokes bounded
Python workers. Opening the repo, loading a skill, or running tests starts no
campaign or background agent service. To do other work, simply ask for it:

> Repair this compiler bug. Do not start discovery, sweeps, or hosted calibration.
> Run the relevant tests and the all-15 correctness suite before benchmarking.

You can also request one named skill stage without authorizing a full campaign.
[Research operation and recovery](docs/research-loop.md) gives the actual CLI,
queue, confirmation, persistence, and activation instructions. Pausing/ending an
agent session leaves durable records; it does not leave agents researching in the
background. Inspect recorded live GPU jobs before resuming interrupted work.

## Correctness before performance

Every future candidate **and comparison baseline** must pass all 15 published
problems at unchanged tolerances under the current evaluation version before its
GPU benchmarking is valid. The research tools bind this gate to frozen source
and evaluator identities. A source change requires a new gate. Partial sweeps
are useful for correctness debugging and cannot qualify performance evidence.
Fresh independent confirmation is additionally required for promotion.

Compiler prototypes are developed as plain snapshots on Linux. Only evaluated,
shortlisted compilers return here. No Git runs on the GPU host. Local commits
require an explicit request; all Git remote actions and publication remain yours.
Connection details and runtime configuration stay private and ignored.

[Evaluator contract](docs/evaluator-contract.md) explains the loopback runner,
unchanged upstream client, earlier benign hosted checks, and known limits. Missing
local service never triggers hosted fallback. Hosted calibration has a separate
serial request budget and is not an inner-loop dependency.

All Linux workloads run inside the configured container. Invoke GPU commands
through its private host admission wrapper. These are container commands:

```sh
python -m unittest discover -s tests -v
python tools/check_local.py --compiler compile.py --out "$EMMY_JOB_DIR/published-check"
```

The second command checks all 15 with the unchanged upstream client. Add
`--timing` only to request indicative timings after the complete correctness gate.
The local runner is a trusted research tool; it does not sandbox hostile kernels
or establish parity with hosted hardware.

[Private repository publication](docs/version-control.md) describes the owner-run
GitHub steps. This implementation makes a local commit only; it does not push.
