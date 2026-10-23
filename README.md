# Emmy compiler workspace

A direct **Kernel IR → PTX** compiler and a local CUDA research system. The
retained compiler is `51bd61bd...`: all 15 published problems pass on both the
configured RTX 4090 (SM8.9) and official RTX 5090 (SM12.0). The compact 122-case
suite and all 26 required sanitizer checks pass on the RTX 4090. See the
[shared-window repair evidence](docs/shared-window-fix.md) and
[current handoff](docs/HANDOFF.md) for identities, scope and resume instructions.
The [RTX 4060 baseline report](docs/baseline-correctness.md) is historical.

The lowering is ours, using Emmy's typed loader and sibling metadata for ABI and
launch contracts. It emits direct PTX; it does not delegate lowering to Emmy's
renderer, nvcc, or another compiler. Correctness takes priority over performance.

The matmul-first tensor-core design—dataflow recognition, warp-atom lowering,
three-term TF32 precision evidence, split-K/atomics, and the attention
follow-on—is recorded in [docs/tensor-core-plan.md](docs/tensor-core-plan.md).
It is a plan, not an implemented feature.

Optimization proceeds concurrently with scope broadening, but only as
generality-preserving typed-IR analysis and emission. Every source change still
requires all 15 published cases to pass before timing, and tensor-core lowering
will be a separate recognizer/emitter increment rather than a rewrite of the
generic path.

```sh
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python compile.py emmy-hackathon-2026/problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

Python 3.12+ is required. The challenge is pinned as a public submodule. The owner
performs future networked Git operations. requirements.lock records the original
development environment; evaluation environments should retain their own lock.
Torch is unnecessary.

## Working with the agent

Codex and Claude Code share the three skills in `.agents/skills/`:
`kernel-strategy`, `kernel-experiment`, and `kernel-reflection`. Claude Code reads
AGENTS.md through CLAUDE.md and reaches the canonical skills through the tracked
`.claude/skills` directory symlink. Machine-specific runtime configuration is
private and ignored.

**Discovery is opt-in.** Paste this into either Codex or Claude Code:

```text
Use kernel-strategy, kernel-experiment, and kernel-reflection to reduce the
current solution's kernel latency while preserving correctness. Resume from
<research-root>/START.md and run discovery with up to three research
agents on distinct ideas. Stop after 30 minutes or ten evaluated variants,
whichever comes first. Keep hosted calibration off. Save findings and a
resumable handoff in the research store.
```

Replace `<research-root>` with your configured durable record directory. Adjust
the metric, parallelism, and stopping condition as needed. Use the same
prompt to resume; the agent checks existing jobs and checkpoints first.

To stop:

```text
Stop discovery, safely checkpoint our active work, and leave a resumable research
handoff. Close the run and leave discovery off.
```

For ordinary bug repair:

```text
Fix [bug/reproducer]. Do not start discovery, sweeps, or hosted calibration.
Run relevant regressions and all 15 published problems before benchmarking.
```

Opening the repository or loading skills starts no campaign or background
service. The skills and AGENTS.md supply the execution and correctness rules;
[the operation guide](docs/research-loop.md) covers manual commands and recovery.

## Correctness before performance

Every future candidate **and comparison baseline** must pass all 15 published
problems at unchanged tolerances under the current evaluation version before its
GPU benchmarking is valid. The research tools bind this gate to frozen source
and evaluator identities. A source change requires a new gate. Partial sweeps
are useful for correctness debugging and cannot qualify performance evidence.
Fresh independent confirmation is additionally required for promotion.

Routine GPU checks use the [compact 122-case regression suite](docs/gpu-regression.md),
including all 15 published contracts. The full archive is retained for targeted
expansion; previous random confirmation draws do not accumulate into every run.

Compiler prototypes are developed as plain snapshots in the evaluation workspace.
Only evaluated, shortlisted compilers return here. No Git runs on a remote GPU
host. Local commits
require an explicit request; all Git remote actions and publication remain yours.
Connection details and runtime configuration stay private and ignored. The shared
skills do not prescribe Docker, NFS, SSH, a GPU model, or a machine path.

[Evaluator contract](docs/evaluator-contract.md) explains the loopback runner,
unchanged upstream client, earlier benign hosted checks, and known limits. Missing
local service never triggers hosted fallback. Hosted calibration has a separate
serial request budget and is not an inner-loop dependency.

Run these commands in your configured evaluation environment, with an exclusive
GPU admission check for GPU work. Set `EMMY_JOB_DIR` to your job artifact directory:

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
