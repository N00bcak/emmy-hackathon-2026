# Kernel IR compiler workspace

Build a general compiler from Emmy Kernel IR to PTX. Correctness on unseen IRs
and efficient GPU execution are both required. Never trade correctness for speed.

## Read before work

Read `emmy-hackathon-2026/README.md`, `TASK.md`, `MVP.md`, and `docs/API.md`
inside that submodule before compiler or evaluation work. Re-read after updating
the pinned submodule. Its rules and metadata define the challenge contract.
The confirmed target is direct PTX, not CUDA C++. The lowering must be ours;
never delegate it to Emmy's renderer, `emmy compile`, nvcc, or another compiler.
Using Emmy's typed deserialization is required and allowed.

Keep challenge fixtures and harness unchanged in the submodule. Put our compiler
at root `compile.py`; stdout contains PTX only, diagnostics go to stderr.
Read the sibling `meta.json`: entry names, positional arg_order, launch geometry,
shared memory, zero_outputs, tolerances, and ordered multi-kernel execution are
contracts. Do not assume inputs followed by outputs is the ABI.

## Automated discovery loop

Use the repository skills below; these are ordinary Markdown instructions with
no dependency on an agent vendor. Codex discovers `.agents/skills`; Claude Code
discovers the same files via the `.claude/skills` directory symlink and `CLAUDE.md`.

1. Read `.agents/skills/kernel-strategy/SKILL.md` to select the next strategy.
2. Read `.agents/skills/kernel-experiment/SKILL.md` to turn one hypothesis into
   an implementation and measured experiment.
3. Read `.agents/skills/kernel-reflection/SKILL.md` to decide, retain evidence,
   and select the next hypothesis. Repeat within the authorized run budget.

At run start state a wall-time or experiment budget and concrete stop condition.
If the user supplies neither, use at most 30 minutes or 10 experiments, whichever
comes first, and report unfinished work. A budget limits execution, not evidence
standards. Do not spawn workers or saturate the shared GPU runner by default.
Use sequential remote evaluation, bounded retries, and timeouts. No paid compute,
public submission, publishing, or dependency upgrades without authorization.

Establish correctness across all 8 pointwise/reduction MVP problems first, then
extend coverage to matmul, fused, and attention. The challenge ranks hidden-set
correctness, code review, then speed. Within correct coverage, pursue measured
latency improvements and compiler efficiency; compile in seconds per kernel.
Never hardcode fixture names, expected values, or open-set shapes as solutions.
General static-shape specialization derived from the input IR is appropriate.

Before finishing, run relevant regression cases and report actual pass/fail/
unverified counts, category coverage, performance evidence and its limitations,
and the next hypothesis. Missing GPU/network access is unverified, never a pass.

## Git and compiler transfer boundaries

All Git operations belong on the owner's local machine. Never run Git commands
on a remote GPU host, including inside its containers. Remote workspaces are
plain source snapshots and experiment directories, not Git checkouts.

Do not run Git network or remote-management actions from the local machine:
no clone, fetch, pull, push, remote changes, or networked submodule operations.
The owner chooses what to retain and performs repository publication. Do not
commit or discard candidates on the owner's behalf. Local read-only Git checks
and requested repository setup are permitted.

Develop compiler prototypes on the GPU host. Copy only locally GPU-evaluated,
shortlisted compiler candidates back into the owner's repository, with their
benchmark/correctness evidence and limitations. Keep rejected prototypes and
measurement artifacts on the GPU host/NFS. Transfer files without Git.
