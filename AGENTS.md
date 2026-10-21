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

## Local research loop

Keep shared research policy separate from machine configuration. Research roots,
storage backends, SSH details, GPU selection, mounts, container settings and
launch commands belong in ignored private runtime configuration. Resolve the
user's configured environment; never assume this owner's deployment. The supplied
restricted evaluator requires Linux/CUDA, but neither Docker nor NFS is required.

Discovery is opt-in. Start or resume a research campaign only when the user
explicitly requests it. Routine implementation, bug repair, code review, and
correctness debugging do not start agents, sweeps, calibration, or a discovery
run automatically. Reading a skill or opening the repository starts no service.
When discovery is requested, read [docs/research-loop.md](docs/research-loop.md)
and use the three shared kernel skills. Codex discovers `.agents/skills`; Claude Code discovers the same
files through `.claude/skills` and `CLAUDE.md`. Keep one canonical copy.

The active session coordinates up to three native subagents. Check actual
capacity, give distinct questions and isolated workspaces, and return compact
findings with durable evidence links. Use two coverage responsibilities and one speed
responsibility until ALL five backlog items close: m01_matmul_512,
m02_matmul_rect, f02_row_softmax, a01_attention, a02_attention_128. Completing one
assignment means taking the next unresolved item. Blocked workers may explore
speed temporarily without dropping coverage responsibility.

Research freedom matters: agents may build probes, redesign lowering, combine
related mechanisms, pivot or reject an idea. Search prior mechanism/experiment
records before implementation. State the applicable semantics, prediction,
cheap discriminating test, opportunity and budget. Address prior contradictions;
do not manufacture a candidate to satisfy a workflow. Rough cost models guide
priorities; uncertain estimates do not veto unconventional ideas.

The coordinator owns generators/oracles; independently validate and freeze them
before coverage research. Compiler agents must not edit their own evaluator.
Use fresh typed IR variants and independent plain NumPy references. Fix numerical
tolerances per version; changing a generator/tolerance creates a new evaluation
version and requires reevaluating both baseline and candidate.

CPU preparation uses two workers, cached cases/references and bounded sweeps
(default 16 variants/hypothesis, ready queue 16). Preserve metadata launch/ABI.
Static activation checks precede GPU timing. Resource increases are warnings.
GPU jobs use host admission and our shared lock. Concurrent correctness is only
screening; guard every buffer and quarantine/recheck all implicated batchmates.
Every promotion requires fresh isolated confirmation; m02 atomics require an
isolated first check, resets and Compute Sanitizer. Timing is exclusive, paired,
case-specific in its noise calibration, block-checkpointed and independently
confirmed after selecting sweep winners. Never infer improvement from one smoke
latency or hide regressions in an aggregate.

Persist source, exact cases/inputs/references, IR/PTX/cubin/SASS, commands,
environment, admission, measurements, failures and resumable handoffs under the
configured durable research root. Conversation memory is supplementary. Verify
identities and inspect recorded live jobs before restarting; never redraw an
inconvenient confirmation batch or kill by heartbeat age alone.

Use explicit bounded runs. Default research/smoke budget is 30 minutes or ten
variants; infrastructure throughput qualification is at most two hours or 64
prepared replay variants. Infrastructure acceptance is separate from campaign
success. Native agents provide research direction; this setup is not an
unattended provider-specific agent service. No paid compute or publication.

Hosted calibration is authorized only for evaluated shortlists through the
budgeted calibration helper: at most one batch/run, 25 total HTTP attempts,
17 planned, serial and >=2 seconds apart, stop on busy/rate-limit/service error.
Otherwise keep evaluation local; never fall back to the endpoint. Record 4090
and 5090 evidence separately. Keep unsupported/unverified distinct from passed.

Every candidate and baseline must pass ALL 15 published problems at their exact
metadata tolerances before its GPU performance evaluation or benchmarking is
valid. Bind this gate to actual frozen source and evaluator identities; a source
change invalidates it. Partial sweeps and GPU correctness debugging can diagnose
failures but cannot authorize timing or promotion. This requirement applies even
to speed work on a single family and cannot be waived as performance debt.

Never hardcode fixture names/values/shapes as solutions. General specialization
from typed IR is allowed. Before reporting completion, run relevant regressions
and state actual coverage, performance evidence and limits. Closing all five
gaps is the campaign goal; it is not a prerequisite for infrastructure acceptance.

## Git and compiler transfer boundaries

All Git operations belong on the owner's local machine. Never run Git commands
on a remote GPU host, including inside its containers. Remote workspaces are
plain source snapshots and experiment directories, not Git checkouts.

Do not run Git network or remote-management actions from the local machine:
no clone, fetch, pull, push, remote changes, or networked submodule operations.
The owner chooses what to retain and performs repository publication. Do not
commit or discard candidates without the owner's instruction. An explicit
request to commit authorizes a local reviewed commit only; publication remains
owner-controlled. Local read-only Git checks and requested setup are permitted.

Develop compiler prototypes in the configured evaluation workspace. Copy only locally GPU-evaluated,
shortlisted compiler candidates back into the owner's repository, with their
benchmark/correctness evidence and limitations. Keep rejected prototypes and
measurement artifacts in the configured research store. Transfer files without Git.
