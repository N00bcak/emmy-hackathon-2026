# Local kernel research

The active Codex or Claude Code session coordinates the research. Native
subagents investigate distinct questions; Python tools prepare and measure
frozen candidates. The tools do not choose hypotheses or run an unattended
agent service.

Discovery is **off by default**. Opening this repository, loading a skill,
resuming records, repairing a compiler bug, or qualifying the infrastructure
does not activate an R&D campaign. Routine repair uses bounded smoke or
qualification runs without activating discovery.

Only after an explicit user request to start discovery:

```sh
python -m tools.research activate-discovery --reason 'Record the explicit user authorization'
python -m tools.research start-run research-001 --mode research
```

Activation records permission; it launches no process. That permission applies
to one research run. Starting another research run needs fresh opt-in. Stop
admitting research work with either:

```sh
python -m tools.research deactivate-discovery --reason 'User requested a stop'
python -m tools.research finish-run research-001 --reason 'Budget reached'
```

These commands do not kill processes or declare unfinished work complete.
Inspect recorded jobs and perform ordinary owned-worker cleanup. Loading the
repository again does not reactivate discovery.

## Mandatory all-15 correctness

Routine GPU checks use the [audited 65-case regression suite](gpu-regression.md):
all 15 published contracts plus 50 core witnesses. The versioned scope ledger
records targeted mechanism/dtype checks and non-gating stress diagnostics.
Use final-output metadata tolerances, not universal intermediate FP exactness. The complete
archive remains available for targeted expansion; do not replay every earlier
random draw by default. New protocols use one fresh draw per selected family,
with a separate post-freeze confirmation draw. Existing frozen suites stay fixed.

The working baseline must pass **all 15 published problems**. Every valid GPU
evaluation or benchmark requires an immutable all-15 correctness receipt for
the exact frozen compiler source and evaluation suite. Focused one-case timing,
parameter sweeps, and hosted calibration do not waive this requirement.

Default suites include all 15 anchors regardless of the claimed research
family. A missing, failed, duplicated, stale-source, or different-suite gate
blocks valid timing, shortlisting, promotion, and export. Performance debt can
never waive a published correctness failure. Later failures revoke an earlier
gate. Historical ten-case qualification records remain historical evidence;
they do not satisfy this rule and must not be relabeled as all-15 results.

Correctness gates do not decide speed. A performance shortlist or promotion
requires a fresh, noise-calibrated paired wall-clock result for a workload where
the proposed effect can exceed the measurement floor, with no resolved
regression in the measured family and fresh isolated confirmation. PTX size,
SASS instruction/memory/issue counts, spills and occupancy are mechanism
diagnostics: record them, but do not accept, reject or classify a speed change
from static counts. Wall-clock latency is the ultimate arbiter. If valid paired
timing is unavailable or unresolved, the speed verdict is unresolved; static
evidence cannot substitute for it. A launch-bound published case can anchor
correctness and regressions without establishing a speed effect; use a larger
typed-IR witness with an independent reference for that claim.

Mechanism activation is separate from the speed verdict and is judged only on
final post-`ptxas` SASS. Tell every worker to re-implement an optimization whose
intended instruction, load, address or live-register change does not survive
`ptxas` on every timing target. If a surviving form cannot be implemented
legally, fail closed and record the mechanism as inactive rather than timing a
PTX-only effect. Activation evidence includes both-target SASS hashes, resource
counts and the exact surviving delta; static counts still cannot award or veto
a timing result.

Correctness-debug work remains allowed so a broken baseline can be repaired.
Without valid candidate and baseline gates, confirmation runs correctness only
and emits no timing claims. Passing all 15 isolated checks, including the atomic
sanitizer check, establishes the candidate gate. Missing published cases leave
the result unqualified. Check an existing gate with:

```sh
python -m tools.research published-gate SOURCE_OBJECT_HASH SUITE_OBJECT_HASH
```

The trusted evaluator issues gates; there is no gate-issuance CLI. A source fix
can receive correctness-only confirmation and replace the broken baseline after
host attestation. Once the same-source/suite gates exist for both arms, a later
bounded comparison may measure performance. Raw low-level diagnostic timings
are explicitly unqualified, not valid benchmarks.

Use your configured execution environment and GPU admission mechanism. The supplied
execution adapter requires Linux, CUDA, and a restricted compiler identity; Docker
is one deployment option, not a requirement. Do not run Git on a remote GPU host,
change prescribed launch metadata, or copy unevaluated compiler prototypes back
to the owner's repository.

## Start and resume

Choose a durable record directory and a separate mutable workspace directory.
Local disk or shared storage can be used; NFS is not required. Set these paths
in your shell or private runtime configuration, for example:

```sh
export EMMY_RESEARCH_ROOT="$PWD/.emmy/research"
export EMMY_RESEARCH_WORK_ROOT="$PWD/.emmy/workspaces"
```

The examples below assume a minimal compiler tree at `./compiler-source`.

The shared skills and tools do not choose a host, storage service, container,
mount layout or GPU model. Keep those choices in ignored private runtime
configuration and pass the resolved paths to the tools. Existing installations
must explicitly configure their current roots before upgrading; do not initialize
a new store because an old machine-specific default was removed.

The supplied compiler sandbox needs Linux effective UID 0 and libseccomp so it
can drop the compiler to a separate restricted UID. A rootless container can
provide that environment, but Docker is not required. On a Mac, strategy and
record tools can run locally; CUDA execution needs a compatible Linux GPU
environment. Do not bypass isolation or treat assembly as execution evidence.

```sh
python -m tools.research init --baseline ./compiler-source
python -m tools.research start-run setup-smoke --mode smoke
python -m tools.research status
python -m tools.research resume
```

`--root` or `EMMY_RESEARCH_ROOT` is required; there is no machine-specific
record-directory default. `--work-root` or `EMMY_RESEARCH_WORK_ROOT` selects the
workspace directory, defaulting to `workspaces/` inside the record root. Put global options
before the subcommand. Initializing an existing store fails; resume it instead.
The baseline argument must be a minimal compiler-source tree (`compile.py` and
its runtime helpers), never the whole repository. Source snapshots reject
evaluation directories, reference NPZ files, and IR fixtures rather than exposing
them to the compiler subprocess.

Runs require an explicit budget before preparation or GPU claims. Smoke and
ordinary research runs stop admitting new work at 30 minutes or ten evaluated
variants. Qualification runs use two hours or 64 variants. Already admitted
candidates may finish their timing work while wall time remains. A budget
never converts incomplete work into a pass.

The store's `START.md` points replacement agents to:

```text
control/state.json            baseline, five coverage gaps, runs, queues, jobs
knowledge/mechanisms/          explanations, conditions, contradictions, evidence
knowledge/decisions/           coordinator decisions and reasons
experiments/<id>/              brief, notes, handoff, manifest, results, samples
objects/<sha256>/              immutable manifest and actual retained files
suites/<version>.json          immutable evaluation contract identity
```

Before restarting work, inspect `active_gpu`, preparation ownership and recorded
external job identifiers. Heartbeat age does not authorize killing a process or
duplicating its work. The coordinator must inspect and reconcile old jobs; the
tools never kill jobs during recovery.

## Propose, implement, freeze

Search both experiments and mechanism records before substantial implementation.
Then write a short JSON brief containing:

```json
{
  "id": "aligned-loads-01",
  "mechanism": "preserve vector global loads",
  "applicability": "aligned contiguous f32 pointwise reads",
  "prediction": "vector load instructions replace scalar loads",
  "probe": "inspect both targets before timing",
  "opportunity": "four scalar reads may become one vector instruction; bytes unchanged",
  "budget": {"variants": 8, "seconds": 600},
  "semantic_overlap_decision": "Explain related experiments and why this question differs"
}
```

```sh
python -m tools.research search 'aligned vector loads'
python -m tools.research propose brief.json
python -m tools.research workspace aligned-loads-01
```

The proposal records related lexical matches. Exact normalized mechanism and
applicability duplicates require `repeat_reason`; recorded contradictions require
`contradiction_response`. The coordinator reviews semantic overlap and rough
cost bounds. A search match is evidence to inspect, not an automatic rejection
of a different idea. Do not turn uncertain cost estimates into novelty vetoes.

`workspace` copies the recorded parent baseline to a new plain source directory.
It never overwrites an existing workspace. Implement there, record observations
and failures in the experiment's `notes.md`, then freeze the source:

```sh
python -m tools.research freeze aligned-loads-01 "$EMMY_RESEARCH_WORK_ROOT/aligned-loads-01"
python -m tools.research enqueue aligned-loads-01 --hypothesis aligned-loads
```

Revised source requires a new experiment. Frozen source contains actual file
copies, not only hashes. Snapshots reject symlinks and omit Git, private skills,
credential paths, caches, and candidate inboxes. A snapshot requires root
`compile.py`; the evaluator invokes only frozen source. The store root is mode
700 and candidate subprocesses run under another UID. Same-owner arbitrary code
is not an adversarial security boundary.

Bound each initial hypothesis to 16 variants. Sweep only internal choices that
preserve ABI, launch geometry, shared-memory limits, and IR semantics. Retain
unconventional but falsifiable ideas; agents may build small probes or abandon
unpromising ideas without implementing a full compiler change.

## Evaluation and the queue API

`tools.research_store.Store` is the coordinator integration API. It performs no
CUDA work, Git, network access, or compiler execution. Execution adapters supply
the following operations:

| Operation | Contract |
| --- | --- |
| `claim_preparation(owner)` | At most two active preparation claims; hypotheses interleaved |
| `finish_preparation(queue_id, owner, artifact_id, gate)` | Verify retained preparation artifact; `pass` enters GPU-ready work, `inactive`/`invalid`/`failed` screen it out |
| `claim_gpu(owner)` | One recorded GPU job; returns mode, member IDs, and burst identity |
| `gpu_job_ids(job_id, owner, external_ids)` | Persist real worker/runtime job identities before yielding |
| `finish_gpu(job_id, owner, outcomes)` | Caller first synchronizes or retires its worker; record an outcome for every batch member |
| `record_pair_block(...)` | Persist paired samples with burst identity, completeness, observations, and immutable block artifact |
| `issue_evidence(experiment_id, report, artifacts)` | Trusted evaluator issues immutable evidence; no equivalent CLI command exists |

The bounded ready/preparation queue holds 16 variants. Correctness batches hold
at most eight. Switch to exclusive timing when four survivors wait or the oldest
has waited 30 seconds. Give correctness a turn after four timing candidates or
approximately 60 seconds. Finish an already-started paired block before yielding;
preserve the candidate's sample state and burst boundaries. Incomplete blocks
remain diagnostic artifacts and do not count toward paired comparison evidence.

Concurrent correctness is screening only. Buffers require independent guards,
input-integrity checks, and complete output comparisons. Any discrepancy
quarantines the batch. If concurrent failure disappears in isolation, recheck
every batch-mate alone. Every eventual promotion requires fresh isolated
confirmation. Atomic or unfamiliar memory/synchronization work starts isolated;
the atomic matmul coverage item always requires Compute Sanitizer evidence.

Measure baseline-versus-baseline noise by case and size regime. Stop clear losers
early; spend more samples on promising or uncertain comparisons. Start with two
eight-pair blocks and cap complete samples at 200 repetitions per case. Fresh
confirmation is mandatory for sweep winners. Record compilation, assembly,
kernel, and total evaluation costs separately. Report valid variants per hour,
queue starvation and idle causes; generating filler work is not a success.

## Independent references, decisions, and exports

The coordinator owns generator/reference construction. A separate agent validates
the mathematics, schedule legality, fixed tolerances, and tests before coverage
compiler development starts. Softmax references use ordinary stable softmax;
attention references use score matrices, ordinary stable softmax, then value
contraction. They must not repeat the online lowering algorithm.

Keep published tolerances unchanged. Generated matmul uses `rtol=0.001,
atol=0.001`; softmax uses `0.001, 0.00001`; attention uses `0.001, 0.0001`.
The standard generated matrix/attention dimensions are drawn from 1, 3, 7, 16,
31, 32 and 33; cooperative softmax additionally tests 127 and 257 columns. Larger
custom shapes require independent numerical-policy validation before use in a
scoring suite. Validate policies for declared shapes and input ranges. Any tolerance,
reference, or generator-policy correction creates a new suite version and
requires baseline/candidate reevaluation.

```sh
python -m tools.research suite cases-v1 evaluation-contract.json
python -m tools.research reserve-confirmation aligned-loads-01 SUITE_OBJECT_HASH
```

Suite registration retains the contract file itself. The execution adapter also
retains generator/reference source and all exact case inputs. Reserve fresh
confirmation only after source freeze. Repeating the reservation returns the
same seed and identity, including after interruption; do not redraw a difficult
case. Freeze a new experiment to fix source after seeing confirmation failures.

For the supplied execution adapter, use `python -m tools.research_pipeline
suite` to register the protocol with the actual evaluator/generator Python
sources. Changed evaluator source requires a new version. Its `cases` command
returns both the case list and its retained manifest path; `prepare` accepts that
manifest and compiles both candidate and baseline using the CPU workers.

Confirmation has three separate commands:

1. Run `python -m tools.research_pipeline prepare-confirmation EXPERIMENT
   --suite-id SUITE_OBJECT_HASH` in a CPU job. This reserves fresh cases and
   prepares both compiler arms before taking GPU admission.
2. Run `python -m tools.research_pipeline confirm EXPERIMENT --suite-id
   SUITE_OBJECT_HASH` through the configured GPU admission launcher. It checkpoints
   completed isolated cases and can resume safely within another bounded job.
3. After the GPU wrapper exits, run `python -m tools.research_pipeline
   attest-confirmation EXPERIMENT` in a CPU job. This verifies the host's job,
   UUID, periodic telemetry, process ownership and post-job observations, retains
   all audit bytes, and issues a new immutable receipt.

The runtime launcher sets `EMMY_JOB_DIR` to its per-job artifact directory and
provides the attestation evidence: `gpu-attestation.json`, `monitor-result.json`,
`telemetry.jsonl`, admission/before/after GPU and process CSVs, `command.txt`, and
`environment.json` describing the runtime and dependencies. Legacy container
launchers may supply `container.json` instead. Device identity, process ownership,
contention checks and timeouts apply equally to native and container execution;
the adapter must not invent observations when monitoring is unavailable.

The initial confirmation receipt is provisional and cannot support promotion.
Missing telemetry, detected foreign work, or a failed post-job observation does
not become a clean timing result. These observations are not a reservation or
proof that no unsampled transient interference occurred.

The evaluator receipt must name `source_id`, `baseline_id`, `suite_id`, and
`confirmation_id`, retain actual result artifacts, and report passed isolated
cases with guard/input integrity checks. It is not a coordinator-supplied
`passed=true`. CLI decisions verify the receipt and its objects:

```sh
python -m tools.research decide aligned-loads-01 shortlist --evidence-id EVIDENCE_OBJECT_HASH --reason 'Explain evidence and remaining uncertainty'
python -m tools.research export aligned-loads-01 --inbox ./candidates
```

Only shortlisted/promoted evaluated candidates export, always to a new
`candidates/<experiment>/` directory with source and evidence. Nothing overwrites
the owner's `compile.py`. The owner chooses what to retain and publish. A
candidate developed against a stale baseline must be integrated and retested
before shortlisting or promotion.

The coverage backlog tracks all five items: `m01_matmul_512`,
`m02_matmul_rect`, `f02_row_softmax`, `a01_attention`, and `a02_attention_128`.
Read current status and evidence from the research store rather than assuming these stay open.
Two coverage responsibilities remain until all five close with supplied cases,
independent generated variants, and regressions. Finishing one assignment means
taking the next unresolved gap. Coverage success is the campaign goal, not a
prerequisite for accepting the infrastructure.

Use `mechanism` to append structured evidence-backed findings, and `handoff` to
record dependencies, active job IDs, source identity, and the next executable
action. Keep subagent returns compact and link detailed evidence. Rejected ideas
and contradictory observations are part of durable research memory.

Hosted calibration is a separately budgeted activity: default 17 planned requests
below a hard 25-request cap, one batch per run, serialized and spaced by at least
two seconds. Stop on busy, rate limiting, or service errors. Evidence accumulates
by mechanism and case regime; a local speedup alone does not establish a 5090
winner. Control-store commands never contact the endpoint.

Local calibration medians are derived from the matching immutable paired block.
The accepted receipt binds exact case files and each compiler arm’s PTX hashes;
caller-supplied numbers cannot replace measurements or swap arms. Qualification
and CLI drivers are recorded with their jobs; the local suite freezes its actual
scoring, preparation, CUDA worker, generator, store and published-policy sources.
