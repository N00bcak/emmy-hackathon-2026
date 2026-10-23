# Offline audit handoff

Audit completed against the retained source-bound receipts; remote profiling and
durable-store transfer remain blocked by SSH hostname resolution in this side
chat. No escalation was requested, no owned remote job was started, no active
campaign/worker was touched, and compiler/evaluator/cases remain unchanged.

Read REPORT.md, manifest.json, per-case.json and semantic-reference-checks.json.
per-case.csv contains15 published plus32 generated latency/noise rows. Generated
work counts are null because their IR/metadata remain remote. The source manifest
identity and all15 current PTX hashes were independently reproduced locally.
The latency figure uses the actual retained32-pair samples. Metrics use a
theoretical boost FP32 ceiling, not observed/sustained hardware capacity.

The next command, from the repository root, resolves private runtime details:

```sh
cat .agents/skills/emmy-private-runtime/SKILL.md \
    .agents/skills/emmy-private-runtime/runtime-state.json
```

The only remote attempt requested the configured START.md and availability of ncu
through the configured CPU wrapper; hostname resolution failed before connection.
Keep connection details and launch commands in private configuration. Do not
infer service availability from this handoff, start another research run, or
interact with the main chat's workers. Have the active campaign's coordinator
schedule any profiling using its current budget, source/gates and shared
admission. If ncu is available, query its version/sections in CPU mode first,
then profile only the attention entry needed to distinguish score redundancy,
dependencies and issue/memory costs. Preserve normal isolation and atomic checks.
Profiler replay times must remain distinct from unprofiled latency. If unavailable,
retain the counter fields as null; do not change shared-host permissions/services.

Copy this audit directory to a new directory under the configured durable research
root after access returns. Verify copy hashes; no compiler transfer is involved.
Record current gate validity because later revocation cannot be checked offline.

Reproduce the local analysis (repository root):

```sh
.venv/bin/python experiments/2026-10-23-kernel-performance-audit/audit.py
.venv/bin/python .agents/skills/kernel-performance/scripts/metrics.py \
  experiments/2026-10-23-kernel-performance-audit/metrics-input.json \
  > experiments/2026-10-23-kernel-performance-audit/metrics-output.json
MPLCONFIGDIR=/private/tmp/emmy-kernel-audit-mpl .venv/bin/python \
  experiments/2026-10-23-kernel-performance-audit/render.py
```

The script regenerates only its own report data/PTX. It asserts current source,
all15 emitted PTX, published metadata/expectations and retained receipt identities
before producing results. A changed compiler needs new evidence, not a bypass.
