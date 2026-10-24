# Consolidated 100-check compiler gate

Use `tests/experiment-gate.json` for compiler experiments. It selects **66 GPU
cases plus 34 additional CPU leaf checks = 100**, including subtests. The planner
rejects an over-budget selection before invoking a compiler or worker.

All 65 original GPU witnesses remain unchanged: 15 published contracts and 50
core cases. The additional case is the byte-exact export of the already
demonstrated mutable scaled-score alias regression. No new GPU scenario,
reference, tolerance, launch geometry or ABI was generated during consolidation.

## What was consolidated

| Previous work | Consolidated execution |
|---|---|
| 15 published CPU compile/assembly cases, also run on the GPU | One source-bound published case carries entry checks, both-target assembly and GPU output checks. |
| m01 compiled again for index-hoisting assertions | The same assertions inspect that published GPU case's retained PTX and original IR. |
| m02 compiled again for asynchronous-copy fallback assertions | Those assertions inspect the same published GPU case's retained PTX. |
| Separate natural-log and scalar-math fixtures | One scalar-math program contains the original F32 log operation and preserves the opcode and ln(2) assertions. |
| Six static/dynamic copy-address inputs | Still **six distinct CPU checks**, not one method. |

The arithmetic is `118 - 15 - 2 - 1 = 100`. All 111 existing CPU assertion sites
are retained. Moving assertions onto the same compiled program removes duplicate
case execution; it does not waive the assertions. Standalone `CompilerTests`
remains available and now has 51 leaf cases. Do **not** run that whole standalone
suite in addition to the combined gate: that would repeat shared cases and
exceed the budget again.

## Execution

From the repository root, before preparation or GPU admission:

```sh
python -m tools.experiment_gate plan --out /tmp/emmy-gate100.json
```

The output includes a normal `cases` list for the existing isolated preparation
and GPU correctness workflow, plus the complete CPU leaf inventory and a plan
identity. It pins the test source, runner, original GPU policy and case records.
An extra GPU case requires a new reviewed consolidation; a 101st case fails.

After source-bound GPU confirmation, run these **inside the configured Linux
CPU runtime**, with fresh output paths:

```sh
python -m tools.experiment_gate prepare-shared \
  --root "$EMMY_RESEARCH_ROOT" --gpu-evidence "$GPU_EVIDENCE_ID" \
  --out /tmp/emmy-shared-checks.json

python -m tools.experiment_gate check-cpu \
  --root "$EMMY_RESEARCH_ROOT" --shared-id "$SHARED_CHECKS_ID" \
  --out /tmp/emmy-cpu-checks
```

`prepare-shared` returns `SHARED_CHECKS_ID`. It verifies the accepted GPU receipt,
the exact 66 case identities, source bytes, original published IR/metadata and
both-target assembly artifacts. `check-cpu` revalidates the source and plan,
runs the shared assertions on those 15 published programs, and executes the 34
additional leaf checks under a separate restricted UID. Unexpected parameter
cases, skipped tests, missing methods or changed source fail closed.

An edited compiler needs GPU evidence for its **exact new source** before it can
reuse any compiled program. Existing all-15, guard, sanitizer, reset, admission
and independent performance-confirmation requirements remain unchanged.

## Evidence scope

The consolidation qualification used the retained compiler's unchanged,
source-bound 66-case GPU execution record and freshly executed CPU checks. It
did not rerun the GPU or relabel the old 118-check run as compliant. See
`docs/evidence/experiment-gate-consolidation.json` for identities and results.
Historical campaign receipts remain untouched. The already spent official
batch is not repeated, and the compiler bytes remain exactly those it checked.
