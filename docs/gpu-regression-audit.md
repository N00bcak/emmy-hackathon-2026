# Remote regression coverage correction — 2026-10-23

The previous 73-case selection was not adequately justified. Checking case IDs,
file hashes and prior passes established provenance, not behavioral replacement.
It omitted distinct remote programs. The corrected routine selection is **122
cases: 15 unchanged published contracts and 107 generated witnesses**. The compiler
remains `f47c21b0…`; this change makes no performance claim.

## What was inspected

The audit read the remote archive's 1,114 case records and actual IR/metadata
JSON documents, then identified the exact 278 cases in the retained compiler's
`a33674b0…` RTX 4090 receipt. It did not use the local suite as the case inventory.
Every one of those 278 remote records and all its manifested files was checked
again against its content hashes and the existing isolated, guarded pass receipt.
No input/reference/tolerance was changed, and no new GPU execution occurred.

[The coverage manifest](../tests/gpu-regression-coverage.json) records each case's
identity, record hash, actual AST digest, retained witnesses, exclusion rationale,
and sample-normalized topology comparison. The policy binds its hash; the selector
rejects a changed manifest, a missing witness, or lost recorded IR surface tokens.

## Restored coverage

| Remote behavior | Why the earlier witnesses were insufficient |
|---|---|
| Async-ring lifetime | Simple async transport does not test dead lookahead, short prefill (including invalid prefill), nondivisible counters, observable padding or four-slot/lead-three staging. |
| Exact shared tree | Ordinary reduction does not enforce length-2/4/8 F32 association, immutable snapshots and observable scratch continuation. |
| Per-lane serial addition | Tree reduction does not preserve strict serial order, all four tail residues or explicit versus implicit initialization. |
| Shared serial dot pairs | Generic matmul does not cover alignment/stride, both seed modes and the pair-load tail boundaries. |
| Mixed F16 result boundary | A round-trip cast does not test F32 computation followed by one F16 rounding across the combined operation controls. |
| Attention dataflow | Generic attention does not replace seedFalse/seedTrue KEY loops, segment widths 16/32, partial fallback, full-segment tails, wide cancellation order, duplicate normalization or distinct-query normalization partitions. |
| Empty BOOL folds | A positive-trip fold does not test initial-carrier preservation at zero trips. |
| Independent schedules and boundaries | Restore scalar/width-two pointwise, standalone serial matmul, generated atomic matmul, RMS/L2 normalization, serial and single-slice cooperative softmax, ordered three-kernel attention, direct uniform-loop barriers and width-eight KEY scope. |

One existing single-trip nested-fold sample was replaced with its existing
multi-trip witness, preserving the same control topology and all eight observable
scope outputs. No compiler or generator was changed and no new behavior type was
created. The campaign registry remains 82 catalogued + one reserved = 83/100 types;
case instances and behavior types are different counts.

## Why the remaining 156 cases are not routine defaults

For **154**, at least one retained witness has the same AST topology after
normalizing random tensor-name suffixes and numeric literals. Boolean flags,
operators, variable names and keyword structure remain in the comparison.
Numeric normalization is diagnostic: tree lengths, ring stages, zero-trip paths,
seed modes, segment widths, tails and numerical domains were reviewed separately
and retained where they distinguish behavior. All 94 extracted statement,
operator, dtype and control-flag tokens remain represented; token coverage alone
was not used as a replacement argument.

The other **two** are broader-program replacements. All eight top-level statements
of the earlier lexical-carrier program occur in order in the combined version;
all eleven statements of the earlier BOOL Assign-less program occur in order in
the broader scalar-relations version. Both checks replay against the remote IR.

These checks support mechanism coverage, not mathematical equivalence across
inputs or unseen IRs. Archived seeds, shapes, constants and input domains still
provide additional sampling. They remain intact for targeted expansion and fresh
independent confirmation; removal from the default does not delete that evidence.

## Evidence and resumption

Under the privately configured research root, `regression-suites/compact-v4/`
contains the policy, coverage map, replayable `verify_coverage.py`, selected case
list and `verified-source-evidence.json`. That CPU verifier rechecks actual remote
files, AST mappings, compound-program inclusion, all selected surface features,
and the source-bound pass receipt before installing the default. The successful
CPU job is `20261023T083953Z-research-cpu-JZXbus`. A first attempt stopped before
activation because Python 3.12 and 3.13 format empty AST fields differently; the
replay now uses an explicit, version-independent field serialization. Its failed
attempt and staged files are preserved in the campaign audit directory.

Local and remote
policies and coverage manifests are byte-identical. The current compiler pointer
and official RTX 5090 m01 failure remain unchanged.

Thirty-four selector/pipeline CPU tests passed, including audit-tampering and
unrepresented-case rejection. Discovery remains off. Use the synchronized selector
and targeted archive expansion described in [the suite guide](gpu-regression.md).
