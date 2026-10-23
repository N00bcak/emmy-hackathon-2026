# Compact GPU regression suite

The routine suite is **73 cases / 73 kernels**: the unchanged 15 published
contracts and 58 generated behavior witnesses. This replaces cumulative archive
replays as the default, including the recent campaign’s 263 generated cases. Exact cases, references, tolerances, launch
geometry, guards, isolated repetitions, and required sanitizer flags are retained.
The compiler contains no case-selection logic.

[The policy](../tests/gpu-regression.json) records every retained case's content
identity, complete case-record hash, and reason for inclusion. It covers scalar
ABI widths and unused parameters; XYZ launch geometry; typed arithmetic,
conversions, bitcasts, NaN truth and natural log; expression cache keys; lexical
Tile/Init scope; partial reduction identities and explicit seeds; tree/warp
carrier lifetime; live loop steps and captured ends; signed addresses; unaligned
vectors; async/shared transport; fused attention and cooperative softmax tails.
The `_gid`/partial-warp witness and both previously failing loop-header cases
remain explicit regression anchors. The current policy also retains representative
witnesses for 64-bit shifts, Load lifetime, nested Tile Init, opaque expression
keys, mutable and Select shadowing, partial-CTA uniform loops, BOOL carriers,
nested seeded folds (including optimized KEY loops), and softplus/erf composition.
Overlapping schedules and random seeds remain available through targeted expansion.

This is a small set of witnesses for those mechanisms, not a proof of correctness
for every IR or an assertion that every archived case is equivalent. Repeated
random draws and shape permutations no longer accumulate into a mandatory replay.
Additional operator, dtype, schedule, and input-domain permutations remain in the
archive for changes affecting those paths. CPU deserialization, rejection, static
PTX/assembly and oracle tests retain their wider matrices without taking GPU time.

## Select cases

Resolve runtime and archive paths from private configuration. A catalog is an
existing case-list, a JSON list of records containing `path`, or an inventory
mapping case paths to records. Multiple catalogs can be supplied. Selection
deduplicates aliases by content identity and verifies every selected file and
the entire frozen case record, including sanitizer/isolation flags.

The canonical `regression-suites/current.json` manifest binds the retained source,
compiler hash, policy hash, catalogs and selected case list. `retained/current.json`
is an alias of that manifest; `retained/compile.py` points to its immutable source.
`retained/versions.json` groups historical source identities by compiler hash.
Use the synchronized defaults; stale source or policy identities fail closed:

```sh
python -m tools.regression_suite --root "$RESEARCH_ROOT" --out "$NEW_CASE_LIST"
```

An explicit catalog can also be selected with `--catalog "$CASE_CATALOG"`.
To install a revised source-bound default after a closed run, supply `--root`,
`--install`, `--catalog`, `--policy` and a new `--out`. Installation refuses active
owned GPU work, preparation or discovery. It updates pointers, not frozen objects,
and does not create a correctness gate.

The current 73 witnesses were checked against the retained compiler's existing
278-case RTX 4090 correctness receipt, including input identities, isolation,
guards and required sanitizer evidence. Consolidation performed no new GPU run.
The official RTX 5090 `m01_matmul_512` failure remains unresolved; this selection
must not be represented as an all-15 pass on that device.

The output is the normal `cases` list accepted by the existing restricted CPU
preparation and admitted, guarded GPU correctness adapters. It does not establish
a correctness gate or authorize benchmarking. Missing witnesses, changed case
bytes, or missing/duplicated published contracts fail selection. Output creation
is exclusive so an interrupted reservation cannot silently be replaced.

For a changed mechanism, expand only its relevant archived version:

```sh
python -m tools.regression_suite --catalog "$CASE_CATALOG" \
  --expand-version tree-halve-live-shared-v1 --out "$NEW_CASE_LIST"
```

Use `--full` for an explicitly requested complete archive replay. Keep all archives
and failure evidence; selection does not delete or rewrite them. A newly found
failure earns a regression witness when it exposes a distinct mechanism; another
seed or size alone does not require permanent growth of the routine set.

## Fresh confirmation and caching

New protocol registrations default to `research-all15-compact-v2`, with **one
fresh generated case per selected research family** for exploration and again
after source freeze for independent confirmation. For eight selected families,
that means eight fresh cases rather than 32. Select the families affected by the
change; the 15 published contracts remain mandatory regardless of family.
The fixed routine witnesses complement fresh draws. In particular, the routine
softmax witness retains the 257-column path when a single fresh draw uses 127
columns. Fresh cases do not become another permanent cumulative replay batch.
Use `--cases-per-family` when a hypothesis needs a larger discriminating sample.

Existing frozen evaluator snapshots and reservations retain their original
sampling policy. The new protocol requires new source/suite-bound gates for both
arms before performance evaluation; old gates cannot qualify the changed
protocol. Tolerances, sanitizer policy and isolated confirmation are unchanged.

Reuse exact compiled artifacts and prior correctness receipts only while their
source, evaluator, case/input/reference identities, launch contract and required
safeguards remain valid. A changed compiler needs fresh GPU correctness. A
changed evaluator or oracle requires a new version and reevaluation of both arms.
Do not recompile or rerun unchanged inputs merely because the archive grew.
Timing remains a separate exclusive paired experiment after both all-15 gates;
this consolidation makes no compiler-latency claim.
