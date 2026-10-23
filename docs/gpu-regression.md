# GPU regression suite

The routine gate is **65 cases**: all **15 published contracts** and **50 audited
core witnesses**. [The policy](../tests/gpu-regression.json) binds the original
case-record hashes. [The scope ledger](../tests/gpu-regression-scope-v6.json)
accounts for every case in the 100-case bundle and records its role and trigger.

Floating-point acceptance is final-output error within the case's unchanged
rtol/atol. Selective FTZ, approximate math, FMA and reassociation are permitted
when they meet that budget. Correct ABI, output membership, indices, masks,
collective participation, state lifetime and integer/transport behavior remain
required. A stress fixture does not establish a universal precision requirement.

The other 35 bundled cases remain available: **28 targeted** mechanism/dtype
checks, **6 stress diagnostics**, and **1 weak numerical oracle** awaiting an
independently validated replacement. Their inputs, references and tolerances
are unchanged. The [test audit](test-relevance-audit/REPORT.md) explains each case
and the remaining coverage gaps; it does not declare targeted tests redundant.

## Select and verify

All 100 payloads are included in [tests/gpu-cases](../tests/gpu-cases/README.md).
From the repository root, use fresh output filenames:

```sh
python -m tools.bundled_regressions
python -m tools.bundled_regressions --suite campaign100 --out /tmp/emmy-catalog.json
python -m tools.regression_suite --catalog /tmp/emmy-catalog.json --out /tmp/emmy-core65.json
```

The first command verifies every payload and resolves the default core65
selection. The selector produces exactly 65 cases, including all 15 published
problems, and retains both assembly targets, sm_89 and sm_120.

For a changed mechanism, inspect its scope-ledger `run_when` entries and add the
relevant case IDs explicitly with `--include-case CASE_ID` (repeatable).
`--expand-version VERSION` remains available for versioned families. `--full`
replays the supplied catalog explicitly; it is not the routine default.
Expanded cases in this scope retain their full record-hash checks, including
sanitizer and isolation flags.

## Historical policies

The superseded 93-case policy is preserved byte-for-byte as
[compact-v5](../tests/gpu-regression-compact-v5.json); compact-v4 and its scope
history remain unchanged. To reproduce v5, pass
`--policy tests/gpu-regression-compact-v5.json` to the selector with the full
catalog. `python -m tools.bundled_regressions --suite default93` verifies and
resolves the historical selection. See the [v5 record](gpu-regression-v5.md)
for historical receipts; they do not qualify this new policy automatically.

## Qualification

Selection is not a new GPU execution receipt. Source-bound all-15 passes for
both comparison arms remain mandatory before timing. A research runtime must
freeze this v6 selection under a new suite identity and qualify both arms before
using it; existing v5/100-case receipts must not be relabeled as v6 evidence.
Preserve admission, full-output checks, guards, atomic resets, required
sanitizers and independent confirmation. Assembly for sm_120 establishes
instruction legality, not RTX 5090 execution correctness.
