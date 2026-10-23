# Historical 93-case GPU regression suite

The default is **93 cases**: the unchanged 15 published contracts and 78
additional witnesses for hidden-set behavior and existing marked operators.
The previous 122-case policy remains selectable as
[compact-v4](../tests/gpu-regression-compact-v4.json). No fixture, reference,
tolerance, launch metadata, guard or sanitizer requirement was changed.

The [default policy](../tests/gpu-regression-compact-v5.json) binds complete case-record
hashes. The [scope decisions](../tests/gpu-regression-scope.json) account for all
122 previous witnesses, including each powenn comparison result. Selection checks
the disposition ledger and its archived policy hashes. The old
[coverage audit](gpu-regression-audit.md) describes the historical v4 scope; v5
does not claim its archived scenarios are equivalent to retained cases.

The retained set includes generated programs with independent references,
reordered ABI, partial tiles and warps, reduction seeds and carrier lifetimes,
atomic contractions, async ring lifetimes, softmax tails and distinct attention
normalization graphs. All 21 existing witnesses containing marked operators
remain, including Select, Builtin, FuncCallExpr, CastExpr and FlatIndex. This is
existing marked-operator coverage, not complete support for every marked API.

29 witnesses move to targeted archive checks: extra scalar-width and BOOL-carrier
matrices, internal NaN and amplified extreme-domain probes, exotic affine folds,
strict F32 tree/serial accumulation order, a combined ABI stress case, and six
additional attention subgroup-width permutations. These scope decisions preserve
the original cases; they neither relax their tolerances nor establish that their
failures are harmless for every application. Integer and representation checks
still require exactness where their original metadata does.

## Evidence and assembly

The saved powenn comparison passed all 15 published problems on sm_89 with only
PTX target/version headers adapted for that device. Across the former broad suite
it passed 59/122; that number is not a challenge score. The ledger retains failures
when they exercise relevant behavior or the explicitly requested marked operators.
The user reports all trials pass on the 5090. sm_89 results must not be relabeled
as sm_120 execution evidence.

The [source-bound receipts](evidence/powenn-test-scope-v5.json) record each case's
metadata tolerances, saved numerical checks/errors and receipt hashes, plus fresh
PTX/cubin hashes for both assembly targets. Our current compiler assembles all
93 cases (95 kernels) on both targets. Powenn assembles 66 cases on both and passes
50 saved sm_89 numerical checks; 21 cases reject during compilation, six during
assembly and 16 fail numerical checks. The 15 published cases pass for both arms.
Our exported per-case numerical receipts cover those 15; assembly of the other
78 cases is not a fresh GPU correctness receipt.

Compiler assembly tests now assemble their generated PTX for **both sm_89 and
sm_120** with a compatible PTX version and target header. This checks instruction
legality, not GPU results. Runtime correctness uses the case's own frozen rtol/atol
and independent references. Source, evaluator, case and device identities must
accompany receipts; assembly success alone cannot satisfy a numerical gate.

The condensation itself claims no compiler performance improvement. The subsequent
[powenn merger](powenn-merge.md) records the incorporated source and its results.

## Selection and confirmation

The fixture payloads are now bundled in [tests/gpu-cases](../tests/gpu-cases/README.md).
Run `python -m tools.bundled_regressions` to verify all 100 copied cases and the
unchanged default 93-case policy. Its `--out` option resolves portable fixture
paths into a local catalog for the existing selector. The seven campaign extras
remain separate from the default selection; no tolerance changed during export.

Resolve runtime paths from private configuration. Select from the existing archive:

```sh
python -m tools.regression_suite --catalog "$CASE_CATALOG" --out "$NEW_CASE_LIST"
```

Use `--policy tests/gpu-regression-compact-v4.json` for the former scope,
`--expand-version VERSION` for affected archived mechanisms, or `--full` for an
explicitly requested archive replay. Selection verifies payload and entire record
hashes, deduplicates aliases, and requires exactly the 15 published contracts.
It never rewrites or deletes archived cases. Output creation is exclusive.

`--root "$RESEARCH_ROOT"` uses the runtime's installed source-bound defaults.
Those defaults must be explicitly updated after a closed run using `--install`,
`--root`, `--catalog`, `--policy` and a fresh `--out`; changing this checkout alone
does not change an installed runtime. Installation refuses active owned work.
It updates pointers and does not grant a correctness or performance gate.
The configured runtime was updated to v5 after validating all 93 records against
the unchanged 1,114-case archive; 20 selected cases retain sanitizer requirements.

Fresh hidden-set draws complement fixed witnesses. Use the families affected by
a change, with independent confirmation after source freeze. Both baseline and
candidate must pass all 15 published problems at exact metadata tolerances before
performance evaluation. A source change invalidates that gate. New evaluator or
oracle versions require reevaluating both arms. Retain isolated execution,
buffer guards and required sanitizer checks; timing remains exclusive and paired.
