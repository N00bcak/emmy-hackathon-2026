# P2 shared value reuse: scaled f02 active and faster

## Outcome

P2 implements the shared typed value-reuse analysis. Its f02 scalar second-pass
row cache activates at the 3072x2048 scaled witness and survives `ptxas` on
both targets. Hosted RTX 5090 timing resolves f02 at 0.763x baseline, and an
independent run reproduces it. P2 therefore meets the plan's row-reuse timing
exit on f02, with no resolved regression on the f01/f03 guards.

- Base: `d8f65f0`
- Candidate `compile.py` SHA-256: `0a6f12033a942f4da9176ebf1e963c16193d8883610d3d4807bbc9870ab0de2a`
- Branch: `compiler-design/p2-shared-reuse`; integrated into the local
  `compiler-design/access-reuse-plan` worktree, not published.

## Mechanism

`ReuseValue`, `reuse_expr_key`, and `ReuseAnalysis` provide typed structural
identity, transitive readonly-load summaries, CTA-uniformity facts, and
fail-closed shared-hoist admission. The scalar phase path caches the exact
per-thread row values between a reduction and its following map, with strict
header, affine-load, alias, register-budget and fold-order checks. The existing
four-wide f01/f03 row cache remains ahead of the scalar path.

## Correctness

All checks used the candidate hash above:

- CPU suite: 50/50 pass, including scalar-cache activation and fail-closed
  tests plus structural A79/A83 coverage.
- Published contracts: 15/15 pass at exact metadata tolerances.
- Routine audited policy: 65/65 pass, including A79/A83; 16/16 sanitizer checks;
  source unchanged. The merged-tree rerun reproduces 65/65.
- Scaled full-output witnesses: 8/8 pass.

Receipts: [published.json](published.json) (`1b137506…`), [core65.json](core65.json)
(`5ded5c75…`), [scale_gate.json](scale_gate.json) (`7cdddd8d…`) and
[a79_a83.json](a79_a83.json) (`812c26f7…`). All bind candidate hash `0a6f1203…`.
The activation bundle is [acceptance.json](acceptance.json) (`22a5b2b9…`) and
[sass_audit.json](sass_audit.json) (`404ab0dc…`).
The integrated-tree receipts were regenerated after merging P2 and reproduce the
same results. The earlier pre-merge P2 receipts (`78b04e54…`, `2df464c2…`,
`cbffcc5c…`) remain in the research store for provenance.
The candidate and test files are preserved as [compile.py](compile.py) and
[test_compiler.py](test_compiler.py).

## Final-SASS activation

Candidate, original baseline, and forced-fallback arms were assembled with the
same `ptxas -v -arch=<target>` flags. Final SASS was disassembled with `cuobjdump
-sass` on both targets. Evidence root:
`.emmy/research/performance-v1/p2-gates-0a6f1203/hard-sass-acceptance/`.

| Case | Target | Candidate / fallback cubin | Candidate / fallback SASS | LDG | STG | Registers | Spills |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| f02 3072x2048 | `sm_89` | `358e946a…` / `8bc3fb43…` | `b4f77e0c…` / `9930e109…` | 16 / 48 | 16 / 1 | 40 / 40 | 0 / 0 |
| f02 3072x2048 | `sm_120` | `a02885c0…` / `b4c66738…` | `0fa2db8d…` / `1fdb236e…` | 16 / 48 | 16 / 1 | 40 / 40 | 0 / 0 |

The `sm_120` values above are from the independent `sass-final` rerun. An older
acceptance bundle built the same PTX differently and reported
`7e56b562…`/`99f99030…`; it disassembles to the same instruction stream,
resource counts and `LDG` counts. Hash comparisons are therefore made within one
build mode, and the covarying SASS result is the activation evidence on both
targets.

An independent rebuild of the stored PTX is recorded in
[p2-p4-sass-recheck-20261027.json](../p2-p4-sass-recheck-20261027.json)
(SHA-256 `a0c7dc94023f47436cd18ec305df7dc6671fc771177fa83971cf7b7a8a2f728e`).
It records raw and normalized instruction hashes for every arm and target, plus
registers and spills. The integrated PTX build and the older `sass-final` build
agree on decoded `sm_120` opcodes; only their `.headerflags` virtual-SM marker
differs. Raw hashes are therefore compared within a build mode.

The fallback's numerator map re-issues its `x` loads; the candidate maps them
from registers retained across the CTA total. SASS instruction counts fall
1120->688 (`sm_89`) and 1160->720 (`sm_120`). The scalar-off forced fallback is
byte-identical to the original baseline.

The store count changes in the opposite direction: the candidate replaces one
staged vector store with 16 scalar stores. The net global-memory instruction
count still falls from 49 to 32 per target, and the map-phase reload is gone.
This is the recorded register-retention trade, not an unaccounted extra load.

Scaled f01, f03, r04, and a01 are byte-identical to baseline in PTX, cubin, and
SASS. f01/f03 still activate the pre-existing four-wide row cache when that
cache is forcibly disabled; P2 preserves rather than newly introduces it.
Scaled f02 at 4096x4096 still fails closed at `ROW_CACHE_VALUES = 32` and is
not timed as P2 activation. The independent
`sass-final/summary.json` (`d62efe13…`) recompiles all five scaled witnesses
from frozen baseline and P2 sources and reports the same result: f02 differs
on both targets; f01, f03, p02 and p03 are byte-identical.

## Hosted RTX 5090 timing

Seven alternating pairs, warmup 10, 50 iterations. A/A for f02 was median
1.0028 with a 1.69% observed absolute maximum, so its calibrated deadband is
1.69%. The f01 control's A/A maximum was 5.25%; no P2 candidate result exceeds
that conservative band.

| Case | Candidate ratio median | Deadband | Verdict |
| --- | ---: | ---: | --- |
| `f01_rmsnorm_3072x2048` | 1.0025 | 5.25% | unresolved (byte-identical SASS guard) |
| `f02_row_softmax_3072x2048` | 0.7630 | 1.69% | **faster**, confirmed |
| `f03_l2norm_3072x2048` | 1.0000 | 1.0% | unresolved (SASS-identical guard) |

Raw results: [f01_primary.json](f01_primary.json) (`6b42516a…`),
[f02_primary.json](f02_primary.json) (`3fc34945…`),
[f02_confirm.json](f02_confirm.json) (`768d8742…`),
[f03_primary.json](f03_primary.json) (`429a6999…`), and the
[f02 A/A control](aa_f02.json) (`cb67080f…`). All are bound to candidate hash `0a6f1203…`;
the f02 pair also carries the identical candidate PTX hash `947277c7…`.

## Disposition and limits

Local final-SASS activation and a resolved hosted f02 win with isolated
confirmation exist for this source, and the exact source is integrated locally.
One limit remains: the hosted runner stored cubin identities but not cubin
bytes, so the hosted binary itself could not be disassembled. Local activation
is established from the same candidate PTX, but hosted cubin/SASS identity is
unverified. P3 can use this active P2 base, provided it proves final-SASS
activation at both scaled attention shapes before timing.

This is local full-output correctness plus hosted timing on a frozen source. No
hidden-set or streaming-throughput claim follows.
