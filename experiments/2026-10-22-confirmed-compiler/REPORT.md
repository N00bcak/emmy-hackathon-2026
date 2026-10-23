# Confirmed compiler improvement

Only source `e02740fea768412b05c780b12f5d2c560ecce74561b02b66a14ca949615f50a2` is retained. Its compile.py SHA256 is `a8265ab9a78b0663d4daa2a5b16ac62e5a2d5dce6bfe03420306cdbcf893e8a2`; the starting compiler SHA256 was `d35da050e41fd3a172e5611b69a48ccabecb2f1d359aa9a4204e5e2685c9bcbf`. No fixture names, values or shapes are used as solutions. Typed IR determines specialization; metadata ABI and launch geometry remain contracts.

The retained source fixes signed32 Python floor division/remainder, including MIN%-1; restores lexical Loop/StridedLoop axis mappings; and supports conservatively proven complete shuffle subgroups in partial CTAs. It adds conservative integer range proofs, shared-only pure inner-loop unrolling by eight, and synchronous vector global-to-shared transport for aligned F32 copies. Barriers and floating-point accumulation order are preserved.

## Correctness and independent confirmation

All 15 published problems pass at their exact metadata tolerances. Source-bound gate: `b377391c61eb8f8af724574e905c45a7a769bcb65f4e47d625ad7afb7a186881`; suite: `38ac421af22b2cee2106537327511b9d034dcabc84022f33483c672b1ebbdf17`; independent confirmation receipt: `82f1598df1eddbac64993a189a8cd5a8f22737eeb5f9a3f0cc2079e0dd8bccbf`. Published gate checks preceded timing. Existing Linux CPU regressions passed (eight original methods and nine generality methods); sm89 and sm120 assembly checks passed. Only RTX4090 execution was tested.

The final existing-artifact audit qualifies 149 distinct GPU cases: 15 published and 130 generated. References use independent NumPy calculations and fixed per-version tolerances. Fourteen additional raw passing cases are unverified: eight atomic matmuls, two shared reductions and four fused attention cases depend on conditional-only implicit accumulator seeding that the primary IR contract does not establish. Their bytes/results remain preserved. They do not count toward correctness or performance claims. Earlier 163/153 counts were too broad. The qualification ledger identifies every excluded case. Generated m02 coverage is reopened; its published pass remains valid. This is evidence of improved generality, not proof over all legal IR.

## Per-case performance

Published `m01_matmul_512` improved from 128.000006 to 40.959999 microseconds: paired median ratio 0.319999977, approximately 68% lower latency. The gain reproduced after source freeze and selection on a fresh confirmation reservation. The initial screen used 16 pairs with before-timing A/A noise threshold 2.38%; fresh confirmation used 32 pairs and follow-up A/A context of 1.42%. Fresh core follow-up A/A was a separate job, not a calibration inside that original burst. Fresh shared cases used A/A before timing in the same job.

`per-case-performance.csv` preserves all 57 independent raw rows, explicitly marking five unqualified rows. Among the 52 qualified comparisons there is no regression resolved beyond the recorded case-specific noise threshold. Small kernels have broad noise, so this does not prove zero regression. Each 32-pair replay is one captured block; slices are not independent confirmations. Published m02 had an initial neutral 5.12-to-5.12 microsecond median. Most code was unchanged: 45/47 initial suite PTX/SASS identities and 46/47 resource records matched the starting baseline. No spills were introduced. m01 retained 40 registers. m02 used 36 registers on both targets versus 35 on sm89 / 37 on sm120 originally; the +1 sm89 change is a resource warning.

Timing covers exclusive paired warm allocations on RTX4090. No 5090 execution, hosted calibration, streaming workload or hidden-set performance claim is made. No aggregate conceals a per-case regression.

## Rejected and unfinished work

Uniform unroll-four was rejected: a01_attention regressed 199.68 to 296.96 microseconds (+49.3%), despite its m01 gain. Guarded async copy was rejected: m01 regressed 128 to 146.432 microseconds (+14.6%). A scalar async change was within noise; a native async screen was not independently confirmed. Only the synchronous vector mechanism was retained.

Later prototypes remain in the research store and are not shipped:

| Experiment | Observation | Why unretained |
|---|---|---|
| let-snapshot-copy-v1 / 490f4cd6 | Fixed eight independently generated mutable-Let snapshot failures; original e027 failed all eight with guards intact and clean memcheck | No fresh post-selection confirmation of final combined source |
| let-declared-conversion-v1 / 06d786b1 | Fixed eight finite declared-type conversion cases; e027 failed all eight; source06 passed the published/core gate and conversion controls | Snapshot/BOOL/vector/CSE supplemental checks and fresh independent confirmation incomplete |
| nonnegative-unsigned-divrem-v1 / 86d0f4ea | m01 screen 40.96 to 35.952 microseconds; paired median ratio 0.892445 versus 7.60% A/A threshold | Incremental gain screened only, not independently confirmed |
| global-vector-memory-v1 / 3b1e28fa | Published p00-p03 each had paired median ratio 1.0 over 200 pairs; no resolved gain, broad noise | No performance benefit established; code and some sm120 register counts increased |
| shared-affine-hoist4-v1 | Generic affine shared-load vectorization plan only; hypothetical 37.5% fewer load issues for m01 | No source implementation or GPU result |

Known valid Let snapshot and declared-type conversion failures therefore remain in the retained compiler. Do not describe all legal generated cases as passing. The next campaign should first repair the conditional-reduction oracle using explicit Init, validate/freeze a new evaluator version, then reevaluate both baseline and candidate. The corrected shared-alignment generator draft `60bbf302...` is static-only: not approved, adopted or executed. No replacement atomic evaluator version has been created or run.

## Stop and resume

The explicit run began 2026-10-21 23:13:29 UTC with a six-hour / 100-evaluated-variant limit. Hard cutoff: 2026-10-22 05:13:29 UTC. A later clock check was past the cutoff; research was stopped, and only evidence finalization and transfer followed. The last GPU jobs were before the cutoff. The run is closed with discovery disabled; 23 evaluated experiment IDs are recorded in qualified-assessment.json. Hosted calibration remained off. No commit or publication was performed; owner branch remains n00bcak.

Durable evidence is under `<research-root>/campaigns/campaign-20261022-6h-v1/`. Immutable source/case/input/reference/PTX/cubin/SASS objects, commands, environment/admission records, raw measurements, rejected prototypes and failed drafts remain there. Read HANDOFF.md and checkpoint-final.json before any future user-authorized run. Machine connection details remain in ignored private runtime configuration.
