# Bounded compiler campaign

The authoritative run is `campaign-20261023-12h-v2`; its historical name does not change the revised nine-hour limit. It started at 2026-10-22 19:12:11 UTC and stops at 2026-10-23 04:12:11 UTC, or 200 evaluated experiments. Research artifacts live under `/root/nfs/emmy/research` in the configured private Linux runtime. All GPU results below are from the RTX 4090; sm120 results are assembly and native-code inspection only.

## Retained compiler

The installed compiler currently has source identity `549b6047f91818ea876deb76adfc88cabc36f554c05ab2bc51b2c961aa9143f2`, compiler SHA-256 `f47c21b0bb2818df21cd218573cd3cd1231f87955ee78c43a779f5b0be6ebac0`. It passed 278 distinct isolated GPU cases: all 15 published problems at exact metadata tolerances and 263 generated cases. The last candidate was held; its incomplete qualification is recorded separately below.

Ten correctness promotions repair shift/load handling, tile initialization, opaque KEY traversal, mutable lexical shadowing, Select snapshots, partial uniform loops, scalar BOOL SSA, nested seeded-fold scope, small-tail softplus, and small-input erf. These are general typed-IR changes. Numerical fixes have bounded tested domains; this campaign does not establish universal floating-point accuracy.

Three speed promotions were independently confirmed after selection, with fresh reservations and 47 cases × 32 paired measurements. Each pair used an exclusive admitted GPU job, unchanged source and cases, and case-specific same-job baseline A/A noise calibration.

| Promotion | Case | Baseline µs | Candidate µs | Speedup |
|---|---|---:|---:|---:|
| KEY batching | a01_attention | 199.68 | 66.56 | 3.00× |
| KEY batching | a02_attention_128 | 53.28 | 23.552 | 2.26× |
| Serial reduction prefetch | r02_big_sum | 20.48 | 7.76 | 2.64× |
| Serial reduction prefetch | r03_mean | 12.288 | 6.144 | 2.00× |
| Aligned vector dot loads | a01_attention | 66.56 | 18.432 | 3.61× |

These are separate paired comparisons against their respective parents. Do not multiply them into an independently measured end-to-end claim. Every promotion had zero resolved per-case regressions. Raw median increases remain visible, including 16 for prefetch and 10 for vector loads; these were inconclusive under their measured A/A thresholds. The latest vector comparison left 46 of 47 cases inconclusive, including a02.

Full per-case evidence is in `speed-per-case-results-v1.csv`, `prefetch-independent-per-case-v1.csv`, and `attention-vector-confirmed/attention-selected-independent-per-case-v1.csv`. The last directory includes the qualified receipt and all 47 raw paired rows. Resource warnings are retained: vector a01 register use rises while spills remain zero. A lower-register alternative was slower in its screen and was rejected.

## Tests and rejected hypotheses

The semantic inventory has 82 catalogued types plus one reserved type, below the 100-type cap. More individual cases exercise these types; case counts and behavior-type counts are different. New generators and independently computed references were reviewed and frozen before compiler qualification. Two numeric-input duplicates in the last draw were preserved, with no redraw.

The Q-cache variant had no resolved incremental a02 gain. An apparent mixed-F16 arithmetic defect was disproved by inspecting the complete assignment wrapper and running an independent 1,358-value control; its proposed repair was never applied. An attention-prefix candidate passed CPU preparation but its multiply-only gate declined all new divide-score cases, so it received no GPU qualification or speed claim. A broader divide-score prototype stays remote and unqualified.

## Final qualification

At final closure, selected source `212e052936722ec76f1c5c4c6b2243e9ff87fc79168f99284193d46b36eb15e4` (compiler `4ff5a7f903ce477ed88a06cdd9bbbfd68c25141c1835edbc97b60c7b9dc8b040`) was held when the bounded run closed. All 278 unique isolated GPU cases passed for both candidate and baseline; exact-source gates are attested. Independent timing completed 33 of 47 cases, all inconclusive. Six raw median increases are preserved in `final-campaign/final-candidate-all-measured-cases.csv`; none was a resolved regression. The remaining 14 comparisons were not started. No timing was redrawn or inherited and no candidate promotion occurred. Its earlier 11-case screen is evidence for selection only: m01 25.60→20.48 µs, ten inconclusive, no resolved regressions. It is not installed or retained without the fresh 47-case independent confirmation.

The final native audit covers core47 plus fresh12 on both architectures. All six existing attention spill warnings match the retained parent exactly. The source, selection, reservation, draw order and 278-case union were independently verified. A descriptive count-field typo in the baseline plan is preserved and corrected in a separate note; the original plan, cases, evaluator and source were not changed.

## Git and official grading

Work remains on `n00bcak`. No Git network operation, publication or commit was performed. Preexisting user changes were preserved. The revised official-grading cadence counts speed promotions only; three retained speed promotions have not reached the five-promotion trigger. No official request was sent.

See `CURRENT.json` and the latest section of `HANDOFF.md` for the final state and exact next action. Remote `campaigns/campaign-20261023-12h-v2/HANDOFF.md`, checkpoint and immutable objects retain complete provenance, failed attempts, inputs/references, IR/PTX/cubin/SASS, commands and measurements.

Final closure: 33 evaluated variants, ten correctness promotions, three speed promotions, 83 total catalogued/reserved types. Run closed at 04:11:26 UTC before the 04:12:11 UTC limit, all owned GPU/CPU admissions clear and all research agents stopped. The retained source remains f47c/549b with 278 passes and its earlier qualified full47×32 independent timing. No official grading, commit or publication occurred. Final records are under `final-campaign/`.


## Post-campaign official evaluation — 23 October 2026, Singapore

The user explicitly requested evaluation of the retained compiler against the official endpoint. Exact source549b/compilerf47c passed14/15 published problems on RTX5090/sm120. m01_matmul_512 failed262069/262144 values, max absolute error144.74808502197266 at unchanged rtol=atol=0.001. Exact inputs, positional ABI and launch match the frozen4090 case. Independent NumPyfloat64x0@x1 convertedtofloat32 matches the published reference exactly; the official output disagrees. Root cause remains unresolved.

All15cases were evaluated in one16-request batch. The original stop on a numerical failure and continuation of only nine unstarted cases are both preserved; no request/case was retried and no source changed. No busy, rate-limit or service error was bypassed. Timings are indicative, with no paired5090speedup claim. Raw requests/responses, original result, continuation, per-case CSV, reference checks and failure arrays are under [official-endpoint](official-endpoint/summary.json). The official all15 gate is FAILED. Existing all15/278correctness and47paired timing claims describe the4090 only. No commit or publication.
