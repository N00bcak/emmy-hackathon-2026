# Updated main 0914413 comparison

This supersedes the older 2647806 comparison for the newly cached origin/main commit `09144135c03e10cccbce3569ba9e31e8b82217eb`. No Git network operations were performed. The local main checkout and all preexisting uncommitted compiler/owner files remain preserved.

## Result

The updated main has caught up with the retained compiler on the published latency comparison. None of the 15 per-case differences exceeds its measured A/A uncertainty. For example, attention measures 10.24 µs for both sources and matmul 512 measures 19.456 µs for both. The old 19× attention gap does not describe this new main.

Updated main passes all 15 published cases and all 93 default cases, but only 93/100 in the expanded campaign set: 2 compile/assembly failures and 5 GPU correctness failures. The retained source still has its 100/100 confirmation. All original tolerances and case bytes were preserved.

The new vector reduction path is rejected for general integration. It redistributes elements across thread owners without proving that per-thread partials are unobservable or that the loop expression is independent of the old thread owner. On the unchanged existing archived witness, updatedmain produces 105/128 mismatches (max absolute error 268435460); retained e021 passes all 3 repetitions exactly. Guards and inputs are intact and the independently reviewed host observations are clean. The failing main check stops on its first bad repetition. No new case was generated; this archived case is outside the selected 100. See [source review and coverage explanation](vector-legality-review.json) and [independent witness audit](independent-archived-witness-audit.json).

The naive merged prototype remains remote and rejected. The safe merged compiler remains the retained e021 source; no unsafe code was installed, no compiler commit or speed promotion was made.

## Per-case published timing

Both frozen source/suite all 15 gates preceded timing. Execution was on RTX 4090/sm_89; sm_120 assembly establishes no 5090 correctness or speed result. Each case uses two 32-pair A/A blocks and two 32-pair A/B blocks. Ratios are medians of within-block paired-ratio medians. Noise margin is max(1%,3 times the linear 90th percentile of absolute A/A ratio deviation). Absolute median latency columns are descriptive and their quotient may differ from the paired estimator. This warm-cache kernel-event comparison is not a throughput or host-latency claim.

| Case | Updated main us | Retained/safe merged us | Main / retained | A/A margin | Finding |
|---|---:|---:|---:|---:|---|
| a01_attention | 10.240 | 10.240 | 0.998 | 30.0% | inconclusive |
| a02_attention_128 | 7.168 | 7.168 | 0.940 | 42.9% | inconclusive |
| f01_rmsnorm | 5.120 | 5.120 | 1.000 | 60.0% | inconclusive |
| f02_row_softmax | 4.096 | 4.096 | 1.000 | 92.5% | inconclusive |
| f03_l2norm | 4.096 | 4.096 | 1.000 | 75.0% | inconclusive |
| m01_matmul_512 | 19.456 | 19.456 | 1.000 | 16.7% | inconclusive |
| m02_matmul_rect | 5.120 | 5.120 | 1.000 | 75.0% | inconclusive |
| p00_scalar_add | 4.096 | 4.096 | 1.000 | 100.0% | inconclusive |
| p01_vector_add | 4.096 | 4.096 | 1.000 | 100.0% | inconclusive |
| p02_scale_shift | 3.072 | 3.072 | 1.000 | 71.6% | inconclusive |
| p03_relu_mul | 5.120 | 5.120 | 0.994 | 100.0% | inconclusive |
| r01_array_sum | 4.096 | 4.096 | 1.000 | 100.0% | inconclusive |
| r02_big_sum | 7.168 | 8.192 | 0.881 | 42.9% | inconclusive |
| r03_mean | 5.120 | 6.144 | 0.995 | 52.4% | inconclusive |
| r04_row_sum | 3.136 | 4.096 | 1.000 | 100.0% | inconclusive |

The big-sum paired estimate favors main by 11.94%, but remains within its 42.86% noise margin. Such estimates are retained rather than relabeled as proved improvements or exact equality.

## Evidence

[Source identities, gates and comparison](comparison.json), [raw pairs](raw-pairs.json), [independent raw 15 audit](independent-raw 15-comparison-audit.json), [all 100 case outcomes](main100-case-results.json), [archived witness results](archived-perlane-results.json). The original archived IR, metadata, inputs and reference accompany this report under existing-archived-witness/. Full native preparation/PTX/cubin/SASS remain in the configured research store.

Timing evidence: `bd1d55d64aeab4180f7e06aff8246971f854671db7d1c243a226e4bc5690c176`. Main source: `8a649cd89434ebc543b3eb9221b8684a8e22749042f6a4a6b75cfbd84917dd98`. Retained/safe merged source: `e021a2d9d9547c09c7540f5f66ab571010bad2ea84fcea1484f5c9ddb874794e`.

Export archive SHA256 `699c26777c72d2704633dfc74758feb8e84c857b59a3300a01073a1521425ea2`; all 308 exported file hashes verified locally.
