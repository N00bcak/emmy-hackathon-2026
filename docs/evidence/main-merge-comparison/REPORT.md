# Main compiler integration and comparison

Historical comparison against2647806. See [updated main0914413 results](../main-merge-updated/REPORT.md) for the newer cached main.

The tests were committed on n00bcak as `a915e56`. The checkout is now on `main`, fast-forwarded to that commit. All preexisting uncommitted compiler changes and owner notes were preserved byte-for-byte. No Git network operation was performed.

The local main tip before checkout was `3f1b749`; the newer locally cached `origin/main` tip `2647806` was used as the main comparison arm. Its shared-window address fix is already present with identical AST in the current compiler. The merged compiler therefore equals the current retained compiler, source `e021a2d9d9547c09c7540f5f66ab571010bad2ea84fcea1484f5c9ddb874794e`. There is no additional merge speedup and no compiler commit was made.

## Correctness

Both distinct sources passed all 15 published cases before timing, including isolated atomic/reset and Compute Sanitizer checks. Assembly covered sm_89 and sm_120; execution used only the RTX 4090. No RTX 5090 execution claim is made.

The current/merged source has the retained 100/100 correctness confirmation. Cached main passed 53/100 (50/93 in the default suite): 29 compile/assembly failures and 18 GPU correctness failures. The existing cases, references and tolerances were unchanged. Full case identities and errors are in [main100-case-results.json](main100-case-results.json).

## Published-case latency

Warm-cache CUDA-event kernel latency on the 4090, with two 32-pair A/A blocks and two 32-pair A/B blocks per case. Ratios use the median of the two A/B block medians; uncertainty uses max(1%, three times the linear 90th percentile of absolute A/A ratio deviation). Absolute latency columns are descriptive medians; their quotient need not equal the paired estimator. This is a source comparison, not a new optimization promotion.

| Case | Cached main µs | Current = merged µs | Main / merged | Finding |
|---|---:|---:|---:|---|
| a01_attention | 199.680 | 10.240 | 19.425× | merged-faster |
| a02_attention_128 | 98.304 | 7.168 | 13.628× | merged-faster |
| f01_rmsnorm | 5.120 | 5.120 | 1.000× | inconclusive |
| f02_row_softmax | 4.096 | 4.096 | 1.000× | inconclusive |
| f03_l2norm | 5.120 | 5.120 | 1.000× | inconclusive |
| m01_matmul_512 | 68.608 | 19.456 | 3.526× | merged-faster |
| m02_matmul_rect | 5.120 | 5.120 | 0.900× | inconclusive |
| p00_scalar_add | 4.960 | 4.960 | 0.995× | inconclusive |
| p01_vector_add | 4.096 | 4.096 | 0.998× | inconclusive |
| p02_scale_shift | 5.184 | 5.120 | 1.001× | inconclusive |
| p03_relu_mul | 3.072 | 3.072 | 1.000× | inconclusive |
| r01_array_sum | 4.096 | 4.096 | 1.000× | inconclusive |
| r02_big_sum | 20.480 | 7.392 | 2.711× | merged-faster |
| r03_mean | 12.288 | 6.144 | 2.000× | merged-faster |
| r04_row_sum | 4.224 | 4.224 | 1.000× | inconclusive |

Five cases favor current/merged beyond measured noise; ten are inconclusive. In particular m02 estimates current/merged 11.11% slower, within its 75% A/A margin, so this possible regression is retained rather than called a tie. No case resolves as a regression under this criterion.

## Evidence

[Comparison and gates](comparison.json), [all per-case measurements](per-case.csv), [raw pairs](raw-pairs.json), [independent protocol audit](independent-protocol-audit.json), [independent raw-data audit](independent-raw15-comparison-audit.json). The timing receipt is `8547fe557e452db6e3927ad7077c067fee7748db2a4c7ecdebfea38778a8f37e`. Source snapshots, correctness receipts, diagnostic checks and host observations accompany this report. Complete PTX/cubin/SASS and preparation artifacts remain in the configured durable research store; their source-bound identities are retained here.

Export archive SHA256: `e0a10ece12dc0f1276cc9f8a08603c66ee48a222ae5b77df4dc1842e346489a2`. All 219 exported file hashes were verified locally.
