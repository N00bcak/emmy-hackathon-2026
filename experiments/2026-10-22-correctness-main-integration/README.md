# Correctness repair and main integration

The promoted compiler is installed at root `compile.py`. Branch `n00bcak` was fast-forwarded from `3f1b749` to newest checked main `83a394490fbc232e0a0aefae8e5211657460dd6d` after correctness completion. Main was rechecked immediately before installation. No new commit or publication was made. The installer preserved all nine existing untracked files and backed up the corrected pre-main compiler as `corrected-before-main.py`; the earlier original compiler remains `compile-before-correctness.py`.

Compiler SHA256: `ef5e7fe0092f95c86a4effa628c533c6c3ec2e4d16802035ed31073e27d88d6a`. Immutable source: `d883119296f923cc179e64220d8fc3937930ec1cafe15a8e14e2693325f04aed`. Promoted decision: `3eb1c7608c12f81374953108b661428a6885bdaf02ef70980d7f15f6c51a21f2`. Installed maintained-test SHA256: `291595a668e0e2f33bb1fe37b2c06348819636a4be5c8edbd6356202e35f0e02`.

## Verification

- All **15 published cases** passed at their exact metadata tolerances.
- **894 distinct frozen GPU program/input cases** passed: 15 published and 879 generated. The full preselection replay contained862; fresh postselection confirmation added32 unseen generated cases alongside15 repeated published controls. See `correctness-per-case.csv` and `case-union.json`.
- Guard and input checks, isolated contexts and required Compute Sanitizer checks passed. Every GPU job has an admitted, observed-no-contention host record. Exact case/input/reference bytes and all IR/PTX/cubin/SASS remain in the configured durable research store.
- All32 maintained compiler tests passed in the configured Linux runtime on these exact compiler/test bytes. Both-target static assembly checks passed; both late loop repairs left all15 published PTX and all30 sm89/sm120 SASS/resource artifacts unchanged relative to the reviewed integration parent.
- Final performance confirmation measured all47 core cases,32 paired samples and16 same-job A/A pairs per case, after source selection and a fresh freeze/reservation. **All47 were inconclusive: zero resolved wins and zero resolved regressions. No new latency gain is claimed.**

GPU execution used the NVIDIA GeForce RTX4090 (sm89), driver API/cubins and CUDA device-event intervals. Compilation, container/process startup, artifact verification and sanitizer time explain much of the wall time and are excluded from kernel intervals. Measurements reuse warm allocations; very short kernels have broad noise bands. sm120 was assembled and inspected statically; no5090 runtime or hosted calibration was used.

## Teammate review: veracity and impact

The attached document was treated as review evidence, with claims checked against the actual frozen compiler rather than accepted as instructions. Its line numbers did not establish an exact source revision.

| Claim | Finding and action |
|---|---|
| Partial maximum pads with zero | Not reproduced on the audited original `a826...` source: the127-element negative maximum returned−1. Inactive-lane identities, explicit/default seeds and tails were independently tested. The finite default seeds specified by the primary IR remain part of its contract. |
| Scalar-by-value ABI fails | Valid support gap. Added positional metadata-driven scalar binding, runtime values and widths, unused arguments and independent buffer/SSA namespaces. Metadata values are not embedded as fixture answers. |
| Launch geometry uses flat x only | Valid gap. Added full three-dimensional block/grid linearization and preserved private physical launch IDs across lexical shadows. |
| Natural log is base2 | Not reproduced on the audited original source, whose F32 natural log already included ln2 scaling. Math variants, casts, NaN/boolean handling and explicit unsupported domains were checked and repaired where valid. |
| CastExpr/Builtin/FuncCallExpr/FlatIndex absent | Valid support gaps. Added typed general lowering for the declared supported domains, with clean failures for unproved domains. |

Main integration also required native dtype, lexical scope, stable collective-carrier and SSA corrections. Two additional **GPU-reproduced** loop defects were repaired generally:

1. A mutable compound `StridedLoop.step` was captured on entry. The primary update clause re-evaluates it at each backedge. The old integration produced3/3 wrong outputs (max error15) with intact guards; a fresh sanitizer worker repeated the error.
2. A direct variable `StridedLoop.end` aliased a carrier later overwritten by a native I32 shuffle. The primary captures the end once. The old integration produced96/96 wrong outputs, also reproduced under sanitizer with zero memory errors.

The repaired source passed all20 independently frozen old/fresh loop probes, five isolated guarded repetitions and required sanitizer runs. Header dependence and lexical binding determine the lowering; no fixture names, values or shapes select answers. Own-induction-axis end scope remains explicitly unsupported because primary scope evidence is ambiguous. See `loop-header-findings.json` and `loop-header-confirmed20.json`.

## Performance and retained decisions

A previous integration candidate passed778 correctness cases but was rejected for paired ratios2.985 on m01 and1.819 on a02. Its results remain in `rejected-v10-per-case.csv`. General shared-vector/range lowering and Tile decoding were restored, then the exact final source was fully requalified.

The table below shows final independent warm measurements. Ratios are medians of paired ratios, so they need not equal ratios of separate latency medians. Noise bands are case-specific; raw medians alone do not establish a win or regression.

| Case | Baseline µs | Candidate µs | Median paired ratio | Noise band | Verdict |
|---|---:|---:|---:|---:|---|
| m01_matmul_512 | 26.624 | 25.600 | 0.9615 | 11.94% | inconclusive |
| m02_matmul_rect | 5.120 | 5.024 | 1.0000 | 60.45% | inconclusive |
| f02_row_softmax | 4.096 | 3.168 | 1.0000 | 100.00% | inconclusive |
| a01_attention | 199.680 | 200.704 | 1.0051 | 1.54% | inconclusive |
| a02_attention_128 | 53.280 | 53.408 | 1.0000 | 5.66% | inconclusive |

All47 measured cases, raw artifact IDs and host jobs are in `independent-per-case.csv`; the18-case screen is in `screen-per-case.csv`. The attention128 cubin/SASS matched the qualified baseline on both static targets. Matmul retained a different backend loop structure but showed no resolved latency regression. A speculative stronger unroll was therefore left unimplemented. A separately reviewed new unroll-hint oracle is retained for future work; no genuine cases or compiler candidate used it in this run.

## Evidence and limits

- Full862 correctness receipt: `16abf6be48dda88276337051b862e32c5919ab62f4b32cefc3d95f63f0259a41` (`full862-correctness-receipt.json`).
- Fresh47 correctness receipt: `79795ccd3503f7008bfb00cd901c10b8a5fb2398cd00c906c4dfa0c9c0e00873` (`fresh47-correctness-receipt.json`).
- Independent paired receipt: `e951e625fb9451da81fa1dfdfe144836887c4b23844066a12a0d62a38f95b5d7` (`independent47-timing-receipt.json`).
- Selection, source, inputs, references, commands, admission, failures, rejected prototypes and handoffs are retained under the privately configured research root, campaign `campaign-20261022-12h-v1/integration-main-83a3944`.

`limits.json` preserves explicit unsupported/unscored distinctions. Power remains unsupported; four extra main cases have unsupported or validator/reference-shape limitations, and14 ambiguous conditional-seed oracle cases remain quarantined and excluded. These are not counted as passes. The challenge submodule, fixtures, harness, tolerances and pinned commit were preserved. Root maintained tests were updated only to recognize valid direct-remainder hoisting while retaining the original structural checks.

See `HANDOFF.md` for resumable state and next research questions. Compiler and maintained-test changes remain uncommitted.

Run closed at 2026-10-22T18:17:08.206543+00:00: 28/200 variants, 10.698/12h. Discovery is off and all workers joined. See FINAL-RESULT.json and RESEARCH-HANDOFF.md.
