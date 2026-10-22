# Adversarial audit: TF32 tensor-core recognizer

Date: 2026-10-23. No compiler or test files were edited. All generated files are under experiments/tensor-core-audit/ or /tmp.

> **Historical record.** This report is bound to the pre-repair source
> `tensor_core.py` hash `375f3031...` and describes that earlier checker. Its
> confirmed false accepts were repaired and replayed; see
> [Post-audit disposition](#post-audit-disposition). Do not treat the pre-fix
> "actual" rows below as current checker behavior.

## Reproduce

    cd /home/ada/Projects/emmy-hackathon-2026
    .venv/bin/python experiments/tensor-core-audit/probes/run_audit.py
    .venv/bin/python experiments/tensor-core-audit/probes/a21_execute.py
    .venv/bin/python experiments/tensor-core-audit/probes/make_m01_copy_shift.py

## Critical findings

1. False Plan: CTA-dependent operand reads are not enumerated. a01, a02, a21 (typed direct) are all accepted as Plan. Fragment binding only evaluates bid=0. Execution of a21 gives canonical A@B (1.19e-7 vs direct model) but the typed moral result differs by 3.506757.
2. False Plan: output aliasing an input is accepted (a03 output==rhs, a04 output==lhs). Emitter stores through the first parameter from metadata arg_order.
3. False Plan: pipeline copies are neither proved nor emitted (a05 and m01 B-copy +1 shift). m01-shifted PTX hash equals unshifted tensor PTX 84e26af55cd3c696a891d583a04e13b20d48a2f349223f8140311e3de9f14b9b and contains 3 MMA, no cp.async replay.
4. Read bounds are missing (a02 reads row -1 and is Plan). cp.async.wait_group(3) on m01 is also accepted as Plan.

## Probe matrix

| id | expected | actual |
|---|---|---|
| a00-baseline-direct | Plan | Plan |
| a01-cta-row-offset | NoMatch | Plan |
| a02-cta-negative-offset | NoMatch | Plan |
| a03-output-aliases-rhs | NoMatch | Plan |
| a04-output-aliases-lhs | NoMatch | Plan |
| a05-pipeline-copy-mismatch | NoMatch | Plan |
| a06-pipeline-read-outside | NoMatch | NoMatch |
| a07-noop-epilogue | Plan | Plan |
| a08-accumulator-slot-swap | NoMatch | NoMatch |
| a09-shared-copy-swap | NoMatch | NoMatch |
| a10-launch-3-warps | UnsupportedMatch | UnsupportedMatch |
| a11-unused-load | Plan | Plan |
| a12-same-source | NoMatch | NoMatch |
| a13-k1 | UnsupportedMatch | UnsupportedMatch |
| a14-m8 | UnsupportedMatch | UnsupportedMatch |
| a15-typed-direct | Plan | Plan |
| a16-duplicate-output-cell | UnsupportedMatch | NoMatch |
| a17-write-oob | NoMatch | NoMatch |
| a18-nonzero-seed | NoMatch | NoMatch |
| a21-typed-cta-row-offset | NoMatch | Plan |
| m01-b-copy-col-shift | NoMatch | Plan |

## Suggested regression tests

- cta_dependent_read_offset: 32x8x8, 2 CTAs, A row cta_m+g; expect NoMatch.
- read_below_operand: CTA-dependent -1 row offset; expect NoMatch.
- output_aliases_input: rewrite Write.output to lhs/rhs; never Plan.
- pipeline_copy_source_shift / dest_shift / lane_index: reject unless stage and copy/read binding are proved and emitted.
- wait_group_depth: m01 wait_group(3) must be rejected until commit depth is proved.
- Keep a15 and redundant-affine/reordered valid forms as positive regressions.

## Files

All created files are under experiments/tensor-core-audit (plus /tmp scratch). See probes/run_audit.py, probes/mutate_helpers.py, probes/*/ cases, probes/a21_execute.py, probes/make_m01_copy_shift.py, run.log, compiler-hashes.txt. The machine-readable probe rows are embedded in `run.log`; this first audit did not write a separate `probes/results.json`.

## Post-audit disposition

This report remains bound to the pre-repair source hash recorded above. The
confirmed false accepts were repaired and the affected probes were replayed:

| id | pre-fix | post-fix | disposition |
| --- | --- | --- | --- |
| a01/a21 CTA-dependent operand drift | Plan | NoMatch | Rejected before endpoint sampling |
| a02 CTA-dependent negative offset | Plan | NoMatch | Rejected by the same CTA-coordinate proof |
| a03/a04 output/operand aliasing | Plan | NoMatch | Output alias check added to `MatmulPlan` |
| a05 pipeline copy mismatch | Plan | NoMatch | Copy producer/read binding proved; no unproved pipeline is emitted |

Valid direct and staged plans remain `Plan`. The false accepts are closed.
Permanent regressions are in `tests/test_tensor_core.py`. Post-repair source
identity: `tensor_core.py`
`ca5af9c9f41826f9e2000de7a8537745c0393007d8eaaaf9fd212b3bfd3364a6`.

The focused suite passed 10 tests when this historical audit ran with
`.venv/bin/python -m unittest tests.test_tensor_core -v`. The current suite has
13 tests and passes separately. This audit did not rerun the 15 published cases
or GPU execution; those are covered by the frozen evidence bundle. No
performance, promotion, TMA, or matrix-transport claim is made.
