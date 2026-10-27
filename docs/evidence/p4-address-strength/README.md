# P4 address strength reduction

Evidence-only snapshot. The current P4 result is the P3-based candidate
`c0dcf332...` in [c0dcf332/REPORT.md](c0dcf332/REPORT.md): its one-shift
mechanism survives `ptxas` on both targets, all correctness gates pass, and a
fresh hosted p02 primary is **unresolved**. P4 is not merged. The older
`74164ed5...` material below is retained as the P2-based history.

The P3 rebase has now been gated and timed in the isolated worktree
`emmy-p4-p3-rebased`. Its report records CPU `60/60`, published `15/15`,
core65 `65/65` with 16 sanitizers, all named guards, and scaled full outputs.
The hosted A/A median was `0.986634` with 12.16% observed maximum bias; the
p02 primary median was `1.002463` against a `0.052509` preregistered deadband.
No confirmation was run because the primary did not resolve faster.

## Frozen identities

| Item | SHA-256 |
| --- | --- |
| Candidate `compile.py` | `74164ed51aff8fc3886e4b5daa029f526d26d4fe5e2d0cf86ac706ec1cfb602d` |
| Candidate `tests/test_compiler.py` | `9a6a2d2112a309250160eb636e5a3229d0b3a12f66baa89ae90d945ce3247f0a` |
| Accepted P2 baseline `compile.py` | `0a6f12033a942f4da9176ebf1e963c16193d8883610d3d4807bbc9870ab0de2a` |
| Bound local correctness receipt | `16fc19eaf31fdba2128df4404c0aa266960cc9e40140340213686c8a6bc8b883` |
| Pre-registered timing protocol | `fcfed74ae7a339e3691475335c02a78ff68715742cf2d231eba28025690dcdb0` |
| P02 hosted result | `a60f70b5fd4726e04af3631add5d5c24ac86bbd624006d89308001e764f1daff` |
| P03 hosted result | `a6b1388aef182d6b969598f5228a1001a60367479f0ce7a51edefef961efb39d` |

## Hosted RTX 5090 result

Both cases were fresh candidate-vs-baseline runs with the frozen sources and
the same bound local full-output receipt. Deadbands were pre-registered and
were not changed after measurement. No results were pooled.

| Case | Median ratio | Deadband | Direction | Confirmation |
| --- | --- | --- | --- | --- |
| `p02_scale_shift_8m` | `1.0` | `0.052508751458576364` | unresolved | not run |
| `p03_relu_mul_4m` | `1.0` | `0.06844741235392304` | unresolved | not run |

The P02 ratios were `0.8754027926960257, 1.1263803680981597,
1.1265206812652069, 0.8813376483279395, 0.9076751946607341, 1.0, 1.0`.
The P03 ratios were `1.0018083182640145, 0.996415770609319,
1.0036101083032491, 1.0, 1.0, 1.0, 1.0036101083032491`. Neither median
reaches its threshold, so no isolated confirmation was permitted.

## Activation and correctness

- Final SASS was independently verified on `sm_89` and `sm_120` for both
  scaled primaries. Candidate SASS removes the fallback `SHF.L.U32 ...,2`
  and uses `IMAD.WIDE` with byte scale `0x10` instead of `0x4`.
  LDG/STG counts, registers/spills, and occupancy are unchanged. See
  `sass-activation.json`, `counts.json`, and `sass/`.
  A coordinator rerun on the P3-rebased source reproduced the same p02 PTX and
  SASS hashes. p03 `sm_89` dropped 56->48 instructions with identical LDG/STG;
  p03 `sm_120` and p02 `sm_89`/`sm_120` remain 48/56 instructions with one
  fewer `SHF`. The named map follow-ons (`f01`, `f03`, `r04`) are byte-identical
  to their forced fallback on both targets, so P4 fires only on the pointwise
  primaries here.
- Bound local full-output correctness passed `p02_scale_shift_8m`
  (8,000,000 elements) and `p03_relu_mul_4m` (4,000,000 elements) with zero
  mismatches. See `local-correctness-receipt.json`.
- Published 15/15, routine 65/65 with 16 sanitizers, all ten guards
  (`A24 A25 A30 A31 A50 A51 A58 A59 A90 A91`), and targeted A95 plus its
  sanitizer passed. See `gates.json` and `artifacts/`.
- Local RTX 4060 values in `artifacts/paired-proxy.json` are proxy evidence
  only, not hosted acceptance.

## Disposition

Keep P4 unmerged: no primary produced a resolved faster result, so the
required confirmation stage was not reached. Neither primary shows a resolved
regression. The SASS activation remains valid mechanism evidence.

P4 is a separate source identity (`74164ed5...`) based on accepted P2
(`0a6f1203...`), not on the integrated P3 source (`1f220a4d...`). Merging P4
after P3 therefore requires a rebase, a fresh final-SASS activation audit, and
fresh correctness and timing receipts bound to the new source hash.

## Evidence files

- `plan.md`: pre-registration with frozen hashes, deadbands, and thresholds.
- `commands.md`: exact CPU, binding, and hosted commands actually used.
- `local-correctness-receipt.json`: bound full-output receipt.
- `p02-primary-result.json`, `p03-primary-result.json`: fresh hosted results.
- `sass-activation.json`, `counts.json`, `sass/`: final-SASS activation.
- `gates.json`: correctness gate identities and pass status.
- `artifacts/`: raw guards, A95, scaled correctness, published-15, proxy, and
  hosted PTX/compiler stderr.
- `SHA256SUMS`: hashes for every file in this directory.
