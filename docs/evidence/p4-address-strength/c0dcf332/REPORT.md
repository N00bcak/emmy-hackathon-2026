# P4-on-P3 address strength: final SASS active, timing unresolved

## Disposition

**No merge.** Keep P4 unmerged in `compiler-design/p4-on-p3`.

The one-shift candidate is a real final-SASS effect, not a PTX-only effect:
`ptxas` removes one `SHF.L.U32` and changes the following byte-scale
`IMAD.WIDE` immediate on both sm_89 and sm_120. It preserves the generic
fallback, correctness, ABI ownership, exact indices, LDG/STG counts and
register use. However, a fresh preregistered hosted p02 primary is unresolved:
median candidate/baseline ratio **1.002463** against deadband **0.052509**.
The stronger immediate-displacement form was rejected; it did not produce a
generic primary-path SASS win and was not allowed to authorize timing.

Base source is the integrated P3 main tree at `1f220a4dabcc436c49ab933bcc505e0e4d6ad1e04aef347a14a0b6cbd2fd0d1d`.
The isolated worktree is `/home/ada/Projects/emmy-p4-p3-rebased`, branch
`compiler-design/p4-on-p3`; the main tree was not modified.

## Frozen identities

| Artifact | SHA-256 |
| --- | --- |
| candidate `compile.py` | `c0dcf3321d209ef5de1205a8e592890810a155c6b9aa687abb38b434b88d62e7` |
| candidate `tests/test_compiler.py` | `1ede17e49923ba1625e48ebc8ff85817444fb22f313f69d2022316627a03ff2c` |
| P3 baseline `compile.py` | `1f220a4dabcc436c49ab933bcc505e0e4d6ad1e04aef347a14a0b6cbd2fd0d1d` |
| forced-fallback `compile.py` | `488e135d211ea4661f60d344c888076caa5150aa4d41e7fc0ea99e68b12f6654` |
| local correctness receipt | `52283c3e8ddd13ea1cc90b524b6ab4bf581d9ea06fb0f6e1e86d671ba4c7d229` |

The only candidate/fallback source difference is the inserted `return None`
at the top of `affine_byte_scale`, which forces every address through the
pre-P4 generic lowering. No case names, shapes, or ABI assumptions were added.

## What survives ptxas

Both arms were compiled with the identical command:

```text
ptxas -arch=<sm_89|sm_120> -v <kernel.ptx> -o <module.cubin>
cuobjdump -sass <module.cubin>
```

The relevant final-SASS delta is exactly the affine vector address:

```text
fallback: SHF.L.U32 R0, R0, 0x2, RZ ;  MOV R13, 0x4 ;  IMAD.WIDE ..., R13, base
candidate:                            MOV R13, 0x10 ; IMAD.WIDE ..., R13, base
```

Both targets preserve the intended direct affine scale: `ptxas` folds the
element shift into the byte-scale immediate and does not re-materialize it.
Non-NOP counts
are unchanged except for p03 sm_89, where the shift removal also shortens
the emitted body by one instruction. Inactive trailing `NOP` padding differs
between arms, so both the total decoded opcodes and non-NOP counts are listed:

| Case | Target | SASS SHA-256 | Opcodes | Non-NOP insns | LDG | STG | IMAD | SHF | Regs | Spill bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p02 candidate | sm_89 | `e8f55943a983bb7e71a1db3982011f0aa3b417fab9413b942d6c410564fde36e` | 48 | 35 | 7 | 5 | 3 | 0 | 16 | 0 |
| p02 fallback | sm_89 | `36e2cb0d3fdfe5af8f7c6c4bc75077669ff6a7b05fe6958b89dd2a5447ff32da` | 48 | 36 | 7 | 5 | 3 | 1 | 16 | 0 |
| p02 candidate | sm_120 | `ec833ef6327a378f42f07120557a32663e90c13b1c78aefcf4abc18f3c5af085` | 48 | 35 | 7 | 5 | 3 | 0 | 18 | 0 |
| p02 fallback | sm_120 | `315f7e239754624aedf9b69456271fc360d50c17dad32d181586c09801ea1fd0` | 48 | 36 | 7 | 5 | 3 | 1 | 18 | 0 |
| p03 candidate | sm_89 | `8fe5309a5ceaa3308bc915acf14c27696334a400e2456055f31eed22becc1b9f` | 48 | 40 | 10 | 5 | 4 | 0 | 20 | 0 |
| p03 fallback | sm_89 | `3c3c6b65d587564e8acfa40c80bb03669e9b3793011cc6ba59164aedb4161aaa` | 56 | 41 | 10 | 5 | 4 | 1 | 20 | 0 |
| p03 candidate | sm_120 | `620f26b1cf2b8f67f755f07df3e9c2a1b254f338810a787b8496a33548ea26f0` | 56 | 43 | 10 | 5 | 4 | 0 | 20 | 0 |
| p03 fallback | sm_120 | `bc13694ab3735a769fcfa437f409b087f61f4254f83e041d5a60d4d45769330c` | 56 | 41 | 10 | 5 | 4 | 1 | 20 | 0 |

Candidate PTX hashes are `0153cd1bce858b7c6730cfadc56da14c0e1cc96b9b703f0eaf033d0ed989ca79`
(p02) and `4f56bdc8dc7408c0821c44ecb1aa9f4ef347f11f03aa35232600c9c55cef25a0`
(p03). Both targets have 0 spill stores and 0 spill loads. Exact cubins,
disassemblies and `ptxas -v` logs are in [sass](sass).

An independent rebuild of the stored candidate and fallback PTX is recorded in
[p2-p4-sass-recheck-20261027.json](../../p2-p4-sass-recheck-20261027.json)
(SHA-256 `a0c7dc94023f47436cd18ec305df7dc6671fc771177fa83971cf7b7a8a2f728e`).
It reproduces every P4 SASS hash and count above, and confirms that f01, f03
and r04 are byte-identical between candidate and fallback on both targets.

The f01-only manual displacement PoC did produce
`LDG.E.128 [base+0x800]`, `[base+0x1000]` and `[base+0x1800]` (552 vs 568
instructions), but it was hand-shaped PTX. p02 and p03 each have only one
vector group per buffer, so the mechanism cannot reduce instructions there.
It therefore remains a rejected non-generic experiment, not P4 evidence; see
[sass/f01_disp.ptx](sass/f01_disp.ptx).

## Correctness bound to `c0dcf332`

- CPU compiler tests: **60/60 OK**.
- Published contracts: **15/15 PASS** at exact metadata tolerances; the
  reordered-ABI check also passes.
- Audited core65: **65/65 cases, 67 kernels, 16 sanitizer-required cases**;
  compile summary has no failures and runtime `all_passed` is true.
- Ten fail-closed guards `A24 A25 A30 A31 A50 A51 A58 A59 A90 A91`: **PASS**.
- `A95` and Compute Sanitizer: **PASS**.
- Scaled full-output checks, 3 repetitions each: p02 8,000,000 elements and
  p03 4,000,000 elements with **0 mismatches**; f01, f03 and r04 also pass.

Receipts: [gates/published.log](gates/published.log),
[gates/cpu-test-compiler.log](gates/cpu-test-compiler.log),
[gates/compile-summary.json](gates/compile-summary.json),
[gates/results.json](gates/results.json),
[gates/guards.json](gates/guards.json),
[gates/a95.json](gates/a95.json),
[gates/a95-sanitizer.json](gates/a95-sanitizer.json), and
[gates/local_correctness_receipt.json](gates/local_correctness_receipt.json).

The core65 compile receipt binds `compiler_sha256` to `c0dcf332`; the local
hosted correctness receipt likewise binds `c0dcf332`, has
`source_unchanged: true`, and marks all five scaled cases
`full_outputs_pass: true`. The earlier top-level run `results.json` was
replaced by the scaled run; the 65 per-case result files remain in the run
root, and the derived compile receipt and run log are preserved here.

## Hosted RTX 5090 timing

Protocol: frozen p02 scaled case, one fresh candidate-bound A/A, then exactly
one primary against P3 with the preregistered p02 deadband. Warmup 10,
50 iterations, 7 alternating blocks; correctness is from the local full-output
receipt. No confirmation was run because the primary did not resolve faster.

| Run | Comparison | Median ratio | Deadband | Verdict | Receipt SHA-256 |
| --- | --- | ---: | ---: | --- | --- |
| A/A | `c0dcf332` vs `c0dcf332` | 0.986634 | 0.121636 observed max bias | unresolved | `8cfc695f882df6cb5c122bf2a9173793f9e6c508960c13826cc4eca004a23b84` |
| Primary | `c0dcf332` vs P3 `1f220a4d` | 1.002463 | 0.052509 preregistered | unresolved | `b4f1f216562b56f9d0315636e034c42b4fd526e678b46824e33f023f245afb1f` |

The A/A was noisier than the preregistered deadband (one pair was 12.16% off),
while the primary spread was 0.939 to 1.133. The primary therefore supplies
no resolved win at this scale; the mechanism's SASS delta is real but too small
to establish a speedup. Raw receipts are
[p02-aa-result.json](p02-aa-result.json) and
[p02-primary-result.json](p02-primary-result.json). Each used 16 HTTP requests
and `max_return=0`; no hosted process remains live.

The earlier `p02-aa` receipt in the research root is historical and not used:
both its arms were P3 (`1f220a4d`), so it did not bind the candidate.

## Merge recommendation

**Do not merge P4 into main.** The one-shift lowering is correct and survives
`ptxas` with a genuine `SHF` removal on both targets, but it has no resolved
hosted timing win and the stronger generic displacement form was rejected.
Reopen only with a mechanism that reduces dynamic work on the scaled primary
path and can clear the preregistered hosted deadband.
