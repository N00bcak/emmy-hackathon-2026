# P3 sibling-loop fusion: final evidence

Status: **accepted and integrated locally in the main worktree**.

The P3 mechanism is a generic, typed-IR sibling-dot fusion with a final-SASS-verified
effect at `a01_attention_512x64`. The original candidate regressed
`a01_attention_256x64`; a structural guard admitting only a complete per-thread
configuration batch makes that path byte-identical to accepted P2. On the frozen
candidate, 256 is unresolved/neutral and 512 is a confirmed 8.54% faster on the
hosted RTX 5090. Both scaled cases pass full-output correctness.

The main checkout now contains the frozen source and test files below. The P3
worktree was used for the original checks; the merged tree was independently
rebound to the same source identity and rerun.

## Frozen identities

| object | SHA-256 |
| --- | --- |
| candidate compiler `compile.py` | `1f220a4dabcc436c49ab933bcc505e0e4d6ad1e04aef347a14a0b6cbd2fd0d1d` |
| candidate focused tests `tests/test_compiler.py` | `d8ce01b75d73db8ec4a1a3422e0398d43cc335a9ebbb7d3a61470cac3360e64c` |
| accepted P2 baseline `compile.py` | `0a6f12033a942f4da9176ebf1e963c16193d8883610d3d4807bbc9870ab0de2a` |
| core65 policy `tests/gpu-regression.json` | `1822bb1dd5bc30bb1bd824b424d36004402cda92216a921f8a27ef5dc0c21f54` |
| core65 case list | `31c2ad5d28ad7a2dc9161e8a2f5f3c035bff0ff50ec915a0a4ca0c73a47df0d4` |
| named guards list | `a7126f3922283c130d57452a2a88e6550d76c5efe6c9516fc0b47d1fd7ff2758` |
| scaled full-output receipt | `ef4cbc82b7896f323a2455d4b90fb90aa102939d6998a7cde55116ebf15ee7df` |
| SASS summary | `d4585aa5f3c8e378ad06ab6a84f539630492b747b930b0f4e72811e15d1cfd18` |

Worktree: `/home/ada/Projects/emmy-p3-sibling-fusion`, branch
`compiler-design/p3-sibling-fusion`, based on accepted P2 (`0a6f1203`).
The submodule and `.venv` are read-only symlinks to the main checkout.

## Mechanism

Two generic P3 paths are implemented before the ordinary loop/hoist lowering:

1. **`sibling_dot_groups` / `emit_sibling_dot_group`** admits adjacent seeded dot
   loops with one structurally identical shared readonly operand. It emits one
   shared B load per inner coordinate and each sibling's own product/fold in its
   original source order.
2. **`sibling_work_batch` / `matmul_work_loop_batch`** admits the hoisted matmul
   work loop when the thread's work items differ only in the CTA-configuration
   coordinate. The shared B vector is loaded once per K step and feeds each
   independent accumulator set.

The final candidate also contains a structural guard on the work batch: only a
batch covering the thread's **complete configuration run** is admitted. A
partial batch duplicates outer work-loop control and extends live ranges without
removing the residual trip; it now fails closed to P2. This is based on the
typed work/config relation, not a fixture shape or name.

Admission/fail-closed conditions:
- statically identical start, step, and bound (loop extents must agree);
- no cross-iteration carried state except independent accumulators;
- no possibly-aliasing write to a fused read (whole-kernel written-buffer check);
- preserved per-element order for every accumulator (no reassociation);
- explicit live-value budget (`SIBLING_FUSION_MAX_VALUES = 32`,
  `SIBLING_WORK_BATCH_MAX_VALUES = 40`);
- complete aligned/config-only work run for the batched matmul path;
- existing KEY batching retained and preferred; fusion never replaces it;
- unsupported structure falls back to the untouched lowering.

The change is generic typed-IR structure analysis. It does not match attention
names, statement spellings from fixtures, or hardcoded shapes. Evaluator,
generator, reference, and tolerance files were not edited.

## Source changes

- `compile.py`: `sibling_dot_shape`, `sibling_dot_groups`,
  `emit_sibling_dot_group`, `sibling_work_batch`, `matmul_work_loop_batch`,
  the two explicit budgets, and the complete-run guard. Accepted P2 scalar-cache
  code is unchanged.
- `tests/test_compiler.py`: focused generated/structural tests for equal and
  unequal bounds, distinct operand identities, aliasing writes, independent
  accumulators, per-accumulator fold order, and fail-closed work batching.
- `docs/evidence/p3-sibling-fusion/`: this report and the evidence bundle.

## Final-SASS activation

Both arms were compiled from complete compiler directories (including
`tensor_core.py` and `tf32_emitter.py`) with identical flags:

```sh
python <arm>/compile.py <scaled-case>/k_matmul_softmax_reduce_b55365.ir.json > kernel.ptx
ptxas -arch=<sm_89|sm_120> kernel.ptx -o kernel.<arch>.cubin
cuobjdump -sass kernel.<arch>.cubin > kernel.<arch>.sass
```

Resource counts come from `ptxas -v` on the same PTX; its disassembly hashes were
checked identical to the primary cubin. All hashes and opcode counts are in
`sass-1f220a4d/sass-summary.json`; SASS/PTX/cubin files are retained per arm.

| shape | target | instructions | LDG | FFMA | IMAD | registers | spill store/load | final SASS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `a01_attention_256x64` | sm_89 | 3968 → 3968 | 897 → 897 | 658 → 658 | 856 → 856 | 168 → 168 | 0/0 → 0/0 | **byte-identical** |
| `a01_attention_256x64` | sm_120 | 4096 → 4096 | 897 → 897 | 658 → 658 | 827 → 827 | 184 → 184 | 0/0 → 0/0 | **byte-identical** |
| `a01_attention_512x64` | sm_89 | 7872 → 6248 | 1737 → 1282 | 1937 → 1137 | 1492 → 1295 | 255 → 128 | 12/20 → 0/0 | changed |
| `a01_attention_512x64` | sm_120 | 7744 → 6280 | 1737 → 1282 | 1937 → 1137 | 1461 → 1294 | 255 → 168 | 16/12 → 0/0 | changed |

PTX identities:
- 256 base/candidate: `44333c2098c5dcc1784bcf041359655fb9a3cb659d0ae9620f4e1ba5c090f675` (identical)
- 512 base: `b1e620f31ec310aa6f4f5bb1617b8bda13c898a14b539f48829cf06f7e0dddf6`
- 512 candidate: `63476e5e6c28886e2bcf54c7d43a9dcb11816831d183695403f601c41f88fe24`

Cubin identities (sha256):

| shape | target | baseline | candidate |
| --- | --- | --- | --- |
| 256 | sm_89 | `edc409adc452889593c1ab84c1439b2ac398bcafef2a2ab3038872e4f7c2256d` | same |
| 256 | sm_120 | `62d81f0a472dae296f452c2886587ca58e9a1f88526bd048c8e68292568cf23e` | same |
| 512 | sm_89 | `d91b6c609e76328bb534d493d878c8142dbb9ab71f9768a04e3ac1a0fa55bb7f` | `a0697b506c7ad8bb860be01d96a13573dbdbf676f9e9d608f274ea656f9ccd3b` |
| 512 | sm_120 | `d23a22d70229f5afc3caef19d222cac09fb90623b0f080e780a2e7024db7e395` | `2d19d4fe57e872fb4a14e7487f1ca6c85acb4b7c724742f3a1bc5ae42801b2f7` |

Final-SASS identities (sha256):

| shape | target | baseline | candidate |
| --- | --- | --- | --- |
| 256 | sm_89 | `b610406921c0d918e50cfbec3ad78acf878198c588dc3209fce2b5c3e7562fe9` | same |
| 256 | sm_120 | `ef1d3bca3e2d0b2c298e737f317c526a6494ba0c00f50299a346e77e4d51bd7c` | same |
| 512 | sm_89 | `90f8eab6666743f31edc9becdeb99354d7176c03489574a570b9aa16bf47b0ba` | `fd07ba488398946f09aa9c278e10a573e9d73f6fd9da07e253d6de5596487ff2` |
| 512 | sm_120 | `8a4674cd30eac2e82a0ce1a8497b677aec354fb5b57970e03357ff94b097488d` | `153d8a54591127df22d4d3cc38d3ac459dfbcb6ef9a5eff082d7f757c2f4ce90` |

The surviving sm_89 sequence is a shared `LDG.E.128` feeding multiple FFMA
chains, e.g. baseline `13 00 LDG.E.128 R16` followed by interleaved address
arithmetic and 128-bit loads, versus candidate `0460 LDG.E.128 R8, [R8.64]`
followed by `04e0 FFMA R27, R8, R57, R27`, `04f0 FFMA R40, R9, R57, R40`,
etc.: one four-wide load is reused across sibling accumulators. This is final
SASS, not merely PTX. The 256 SASS is byte-for-byte the P2 SASS.

Occupancy note: resource limits differ by target. The 512 baseline is at
255 registers with spills; the candidate uses 128 (sm_89) / 168 (sm_120)
registers with no stack frame or spills, while shared memory remains 16,544
bytes and the launch contract is unchanged. These are mechanism diagnostics,
not a performance claim. No occupancy conversion beyond the recorded
registers/shared-memory/resource-usage logs is asserted; the raw logs remain in
the artifact for audit.

## Correctness gates (frozen source)

All receipts below bind `compiler_sha256 = 1f220a4d...`, and all record
`source_unchanged`/`compiler_unchanged = true` unless noted.

- **Published all-15:** 15/15 pass at exact metadata tolerances.
  `.emmy/evidence/p3-sibling-fusion/published-gate-1f220a4d-20261027/results.json`
  (sha256 `cba6046aaa5a169f03c80d52f334f4b053635748509814498f790f81840f69b8`).
- **Core65:** 65/65 pass, 16 sanitizer-required cases pass
  `compute-sanitizer --tool memcheck`.
  `.emmy/evidence/p3-sibling-fusion/core65-1f220a4d-20261027b/results.json`
  (sha256 `786574ee42e3170bdd947a052f258f24bfb49380d0f911e3440fc4ac094f7227`).
- **Named P3 guards:** A57, A62, A71, A72, A75, A76, A79, A80, A81, A82, A83,
  A84, A85, and targeted A95 all pass. A71/A72/A85 preserve fold order and output
  ownership; A95 covers axis rebinding. Receipt:
  `.emmy/evidence/p3-sibling-fusion/named-guards-run-1f220a4d/results.json`
  (sha256 `8d4d7fab533798c3779962c11c740bb5ed30f6a66937af38058740740348894b`).
- **Scaled full-output correctness:** `a01_attention_256x64` (16,384 elements)
  and `a01_attention_512x64` (32,768 elements) pass, all outputs covered,
  repetitions=3. Receipt:
  `.emmy/evidence/p3-sibling-fusion/scaled-correctness-1f220a4d/results.json`
  (sha256 `09d6c09bea825d96bec981cc4809cd514203cb155e7dbe5ea9176251a201da4d`),
  `scale-gate.json` (sha256 `ef4cbc82...`). The accepted-P2 baseline also has a
  matching full-output receipt at
  `/home/ada/Projects/emmy-hackathon-2026/.emmy/research/performance-v1/hosted-p3p4/p2-scale-receipt-1f220a4d/scale-gate.json`
  (sha256 `6186b9e0df3890c3fa30ed266705799b7b74ce355cbcdfca38b2687515a031fb`).
- **CPU:** 56/56 compiler tests pass (`cpu-compiler-1f220a4d.log`, sha256
  `ba36890929dfbc9f7ddfe28caf25da1a6d9772b3564a61157b7c88f8f7e214a0`).
  182/182 non-sandbox repository tests pass (`cpu-nonsandbox-1f220a4d.log`,
  sha256 `57e7839e5197cb15bdd6b6a77d5b12c56fe4adfe1f7d49ca18a4a9640b83f63e`).
  The full discovery log runs 196 tests and has 9 errors, all in
  `tests/test_research_prepare.py` because this session lacks root +
  libseccomp; no compiler or P3 test fails. This is recorded as an environment
  limitation, not a pass.

The first core65 attempt used a relative `--out` and failed all cases at the
worker path-resolution boundary before executing GPU work; it is retained as
`core65-1f220a4d-20261027` for audit and is superseded by the absolute-path
run above.

Merged-tree rebinding ran from `/home/ada/Projects/emmy-hackathon-2026` against
the same `compile.py` SHA-256 and 56-test suite:

- CPU compiler suite: 56/56 pass.
- Published contracts: 15/15 pass. Receipt
  `.emmy/research/performance-v1/merged-p3-gates-20261027/published15/results.json`,
  SHA-256 `8cfd68f0d1552be6236916320257f3efa94270096f13f23d733acfe401f36048`.
- Routine policy: 65/65 pass with 16/16 sanitizer checks. Receipt
  `.emmy/research/performance-v1/merged-p3-gates-20261027/core65/results.json`,
  SHA-256 `23b1f1819cffae5839773fa3c148da236506bbf8fc6cb1d3a10e8217c1581a9f`.

## Hosted timing (RTX 5090)

Harness: `/home/ada/Projects/emmy-hackathon-2026/.emmy/harness/hosted_scaled.py`,
server `http://runner.riftstack.ai:48701`, `arch=sm_120`, warmup 10,
iterations 50, 7 repetitions, alternating order. Deadband was pre-registered
at 0.01 and was not changed.

| run | case | baseline | candidate | median ratio | verdict |
| --- | --- | --- | --- | ---: | --- |
| A/A | 256 | P2 `0a6f1203` | P2 `0a6f1203` | 1.000108 | unresolved (calibration) |
| A/A | 512 | P2 `0a6f1203` | P2 `0a6f1203` | 1.000000 | unresolved (calibration) |
| primary | 256 | P2 | `1f220a4d` | 1.000000 | **unresolved/neutral** |
| confirmation | 256 | P2 | `1f220a4d` | 0.999892 | **unresolved/neutral** |
| primary | 512 | P2 | `1f220a4d` | 0.914540 | **candidate-faster** |
| confirmation | 512 | P2 | `1f220a4d` | 0.914569 | **candidate-faster** |

A/A calibrations:
- 256: max absolute bias 0.000648; receipt sha256
  `b7d3437d28525422cc25db5501088aa9b702b19d1417a9789662d6e296081df4`.
- 512: max absolute bias 0.000176; receipt sha256
  `e2f92e07d239c93ee729f378c429d23774ae9b083fcffb1ee7d4cacff9e60e60`.

Primary/confirmation receipts:
- 256 primary `9bdce81eb72b8c577c55ee2d5b1370857d7fccc176bc60457bbf0740b120184d`;
  256 confirmation `673eecd3cbc728588e634926360801071d48ce9175fd1f49a9fc36dfaa8a85f2`.
- 512 primary `0c40074d2d5bfc6716f8b394557714920f67bbb4608738f65f5ae1ce0ef40e13`;
  512 confirmation `cd5e0d8173542b4f1dabcb536a543afce0e296130db690e648e13f4fd4ff3984`.

These are local-harness hosted-runner receipts, not evaluator-issued benchmark
awards. No hosted timing was run before final-SASS activation and correctness
passed.

## Verdict

- **P3 passes the no-regression gate as scoped to the two scaled attention
  primaries.** `a01_attention_512x64` is a confirmed faster result
  (0.9145 primary, 0.9146 confirmation, A/A 1.0000);
  `a01_attention_256x64` is neutral (1.0000 primary, 0.9999 confirmation) and
  its final SASS is byte-identical to accepted P2, so it carries no new risk.
- The original partial-batch regression is structurally separated, not tuned:
  only a complete per-thread configuration run is admitted; the 256 path now
  fails closed to the existing P2 lowering.
- **Disposition: accepted and integrated locally.** The compiler hash must be
  preserved as `1f220a4d...`; any source change invalidates every receipt above
  and requires regenerating SASS and all gates.
- Known limits: local GPU is RTX 4060 (`sm_89`) while timing is hosted RTX 5090
  (`sm_120`); full repository discovery has nine environment-only preparation
  errors; hidden-set correctness is unverified; this is not an evaluator-issued
  qualification.

## Reproduction commands

```sh
cd /home/ada/Projects/emmy-p3-sibling-fusion
.venv/bin/python -m unittest tests.test_compiler -v
.venv/bin/python -m unittest tests.test_compiler tests.test_local_runner tests.test_published \
  tests.test_regression_suite tests.test_research_calibrate tests.test_research_cases \
  tests.test_research_control tests.test_research_gpu tests.test_research_pipeline \
  tests.test_research_verify -v
.venv/bin/python tools/evaluate.py --compiler compile.py --server http://127.0.0.1:48702 \
  --out .emmy/evidence/p3-sibling-fusion/published-gate-1f220a4d-20261027
.venv/bin/python -m tools.bundled_regressions --suite core65 \
  --out .emmy/evidence/p3-sibling-fusion/catalog-core65-1f220a4d.json
.venv/bin/python -m tools.regression_suite \
  --catalog .emmy/evidence/p3-sibling-fusion/catalog-core65-1f220a4d.json \
  --out .emmy/evidence/p3-sibling-fusion/core65-case-list-1f220a4d.json
/home/ada/Projects/emmy-hackathon-2026/.venv/bin/python \
  /home/ada/Projects/emmy-hackathon-2026/.emmy/harness/local_gate.py \
  --source /home/ada/Projects/emmy-p3-sibling-fusion \
  --case-list /home/ada/Projects/emmy-p3-sibling-fusion/.emmy/evidence/p3-sibling-fusion/core65-case-list-1f220a4d.json \
  --out /home/ada/Projects/emmy-p3-sibling-fusion/.emmy/evidence/p3-sibling-fusion/core65-1f220a4d-20261027b \
  --cache /home/ada/Projects/emmy-hackathon-2026/.emmy/research/performance-v1/p2p5-core65-cache
```

Hosted timing used `hosted_scaled.py` for A/A, primary, and confirmation with
`--warmup 10 --iters 50 --repetitions 7`; A/A used accepted P2 for both arms.
All output directories are under `.emmy/evidence/p3-sibling-fusion/` except
the accepted-P2 scaled correctness receipt under the main checkout's
`.emmy/research/performance-v1/hosted-p3p4/p2-scale-receipt-1f220a4d/`.

## Artifact map

- `sass-1f220a4d/`: baseline/candidate compiler copies, PTX, cubins, SASS, ptxas
  logs, and `sass-summary.json`.
- `published-gate-1f220a4d-20261027/`: all-15 receipt and logs.
- `core65-1f220a4d-20261027b/`: audited core65 receipt, prepared cases, and
  sanitizer logs. The earlier relative-path attempt is retained separately.
- `named-guards-1f220a4d.json`, `named-guards-run-1f220a4d/`: selected A57/A62/
  A71/A72/A75/A76/A79-A85/A95 guards and results.
- `scaled-correctness-1f220a4d/`: both scaled full-output receipts and
  `scale-gate.json`.
- `hosted-*/`: A/A, primary, and confirmation timing receipts.
- `catalog-*.json`, `core65-case-list-*.json`: immutable case selections.
- `cpu-*.log`: CPU test logs, including the environment-only failure record.

## Forced-fallback SASS audit

The final activation audit compares the frozen candidate against a mechanically
identical diagnostic arm, `ab85845e77c989172a140a3c2d7508472d5daac5389d21da60462ca55215bab8`,
which differs only by `self.sibling_fusion_disabled = True` in
`Emitter.__init__`. Both arms include the same `tensor_core.py` and
`tf32_emitter.py`, and both use the same `ptxas -arch=<arch>` and
`cuobjdump -sass` commands. This is a forced-fallback audit, not a second
compiler deliverable.

| shape | target | insns | LDG | STG | FFMA | IMAD | regs | spills store/load | final SASS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | sm_89 | 3968→3968 | 897→897 | 8→8 | 658→658 | 856→856 | 168→168 | 0/0→0/0 | byte-identical |
| 256 | sm_120 | 4096→4096 | 897→897 | 8→8 | 658→658 | 827→827 | 184→184 | 0/0→0/0 | byte-identical |
| 512 | sm_89 | 7872→6248 | 1737→1282 | 8→8 | 1937→1137 | 1492→1295 | 255→128 | 12/20→0/0 | changed |
| 512 | sm_120 | 7744→6280 | 1737→1282 | 8→8 | 1937→1137 | 1461→1294 | 255→168 | 16/12→0/0 | changed |

At 512 the candidate sm_89 sequence `LDG.E.128 R8, [R8.64]` followed by
`FFMA R27, R8, R57, R27`, `FFMA R40, R9, R57, R40`, `FFMA R41, R10, R57, R41`,
and `FFMA R42, R11, R57, R42` shows one four-wide load feeding sibling
accumulators in final SASS. The fallback retains per-accumulator load/fold
sequences. Address-form counts also change on sm_89: IADD3 `1481→1348`, LOP3
`55→27`, IMAD `1492→1295`; on sm_120 IMAD is `1461→1294`. STG remains `8→8`.

This confirms the 512 effect survives ptxas on both targets and that the 256
path is structurally fail-closed (identical PTX, cubin, and SASS), rather than
erased by ptxas. Full hashes and artifacts are in
`forced-fallback-sass-summary.json`, `forced-fallback-README.md`, and
`forced-fallback-sass-per-arm/`.
