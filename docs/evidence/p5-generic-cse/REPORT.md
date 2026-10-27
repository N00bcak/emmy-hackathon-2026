# P5 generic CSE, LICM and DCE: deferred

## Outcome

P5 is **deferred with no compiler change**. The no-op inventory found no
remaining repeated or dead work that is both semantically safe for a general
typed-IR pass and plausibly large enough to trade against compile-time and
correctness risk. Existing scoped CSE/GVN and invariant hoisting remain intact.

This is a no-candidate screening result, not a rejected timing candidate and
not a timing win. No whole-function SSA optimizer or general pass was added, no
candidate or forced-fallback source was created, and no new timing was run. In
particular, this report makes no hosted RTX 5090 or other performance claim.

The hard activation rule applies: an optimization is activated only if a
candidate compile and an otherwise identical forced-fallback/baseline compile
both reach final SASS on every timing target, and the candidate SASS shows the
specific surviving instruction, load, address sequence or live-register change
with matching hashes and resource counts. Because no P5 candidate exists here,
there is no activation claim, no timing authorization and no merge candidate.

- Worktree: `/home/ada/Projects/emmy-p5-generic-cse`
- Branch: `compiler-design/p5-generic-cse`
- Base: `d8f65f002db6c14d25c805255f7e7e6400329696`
- `compile.py` SHA-256: `067ef9c4ce31a4bf5ccda90a8d13e9bd1b71b89aff19abe7598cdba4c77e2921`

### Post-P2/P3 source check

The inventory above is bound to `067ef9c4...`, before P2 and P3 were integrated.
A complete current-source audit is now recorded in
[current-source-sass-inventory.json](current-source-sass-inventory.json)
(SHA-256 `05432afbff193c6ac985d7ee4629e037cb26b9eae9abaa4ce0c872363ee1bdec`).
It covers all 14 scaled P5 plan shapes---both `perf-a` and `perf-b`---on the
integrated P2+P3 source `1f220a4d...`, with final SASS for `sm_89` and
`sm_120`, instruction and opcode counts, registers, spill bytes and artifact
hashes.

Twelve shapes are unchanged from the pre-P2 inventory. `f01`, `f03`, `r04`,
`p02`, `p03` and `m01` at both tiers reproduce their prior final-SASS hashes
exactly. The two previously missing `f02` rows were added:

| Case | Target | Instructions | LDG | STG | Registers | Spills |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `perf-a/f02_row_softmax_3072x2048` | `sm_89` | 688 | 16 | 16 | 40 | 0 |
| `perf-a/f02_row_softmax_3072x2048` | `sm_120` | 720 | 16 | 16 | 40 | 0 |
| `perf-b/f02_row_softmax_4096x4096` | `sm_89` | 1120 | 48 | 1 | 40 | 0 |
| `perf-b/f02_row_softmax_4096x4096` | `sm_120` | 1160 | 48 | 1 | 40 | 0 |

The `perf-a` result is P2's scalar row cache firing; the `perf-b` result is P2
failing closed at `ROW_CACHE_VALUES=32`. Neither creates a new P5 candidate:
the retained-register map contains no repeated or dead invariant work that a
general typed-IR CSE/LICM/DCE pass could legally remove, and the fail-closed
map still handles row values one at a time. This closes the earlier f02
inventory gap. It does not establish P5 activation or authorize timing, because
P5 still has no candidate/fallback pair or measured dynamic opportunity.

## No-op SASS inventory

The authoritative inventory is [sass-loop-inventory.json](sass-loop-inventory.json)
(SHA-256 `bd7e108063b3d73b3417e272658b2c5a1fe7eaa7a40a18ed0b498ae1599ab324`).
The post-P2/P3 replacement is
[current-source-sass-inventory.json](current-source-sass-inventory.json)
(SHA-256 `05432afbff193c6ac985d7ee4629e037cb26b9eae9abaa4ce0c872363ee1bdec`),
which supersedes it for the integrated tree and includes f02.
The coarser PTX-level screen [sass-noop-inventory.json](sass-noop-inventory.json)
is retained only as an audit trail of candidates that ptxas later erased; it is
not activation evidence and does not authorize timing. All claims below come
from the final SASS inventory. The artifacts were produced from the frozen
`compile.py` above, assembled for `sm_89` with `ptxas -arch=sm_89 -v`, and
disassembled with `nvdisasm -c`.
Raw PTX, cubin and SASS artifacts and their case manifests remain under
`.emmy/research/performance-v1/p5-current-source-inventory-20261027/`; they are
local research artifacts, not part of this report or a merge candidate.

Final-SASS totals below are for one kernel invocation. "Memory" is
`LDG/STG/LDS/STS`; registers and spills come from the recorded ptxas verbose
output. Loop entries are `instructions; LDG; IMAD; IADD3` for the hot loop.

| Probe | SASS insns | Regs | Spills | Memory | Hot-loop body |
|---|---:|---:|---:|---|---|
| `p02_scale_shift_8m` | 48 | 16 | 0 | 7/5/0/0 | none |
| `p02_scale_shift_16m` | 48 | 16 | 0 | 7/5/0/0 | none |
| `p03_relu_mul_4m` | 56 | 20 | 0 | 10/5/0/0 | none |
| `p03_relu_mul_16m` | 56 | 20 | 0 | 10/5/0/0 | none |
| `f01_rmsnorm_3072x2048` | 536 | 48 | 0 | 59/5/5/3 | 135; 32; 32; 33 |
| `f01_rmsnorm_2048x4096` | 624 | 80 | 0 | 67/9/5/3 | 135; 32; 32; 33 |
| `f03_l2norm_3072x2048` | 488 | 40 | 0 | 52/5/5/3 | 135; 32; 32; 33 |
| `f03_l2norm_2048x4096` | 544 | 48 | 0 | 56/9/5/3 | 135; 32; 32; 33 |
| `r04_row_sum_3072x4096` | 376 | 40 | 0 | 55/1/5/3 | 135; 32; 32; 33 |
| `r04_row_sum_4096x4096` | 376 | 40 | 0 | 55/1/5/3 | 135; 32; 32; 33 |
| `m01_matmul_1024x1024x512` | 2632 | 94 | 0 | 0/5/242/8 | unrolled staged pipeline |
| `m01_matmul_2048x2048x512` | 2632 | 94 | 0 | 0/5/242/8 | unrolled staged pipeline |

### Per-shape final-artifact identities

Each row is one exact scaled shape. `IR`, `PTX`, `cubin` and `SASS` are SHA-256
identities of the frozen input and the actual compilation products. These are
baseline artifacts, not a candidate/fallback comparison.

| Probe | IR | PTX | cubin | SASS |
|---|---|---|---|---|
| `f01_rmsnorm_3072x2048` | `db9cbbb4…` | `fb03da81…` | `c48b2241…` | `9e4e169a…` |
| `f01_rmsnorm_2048x4096` | `7085df97…` | `d4f99862…` | `8b9b7aef…` | `f46e6031…` |
| `f03_l2norm_3072x2048` | `a70e21a7…` | `00ef91df…` | `920e1c23…` | `00599292…` |
| `f03_l2norm_2048x4096` | `67a3d2f7…` | `faee8f9b…` | `33c6d7ef…` | `8cbc6d36…` |
| `r04_row_sum_3072x4096` | `31aeec48…` | `ef24fd87…` | `e9c5f8ef…` | `b83d6315…` |
| `r04_row_sum_4096x4096` | `28692954…` | `a6428a5d…` | `e9c5f8ef…` | `b83d6315…` |
| `p02_scale_shift_8m` | `e72b5ed6…` | `fadb770a…` | `76477660…` | `2f2367ed…` |
| `p02_scale_shift_16m` | `2c2992f1…` | `a183da21…` | `56da410a…` | `4b93e12f…` |
| `p03_relu_mul_4m` | `cd262f51…` | `611a44bd…` | `265d1778…` | `76a003e5…` |
| `p03_relu_mul_16m` | `c7566c57…` | `35e667a9…` | `41cf212a…` | `44f41c18…` |
| `m01_matmul_1024x1024x512` | `23b80966…` | `b366b7bf…` | `8752f0a6…` | `b2287afb…` |
| `m01_matmul_2048x2048x512` | `983adb29…` | `7151cd89…` | `3b3bfc32…` | `5623abcc…` |

The full hashes, ptxas resource text and all kernels are in
`.emmy/research/performance-v1/p5-current-source-inventory-20261027/inventory.json`
and the per-shape directories beside it. Registers and spills differ per shape;
all twelve shapes report zero spill stores and zero spill loads.

### Two-target final-SASS audit

The same frozen PTX was assembled twice for each scaled shape with identical
ptxas flags (`ptxas -arch=<target> -v`) and disassembled with `nvdisasm -c`.
The complete per-target identities and opcode counts are in
[two-target-final-sass.json](two-target-final-sass.json)
(SHA-256 `8f47dbbdac66d5161b5dea6c7d51baab8a105d8bfadeda39593b75b4a8966f1b`).
Both targets assembled and disassembled successfully for all 12 shapes on the
local CUDA 13.4 toolchain. This is a no-op audit:
there is no candidate arm and no fallback arm to compare.

| Probe | sm_89 SASS / regs | sm_120 SASS / regs | sm_89 LDG/STG | sm_120 LDG/STG |
|---|---:|---:|---:|---:|
| `p02_scale_shift_8m` | 48 / 16 | 48 / 18 | 7/5 | 7/5 |
| `p02_scale_shift_16m` | 48 / 16 | 48 / 18 | 7/5 | 7/5 |
| `p03_relu_mul_4m` | 56 / 20 | 56 / 20 | 10/5 | 10/5 |
| `p03_relu_mul_16m` | 56 / 20 | 56 / 20 | 10/5 | 10/5 |
| `f01_rmsnorm_3072x2048` | 536 / 48 | 568 / 48 | 59/5 | 59/5 |
| `f01_rmsnorm_2048x4096` | 624 / 80 | 656 / 80 | 67/9 | 67/9 |
| `f03_l2norm_3072x2048` | 488 / 40 | 520 / 40 | 52/5 | 52/5 |
| `f03_l2norm_2048x4096` | 544 / 48 | 576 / 48 | 56/9 | 56/9 |
| `r04_row_sum_3072x4096` | 376 / 40 | 400 / 40 | 55/1 | 55/1 |
| `r04_row_sum_4096x4096` | 376 / 40 | 400 / 40 | 55/1 | 55/1 |
| `m01_matmul_1024x1024x512` | 2632 / 94 | 2688 / 98 | 0/5 | 0/5 |
| `m01_matmul_2048x2048x512` | 2632 / 94 | 2688 / 98 | 0/5 | 0/5 |

All 24 builds report zero spill stores and zero spill loads. On sm_120 the
pointwise kernels retain the same 128-bit load/store and four-FFMA structure,
but use descriptor-based global accesses and `LDC`/`LDCU` constants. The norm,
row-reduction and matmul kernels retain their loads, shared-memory traffic and
FFMA structure, with target-specific address forms (`IADD3`/`LEA` versus
`IMAD`). Those differences are final SASS, not PTX claims.

Driver-queryable maximum active blocks per SM was collected only on the local
sm_89 device using
the exact generated launch geometry; see
[driver-occupancy-sm89.json](driver-occupancy-sm89.json)
(SHA-256 `e22a490ddd632f7eb11b61ebfb5600db9192d9e215ccab6349e0ec94cfeac541`).
It records `cuOccupancyMaxActiveBlocksPerMultiprocessor` with the generated
block size and dynamic shared-memory value, plus registers, static
shared memory and launch geometry. It is resource feasibility only. No sm_120
device was available locally, so sm_120 occupancy is **unverified**, not
inferred.

### What survives, and why it was not actionable

- The pointwise probes contain repeated PTX constant/address expressions, but
  final SASS folds them. `p02` has one constant load and four `FFMA`; `p03`
  has four `FMNMX`/`FMUL` pairs. The coarse PTX repeat counts therefore do not
  identify surviving machine work, and no dynamic mechanism predicts that a
  generic pass would leave a useful instruction reduction in these kernels.
  Final p02 SASS shows a single `R8` data-wide address calculation, a single
  `R12` output-wide address calculation, one 128-bit vector load/store pair and
  four `FFMA`. The repeated PTX `mad/mul/add` address chains do not survive.
- In `f01`, `f03` and `r04`, the 32-iteration map bodies issue 32 loads plus
  32-33 address instructions per trip. Those addresses depend on the loop
  induction value; they are not repeated invariant expressions. The reusable
  row/base calculation is already represented outside the body. Hoisting an
  address across the induction update would violate `A95`'s exact-index
  requirement and is not a legal LICM transformation.
  The final `f01` body shows `LEA` plus `IADD3` offsets followed by
  `IMAD.WIDE` and a load, so the address arithmetic is part of the live
  loop-varying address sequence; removing it would not be a CSE/LICM win.
- `m01` uses the existing specialized unrolled staged pipeline with immediate
  ring offsets and shared-memory staging. No repeated or dead invariant index
  computation remained to justify a generic pass over that plan.
- No candidate could be justified on load-to-store visibility: a general DCE or
  LICM pass would need stronger proofs around grouped-load snapshots, carried
  updates, shared-memory writes and observable padding. The existing scoped
  passes already encode those restrictions (`A50`-`A52`, `A55`, `A68`).

## Verification

All checks below used the unchanged compiler identity above. The local proxy
host was an NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9), driver
615.78.08, with ptxas from CUDA 13.4 (V13.4.92) and Python 3.14.8. CPU tests were run
with the main checkout's read-only Python environment and a temporary link to
the real fixture submodule; that link was removed afterwards, leaving the
worktree clean.

| Check | Command | Result |
|---|---|---|
| CPU compiler suite | `PYTHONPATH=/home/ada/Projects/emmy-hackathon-2026/emmy-hackathon-2026 /home/ada/Projects/emmy-hackathon-2026/.venv/bin/python -m unittest tests.test_compiler -v` | 44/44 pass |
| Published gate plus reordered ABI | `flock -n /home/ada/Projects/emmy-hackathon-2026/.emmy/research/.locks/gpu.lock .venv/bin/python tools/check_local.py --compiler compile.py --out .emmy/p5-inventory/published-gate-20261027` | 15/15 published pass; reordered ABI pass |
| Routine core65 | `.venv/bin/python .emmy/p5-inventory/run_cases.py --compiler compile.py --out .emmy/p5-inventory/core65-baseline-20261027 --suite core65 --repetitions 2` | 65/65 pass |
| Isolated named sentinels | `.venv/bin/python .emmy/p5-inventory/run_cases_isolated.py --compiler compile.py --out .emmy/p5-inventory/isolated-sentinels-20261027 --suite core65 --case <id> ... --repetitions 2` | 6/6 pass: A57, A68, A79, A83-A85 |
| Targeted boundary cases | `.venv/bin/python .emmy/p5-inventory/run_cases_isolated.py --compiler compile.py --out .emmy/p5-inventory/isolated-targeted-20261027 --suite campaign100 --case <id> ... --repetitions 3` | 6/6 pass: A26-A29, A55, A95 |
| Integrated sentinels on current source | `.emmy/p5-inventory/run_cases_isolated.py --compiler /home/ada/Projects/emmy-hackathon-2026/compile.py --out .emmy/p5-inventory/integrated-1f220a4d-sentinels-campaign100 --suite campaign100 --repetitions 2 --case <19 IDs>` | 19/19 pass: A26-A32, A40, A50-A52, A55, A57, A68, A79, A83-A85, A95 |
| Compute Sanitizer memcheck | `.venv/bin/python .emmy/p5-inventory/run_sanitizer.py` | 16/16 pass; every report `ERROR SUMMARY: 0 errors` |

Full discovery (`.venv/bin/python -m unittest discover -s tests -v`) ran 184
tests with 9 errors, all in `test_research_prepare` because this environment is
not effective UID 0 and lacks its required compile sandbox. The compiler tests
themselves pass; the log is `.emmy/p5-inventory/unittest-baseline.log`.

Frozen result identities (SHA-256):

- Published gate results: `733ff72ee365a20f52a3e1af89f20853a7e71a8dfb0b62e5718191f39ef70be5`
- Core65 results: `4de89ffce1ebde158950c157ed18fbe1e36bac2b2b7e4eb9efd93091b506606f`
- Isolated sentinels: `126158551a8b8908ef6ac6e1716521ce4927d8e27304c330da6570d3968754cd`
- Targeted cases: `1c9d1e97d0b742ad08d0098685f6da6d8cd8ef6de5e1f966483bb0b233d03e62`
- Integrated sentinels on `1f220a4d`: `8e63378d2ad24ef8a5a0e8cc35b07bf4cce09bc78fc8d92cd8b43e1aecc95eb9`
- Sanitizer results: `99f8f2cb38bf6acb87f5f64beda6a837ca07c73e2f0550811c3a5a2657f178c0`

The pre-P2 sentinel files are isolated, one case per job and repetition, under
`.emmy/p5-inventory/isolated-sentinels-20261027/` and
`.emmy/p5-inventory/isolated-targeted-20261027/`. The sentinel cases were also
rerun against the integrated P2+P3 source `1f220a4d`; the checked-in receipt is
[integrated-sentinels-1f220a4d.json](integrated-sentinels-1f220a4d.json), and
its artifacts are under
`.emmy/p5-inventory/integrated-1f220a4d-sentinels-campaign100/`. All sentinel
evidence is correctness evidence only and was not timed.

## Legality boundary

Any future P5 work must preserve floating-point operand order, exact index and
transport semantics, output membership and load-to-store visibility. It must
not reassociate to manufacture a common subexpression, and it must not hoist an
address across a loop-axis rebinding. The named classes in the P5 plan
are covered as follows: cache keys by A26-A29; scopes and lifetimes by A30-A32
and A40; load-name and snapshot behavior by A50-A52 and A55; attention
normalization ownership by A57, A79 and A83-A85; observable stores by A68; and
rebinding safety by A95.

## Hard-gate disposition

- All measurements and SASS evidence here are for local `sm_89`/RTX 4060-class
  execution. They are a proxy, not the coordinator's hosted RTX 5090 result.
- The inventory is static mechanism evidence; it does not prove a dynamic
  speed effect and cannot support a performance promotion.
- The current-source inventory is baseline-only for P5. There is no P5
  candidate or forced-fallback pair, so candidate-vs-fallback activation and
  timing are not applicable rather than satisfied by static counts.
- Hidden-set correctness and non-local GPU behavior remain unverified.
- Recommend keeping P5 deferred and merging only this evidence record (not a
  compiler pass). Reopen only if a new typed-IR mechanism creates a falsifiable
  dynamic prediction, with a fresh source-bound gate and a noise-calibrated
  paired timing run.
- No timing was run because the prerequisite final-SASS activation comparison
  does not exist. This is not a claim that timing was neutral or resolved.
- The two-target audit proves assembly and final-SASS structure, not execution
  on sm_120 hardware: only local sm_89 execution and driver occupancy were
  available. The sm_120 artifacts establish assembly/final-SASS structure only.

## Next action

Retain the existing scoped optimization pipeline, keep P5 deferred, and do not
schedule P5 timing. The current-source inventory now covers all 14 plan shapes;
reopen P5 only if a new typed-IR mechanism creates a falsifiable dynamic
prediction, with a fresh source-bound gate and a noise-calibrated paired timing
run. If P3 or P4 is merged and changes this source identity, rebind the
inventory to that new source before reconsidering a general pass.
