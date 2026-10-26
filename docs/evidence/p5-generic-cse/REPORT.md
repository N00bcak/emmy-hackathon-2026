# P5 generic CSE, LICM and DCE: deferred

## Outcome

P5 is **deferred with no compiler change**. The no-op inventory found no
remaining repeated or dead work that is both semantically safe for a general
typed-IR pass and plausibly large enough to trade against compile-time and
correctness risk. Existing scoped CSE/GVN and invariant hoisting remain intact.

This is a no-candidate screening result, not a rejected timing candidate and
not a timing win. No whole-function SSA optimizer or general pass was added, no
candidate source was created, and no new timing was run. In particular, this
report makes no hosted RTX 5090 or other performance claim.

- Worktree: `/home/ada/Projects/emmy-p5-generic-cse`
- Branch: `compiler-design/p5-generic-cse`
- Base: `d8f65f002db6c14d25c805255f7e7e6400329696`
- `compile.py` SHA-256: `067ef9c4ce31a4bf5ccda90a8d13e9bd1b71b89aff19abe7598cdba4c77e2921`

## No-op SASS inventory

The authoritative inventory is [sass-loop-inventory.json](sass-loop-inventory.json)
(SHA-256 `bd7e108063b3d73b3417e272658b2c5a1fe7eaa7a40a18ed0b498ae1599ab324`).
The coarser PTX-level screen is retained as [sass-noop-inventory.json](sass-noop-inventory.json)
(SHA-256 `dfc4fda71fde845c60500c46edb0c362dead6a86f3beb46c4f282a212b43a5ea`).
Both were produced from the frozen `compile.py` above, assembled for `sm_89`.
Raw PTX, cubin and SASS artifacts and their case manifests remain under
`.emmy/p5-inventory/artifacts/`; they are local research artifacts, not part of
this report or a merge candidate.

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

### What survives, and why it was not actionable

- The pointwise probes contain repeated PTX constant/address expressions, but
  final SASS folds them. `p02` has one constant load and four `FFMA`; `p03`
  has four `FMNMX`/`FMUL` pairs. The coarse PTX repeat counts therefore do not
  identify surviving machine work, and no dynamic mechanism predicts that a
  generic pass would leave a useful instruction reduction in these kernels.
- In `f01`, `f03` and `r04`, the 32-iteration map bodies issue 32 loads plus
  32-33 address instructions per trip. Those addresses depend on the loop
  induction value; they are not repeated invariant expressions. The reusable
  row/base calculation is already represented outside the body. Hoisting an
  address across the induction update would violate `A95`'s exact-index
  requirement and is not a legal LICM transformation.
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
| Isolated named sentinels | `.venv/bin/python .emmy/p5-inventory/run_cases_isolated.py --compiler compile.py --out .emmy/p5-inventory/isolated-sentinels-20261027 --suite core65 --case <id> ... --repetitions 2` | 13/13 pass: A30-A32, A40, A50-A52, A57, A68, A79, A83-A85 |
| Targeted boundary cases | `.venv/bin/python .emmy/p5-inventory/run_cases_isolated.py --compiler compile.py --out .emmy/p5-inventory/isolated-targeted-20261027 --suite campaign100 --case <id> ... --repetitions 3` | 6/6 pass: A26-A29, A55, A95 |
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
- Sanitizer results: `99f8f2cb38bf6acb87f5f64beda6a837ca07c73e2f0550811c3a5a2657f178c0`

The sentinel files are isolated, one case per job and repetition, under
`.emmy/p5-inventory/isolated-sentinels-20261027/` and
`.emmy/p5-inventory/isolated-targeted-20261027/`. They are correctness evidence
only and were not timed.

## Legality boundary

Any future P5 work must preserve floating-point operand order, exact index and
transport semantics, output membership and load-to-store visibility. It must
not reassociate to manufacture a common subexpression, and it must not hoist an
address across a loop-axis rebinding. The named classes in the P5 plan
are covered as follows: cache keys by A26-A29; scopes and lifetimes by A30-A32
and A40; load-name and snapshot behavior by A50-A52 and A55; attention
normalization ownership by A57, A79 and A83-A85; observable stores by A68; and
rebinding safety by A95.

## Limits and disposition

- All measurements and SASS evidence here are for local `sm_89`/RTX 4060-class
  execution. They are a proxy, not the coordinator's hosted RTX 5090 result.
- The inventory is static mechanism evidence; it does not prove a dynamic
  speed effect and cannot support a performance promotion.
- Hidden-set correctness and non-local GPU behavior remain unverified.
- Recommend keeping P5 deferred and merging only this evidence record (not a
  compiler pass). Reopen only if a new typed-IR mechanism creates a falsifiable
  dynamic prediction, with a fresh source-bound gate and a noise-calibrated
  paired timing run.

## Next action

Coordinator review: retain the existing scoped optimization pipeline, keep P5
deferred, and do not schedule P5 timing. If P3 or P4 later changes the relevant
loop or memory structure, rerun the no-op inventory against that frozen source
before reconsidering a general pass.
