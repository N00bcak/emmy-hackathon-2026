# Lessons learned and failed experiments — Kernel IR → PTX optimizer

Consolidated record of every measured, rejected optimization and the durable
lessons the campaign should not have to rediscover. This is a research-store
note in the gitignored `.emmy/evidence/` tree; it cites evidence receipts and
does not describe new work. Nothing below is a merge candidate.

- Frozen `compile.py` SHA-256 (baseline, main):
  `858f5f275ac455423ab197f5932a637ca263137de52b5038ceb5d2f836123685`
- Owner `main`: `09144135c03e10cccbce3569ba9e31e8b82217eb`
- Contract source: AGENTS.md, `emmy-hackathon-2026/README.md`, `TASK.md`,
  `MVP.md`, `docs/API.md`.
- No 93-case archive is available; no 93-case run is claimed. All gates here use
  the 15 published problems plus typed variants and unit tests.

## Workstream terminal states

| WS | Kernel | State | 4060 µs | 5090 µs | Target 4060/5090 µs | Terminal record |
|---|---|---|---:|---:|---|---|
| W1 | f02_row_softmax | AT-TARGET | 5.120 | 3.072/3.104 | 6.672 / 4.760 | `.emmy/evidence/w1-f02/final/TERMINAL.md` |
| W2 | f01_rmsnorm / f03_l2norm | AT-TARGET | 6.384 / 5.120 | 4.800 / 3.104 | 9.000/5.092 / 7.440/4.869 | `.emmy/worktrees/w2-norms/docs/w2-norm-baseline.md` |
| W3 | r01/r02/r03/r04 | AT-TARGET | 4.096/8.672/6.784/5.072 | 3.008/6.816/4.800/3.008 | all met | `.emmy/evidence/w3-reductions/terminal.json` |
| W4 | m01_matmul_512 | REJECTED (no merge) | 125.664 baseline | 17.056 | 58.470 / 14.152 | `.emmy/worktrees/w4-matmul/.emmy/evidence/w4-matmul/diag-occupancy-20261024/TERMINAL.md` |
| W5 | a01_attention | REJECTED (no merge) | 13.312 (frozen) | 8.864 | 10.480 / 7.843 | `.emmy/evidence/w5-attention/terminal/record-v6.md` |
| W6 | a02_attention_128 | AT-TARGET | 8.832 | 6.816 | 10.480 / 7.843 | `.emmy/evidence/w6-attention128/final/terminal-record.json` |

No ACCEPTED branch exists; nothing was merged. Main is unchanged. W4 and W5
branches are retained (rejected, not deleted).

## The two REJECTED workstreams in detail

### W4 — m01_matmul_512

Target is 58.470 us on both devices; the frozen baseline is 125.664 us (4060) /
17.056 us (5090), i.e. ~2.15x and ~1.21x above target. m01 is the only compute-
bounded kernel: the SoL proxy `floor + max(F/P, L/Bw)` with the 2.5x margin is
23.4 us on the 4060, but its contract-fixed launch (grid 128, block 512,
73,728 B shared) forces **512 MiB of shared-memory reads**, which yields a
**shared-bandwidth floor of 104 us on the 4060** and ~14 us on the 5090. That
floor is 121% of the frozen baseline and **1.78x above the 58.470 us target**,
so m01 cannot close without a launch/ABI change (smaller shared tile, padding,
or different launch geometry). This is the documented, measured, geometry-bound
residual. It is not claimable as a win and not attributable to a compiler
choice.

The shared bandwidth figure assumes a spec-sheet peak. The 5090 1792 GB/s figure
was never measured on the endpoint — a `ptxas .version 8.0 / target sm_120`
probe failed and the ctypes `cuDeviceGetAttribute` calls returned zeros (type
mismatch). The 4060 peak (24 SM × 128 B/clk × 1.89 GHz ≈ 5.81 TB/s) is what
matters for the primary closure; the 5090 line is provisional.

**Experiments measured and rejected (all 4060):**

| # | mechanism | 4060 m01 | vs baseline | record |
|---|---|---:|---:|---|
| 0 | frozen baseline | 125.664 | 1.000x | main |
| 1 | warp-tile lane permutation | 155.872 | 1.240x slower | w4 |
| 2 | 4×4 register tile, 128 threads | 139.008 | 1.106x slower | w4 |
| 3 | 4×2 register tile, 512 threads | 205.280 | 1.634x slower | w4 |
| 4 | 4×4 tile + 4-way split-K | 136.448 | 1.086x slower | w4 |
| 5 | swizzled 8×16 warp footprint | (smoke) 128.448 | — | `.emmy/worktrees/w4-matmul/.emmy/evidence/w4-matmul/fp32-swizzle-8x16-20261024/TERMINAL.md` |
| 5b | 8×16 without swizzle (diagnostic) | (smoke) 156.544 | 1.246x slower | w4 |
| TF32 prepack (three-term) | (smoke) 140.480 | 1.118x slower | `.emmy/evidence/w4-matmul/prepacked-gate-20261024/TERMINAL.md` |
| TF32 direct | (smoke) — | — | `.emmy/evidence/w4-matmul/probe-direct-tf32-20261024/TERMINAL.md` |
| H2 (incomplete, no merge) | — | — | `.emmy/evidence/w4-matmul/h2-incomplete-20261024/TERMINAL.md` |

**Occupancy diagnostic** (not a candidate; typed-IR reschedule only, `compile.py`
byte-frozen): doubling residency via 49,152 B (driver-confirmed 2 CTA/SM) ran
117.088 vs 130.160 us = **1.097x**, below the pre-registered 1.2x bar. So
occupancy is a real but minor contributor (≈10%), not the dominant limiter. The
frozen SASS carries 115 IMAD + 18 LOP3 + 18 SHF against 96 FFMA per unrolled
body — the untested lever is issue/throughput pressure from address arithmetic,
not shared traffic or occupancy. The earlier "gap outside compiler control"
claim is withdrawn; the residual is geometry-bound, not unexplained-by-compiler.

**Swizzle proof:** the XOR A-swizzle moved the same 8×16 footprint from 156.544
to 128.448 us (1.22x), confirming the bank-conflict model, but it still lost
because the swizzle makes A non-affine in K (+37 IMAD, +43 LOP3/XOR, +5 SHF
per body vs 96 unchanged FFMA).

### W5 — a01_attention

Target is 10.480 us (both); the frozen baseline is 13.312 us (4060) / 8.864 us
(5090), ~1.25x / 1.13x above target. m02 is 16-CTA / 24-SM (all 24 SMs active).

**Experiments measured and rejected:**

| mechanism | result | record |
|---|---|---|
| hoisted-QK rowmax (typed) | 4060 resolved win (13.312→12.288), 5090 unresolved at 1.000 | `.emmy/evidence/w5-attention/terminal/record-v6.md` |
| PV-loop static removal | destroys correctness; no machine-code win | w5 |
| alpha/1.0 elimination | same 268 FMUL, same SASS on both targets; 1% deadband | `.emmy/evidence/w5-attention/terminal/record-v3.md` |
| H2 / H2c variants | slower or unresolved on 5090 | w5 |
| identity (1.0 scale fold) | unchanged PTX | w5 |

**a01 key bound:** deleting the entire PV/online-softmax loop (which destroys
correctness) still lands at 10.527 us vs the 10.480 us target. The 3417 cycles
the target requires cannot come from any order-preserving instruction removal.
The 4060 rowmax win (12.288 us) is real but still above target; it cannot be
merged because it is unresolved on the 5090.

## Durable lessons

1. **Launch geometry, not lowering, bounds m01.** A smaller shared tile,
   padding, or different launch geometry (smaller block) is the only path under
   the 58.470 us target, and all three change the metadata ABI. Compiler
   codegen cannot close m01 without an ABI change.
2. **Occupancy is not the m01 bottleneck.** Doubling residency bought ~10%, not
   the ~54% a latency-bound model needs. The kernel is issue/throughput-bound.
3. **Shared bandwidth is the real m01 floor.** 512 MiB of shared reads at the
   contract geometry gives a 104 us floor on the 4060. This exceeds target by
   1.78x and cannot be attributed to a compiler choice.
4. **The swizzle confirms bank conflicts but costs more than it saves.** The
   XOR swizzle is a valid bank-conflict fix; making A non-affine in K is too
   expensive (37 IMAD + 43 LOP3/XOR + 5 SHF per body).
5. **m02 has no occupancy issue.** 16-CTA / 24-SM means all 24 SMs run; the
   concern there is atomic ordering, not SM count.
6. **a01 is compute/schedule-bound, not instruction-count-bound.** Deleting
   loop body never clears the target; the residual is arithmetic, not dead
   instructions.
7. **F16 tensor-core path was abandoned.** It adds ~20 lines, ~400 tokens,
   ~60s, and TF32-only correctness. FP32 SIMT is the accepted path.
8. **93-case archive is absent.** No 93-case run was completed or claimed; gates
   use the 15 published cases plus typed variants and unit tests.
9. **No merge happened.** Main is unchanged at `09144135`. Rejected branches are
   retained.

## Reproduction notes

- Baseline gate and assembly are in `.emmy/evidence/baseline/` (4060-gate,
  5090-gate, assembly).
- Both-device paired timings use warmup 10, 50 iterations, 7 alternating blocks;
  all timed jobs held the shared `flock`.
- The 93-case archive is unavailable; do not claim a 93-case run.
