# Timing matrix: RTX 4060 and hosted RTX 5090

This directory records the requested 2x2 matrix:

| Device | `main` | `tensor-core-lowering` |
|---|---|---|
| Local RTX 4060 Laptop, sm_89 | baseline | candidate |
| Hosted RTX 5090, sm_120 | baseline | candidate |

The comparison is one baseline/candidate pair, so each paired result supplies
both branch columns. Local timing uses the recorded
`cuda-graph-paired-v2` protocol with 16 pairs and a matching A/A control per
case. Hosted timing uses the organizer runner's `median_us` for a 10-warmup,
50-iteration launch, repeated seven times per branch and interleaved by branch
order. Hosted numbers are indicative: clocks are not locked and the endpoint
is not an evaluator-issued performance qualification.

## Frozen identities

- Baseline source: branch `main`, commit `2647806c65f13d45268a361d29005b2d34203aec`,
  compiler SHA-256 `6436b729336c7e5e1cbeab6e104745adb8d1b0f68a36906e496c1d08b51abed9`.
- Candidate source: branch `tensor-core-lowering`, commit
  `f2ce24840a00928fdfe8772b0f955b860eaf577e`, compiler SHA-256
  `2d35bf97358e9f8fe9243a4a7b436145662cb543f3a5f35370aecb1146f5a3d0`.
- Candidate runtime helpers: `tensor_core.py` `ca5af9c9...`, `tf32_emitter.py`
  `6a9e1af9...`.
- Challenge submodule: `ec8cde19...`; problem `meta.json` and `expected.npz`
  hashes are in the hosted summary and the prepared manifests.
- Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, UUID
  `GPU-01c8f1da-9f9f-fa3e-bbe2-0771199ab74a`, sm_89.
- Hosted GPU: NVIDIA GeForce RTX 5090, sm_120 (runner health in the summary).

## Correctness gate

Both branches passed all 15 published problems at unchanged metadata tolerances
on the local sm_89 runner and on the hosted sm_120 runner:

- [4060 `main` gate](gate-main/results.json)
- [4060 tensor-core gate](gate-tensor-core/results.json)
- [5090 both-branch gate and timings](hosted/run-5090b/summary.json), with
  `published_correctness.all15_passed: true`, all runs checked against the
  expected outputs, and 271 HTTP responses recorded in `requests.jsonl`.

The ABI reorder probe also passed locally for both branches. These are
published-case results; hidden-set correctness remains unverified.

## Results

Local medians use the confirmed 32-pair blocks (except `m01`, which retains its
16-pair block). Each is paired with a matching A/A control at the same sample
count. Hosted values are the median of seven interleaved per-branch samples from
the runner's reported `median_us`. All times are measured GPU kernel/launch
latency in microseconds; they exclude HTTP, compilation, allocation and transfer
overhead. Ratios below 1.0 mean the tensor-core branch is faster.

| Case | 4060 `main` us | 4060 tensor us | 4060 ratio | 5090 `main` us | 5090 tensor us | 5090 ratio |
|---|---:|---:|---:|---:|---:|---:|
| `p00_scalar_add` | 3.072 | 3.072 | 1.0000 | 2.976 | 2.976 | 1.0000 |
| `p01_vector_add` | 3.072 | 3.072 | 1.0000 | 2.976 | 3.008 | 1.0108 |
| `p02_scale_shift` | 3.072 | 3.072 | 1.0000 | 3.040 | 3.040 | 1.0000 |
| `p03_relu_mul` | 3.072 | 3.072 | 1.0000 | 2.976 | 3.024 | 1.0161 |
| `r01_array_sum` | 3.072 | 3.072 | 1.0000 | 3.008 | 3.008 | 1.0000 |
| `r02_big_sum` | 23.552 | 23.552 | 1.0000 | 21.088 | 21.088 | 1.0000 |
| `r03_mean` | 13.472 | 13.392 | 0.9941 | 12.896 | 12.896 | 1.0000 |
| `r04_row_sum` | 3.072 | 3.232 | 1.0521 | 2.976 | 2.944 | 0.9892 |
| `m01_matmul_512` | 479.232 | 134.144 | 0.2799 | 60.064 | 21.136 | 0.3519 |
| `m02_matmul_rect` | 10.240 | 10.240 | 1.0000 | 5.024 | 5.040 | 1.0032 |
| `f01_rmsnorm` | 5.120 | 5.120 | 1.0000 | 4.768 | 4.784 | 1.0034 |
| `f02_row_softmax` | 4.096 | 4.096 | 1.0000 | 3.008 | 3.008 | 1.0000 |
| `f03_l2norm` | 4.096 | 4.096 | 1.0000 | 2.976 | 2.976 | 1.0000 |
| `a01_attention` | 243.712 | 244.928 | 1.0050 | 96.880 | 96.896 | 1.0002 |
| `a02_attention_128` | 116.944 | 116.784 | 0.9986 | 55.968 | 55.968 | 1.0000 |

## Interpretation and limits

- `m01` is the only case whose PTX/cubin differs between branches; it improves
  by `3.57x` on the 4060 and `2.84x` on the 5090 relative to `main`.
- Every other case has byte-identical PTX and cubins between branches. Any
  measured deviation on those rows is timer noise or host state, not a branch
  effect. This explains the 4060 outliers such as `p00`, `r04`, and `f03` and
  the small hosted sub-1% differences.
- Initial 4060 rows had several timer/order outliers even though the emitted
  cubins were byte-identical. A fresh 32-pair confirmation for every non-`m01`
  case returned `1.0000` for all rows except `a01`, whose `1.0042` matches its
  own `1.0041` A/A control. See `timing/*-confirm.json` and the
  `timing/*-f02_row_softmax-recheck.json` files.
- Hosted results come from the organizer endpoint, not an SSH-accessible host.
  The runner reports indicative timings without clock locking. The hosted run
  made 271 successful HTTP requests, had no busy/rate-limit responses, and used
  a 0.35 s inter-request gap.
- No m02 atomic-specific sanitizer run is claimed here; the reported matrix is
  the requested timing comparison and both branches preserved the published
  m02 result.

## Reproduction

Local gates:

```sh
.venv/bin/python tools/check_local.py --compiler <main-worktree>/compile.py --out experiments/016-timing-matrix-2026/gate-main
.venv/bin/python tools/check_local.py --compiler compile.py --out experiments/016-timing-matrix-2026/gate-tensor-core
```

Local prepared timing and hosted matrix:

```sh
.venv/bin/python tools/prepare_timing.py --out experiments/016-timing-matrix-2026/timing/prepared-all15 \
  --candidate compile.py --baseline <main-worktree>/compile.py \
  --case p00_scalar_add ... --case r04_row_sum
.venv/bin/python -m tools.research_gpu <job.json>
.venv/bin/python experiments/016-timing-matrix-2026/hosted/run_hosted_matrix.py \
  --out experiments/016-timing-matrix-2026/hosted/run-5090b \
  --prepared experiments/016-timing-matrix-2026/timing/prepared-all15
```
