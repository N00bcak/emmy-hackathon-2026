# RTX 4060 baseline audit (2026-10-22)

GPU: NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 8 GiB; driver 615.78.08;
compiler SHA-256 `690a9cffc5353fcd30b93ef6368f002f5bf4f086929c4f857de7c7a8c27db567`.

## Correctness through unchanged submit.py + local runner

| problem | verdict |
|---|---|
| `p00_scalar_add` | pass |
| `p01_vector_add` | pass |
| `p02_scale_shift` | pass |
| `p03_relu_mul` | pass |
| `r01_array_sum` | pass |
| `r02_big_sum` | pass |
| `r03_mean` | pass |
| `r04_row_sum` | pass |
| `m01_matmul_512` | compile-time unsupported |
| `m02_matmul_rect` | compile-time unsupported |
| `f01_rmsnorm` | pass |
| `f02_row_softmax` | compile-time unsupported |
| `f03_l2norm` | pass |
| `a01_attention` | compile-time unsupported |
| `a02_attention_128` | compile-time unsupported |
| `reordered_scale_shift_abi` | pass (ABI probe) |

## Indicative timings (5 independent requests, warmup=3, iters=20)

| problem | compile s | PTX bytes | registers | median us | samples us |
|---|---:|---:|---:|---:|---|
| `p00_scalar_add` | 0.196 | 1901 | {'k_add_pointwise': 20} | 4.016 | 3.888, 4.016, 4.096, 4.096, 3.984 |
| `p01_vector_add` | 0.193 | 1902 | {'k_add_pointwise': 20} | 4.096 | 4.096, 4.016, 4.096, 3.936, 4.096 |
| `p02_scale_shift` | 0.198 | 3626 | {'k_add_pointwise': 28} | 4.096 | 4.096, 3.984, 4.096, 4.096, 4.096 |
| `p03_relu_mul` | 0.185 | 3277 | {'k_mul_pointwise': 28} | 4.096 | 4.096, 4.096, 4.096, 4.096, 4.096 |
| `r01_array_sum` | 0.195 | 3321 | {'k_sum_1_reduce': 16} | 4.160 | 4.208, 4.160, 4.832, 4.096, 4.096 |
| `r02_big_sum` | 0.207 | 3322 | {'k_sum_1_reduce': 16} | 24.576 | 24.576, 24.576, 24.576, 23.552, 23.520 |
| `r03_mean` | 0.196 | 3578 | {'k_mean_ffcca3': 16} | 14.336 | 14.144, 14.224, 14.368, 14.336, 14.336 |
| `r04_row_sum` | 0.204 | 3449 | {'k_sum_1_reduce': 17} | 4.880 | 4.848, 4.928, 4.880, 4.784, 5.008 |
| `f01_rmsnorm` | 0.206 | 4630 | {'k_mean_cbd1de': 17} | 6.704 | 6.864, 6.848, 6.432, 6.704, 6.368 |
| `f03_l2norm` | 0.201 | 3830 | {'k_div_reduce': 17} | 5.120 | 5.072, 5.120, 5.120, 5.120, 5.088 |

## Failure evidence

- `m01_matmul_512`: `Unsupported: Auxiliary/raster tile scheduling unsupported`
- `m02_matmul_rect`: same raster/auxiliary Tile rejection
- `a01_attention`, `a02_attention_128`: same raster/auxiliary Tile rejection
- `f02_row_softmax`: `Unsupported: Accumulator base is unsupported`
- All five fail before PTX assembly; no GPU wrong-answer result exists for them.

## Coverage
- Passing open kernels: pointwise 4/4, reductions 4/4, fused 2/3, overall 10/15.
- Passing categories: pointwise, reductions; fused is partial; matmul 0/2 and attention 0/2.
- CPU contract tests: 16/16 pass with `.venv/bin/python`; bare `python` lacks numpy.
- Hidden set: unverified. These are open fixtures only.

## Limitations
- Local runner timing uses CUDA events and is indicative, not hosted-runner parity.
- No baseline from the reference CUDA path was compiled or timed; these are absolute
  smoke numbers, not speedups. The tx dependence (r02 24.6 us vs r01 4.2 us) is
  the clearest relative signal.
- Five independent requests per case are sufficient for a baseline only; they are
  not paired baseline/candidate measurements, so no optimization is promoted.

