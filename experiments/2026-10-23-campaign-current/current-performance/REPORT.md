Current retained compiler: f47c / source549b. This table projects existing exact-source evidence; no new GPU run, profiler, endpoint request, source change or discovery campaign was started.

The RTX4090 passed all15 published problems and its isolated/sanitizer gate remains valid. Its 47-case independently selected timing has32 pairs per case plus16 baseline A/A pairs. Published kernel and ordered-pipeline latencies coincide because every published problem has one kernel. Warm reused allocation GPU-event intervals exclude CPU compilation, assembly, transfers, host submission, reset and result-checking costs. Full raw evidence and source/suite identities are linked by manifest.json; p10/p90 and per-case A/A noise are retained in per-case.csv/json. Fourteen latest published comparisons were inconclusive; a01 alone was a resolved improvement against that measurement's parent. No minimal-control measurement is available, and no guessed overhead is subtracted.

The official RTX5090 passed14/15. Its entire performance evaluation is unqualified under the all15 rule. Raw medians from5warmup/20timed iterations remain diagnostic only, including21.696us for the incorrectm01 output. Its API does not provide the individual timing samples, dispersion, event/reset implementation, clocks, counters or an identified matching sustained ceiling. No5090MFU or useful-throughput claim is calculated. Different device protocols do not establish a cross-device speedup.

4090MFU uses useful FP32 work / time / theoretical82.6TFLOP/s non-Tensor ceiling from NVIDIA's Ada whitepaper AppendixA; it is not measured sustained utilization. Counts reuse prior typed-IR semantic accounting only after checking every IR/metadata hash and current case identity. Useful GEMM counts use2MNK; attention uses conventional QK+stable-softmax+PV. Exp/div/sqrt/comparisons are separate in per-case.json. Native instruction counts and diagnoses from the older compiler are not reused. Logical traffic assumes each distinct argument is accessed once; logical GB/s is not measured DRAM bandwidth. Actual DRAM/L2/shared traffic, achieved occupancy, observed clocks and reference latencies remain null. Cached physical register/spill records are source-bound assembly evidence; all15published kernels have zero recorded spills on both targets. No profiler was started.

Priority: restorem01 correctness onsm120; the saved exact-input independent diagnostic rules out an observed input/ABI/launch/reference mismatch but does not identify a compiler, assembler or runner cause. Recoverable time is unknown. The next diagnostic should distinguish those causes using immutable savedPTX/inputs before evaluating new speed work. Among correct4090cases, m01=25.60us, a02=23.552us, a01=18.432us andr02=7.76us are the largest current latency observations; this ranking alone does not establish recoverable compiler time. Existing schedule limits (oneCTA for r02/r03, fourCTAs for a01, andm01's72KiB shared allocation) are constraints, not proven bottlenecks.

No owned live jobs remain. Reproduce this offline table with python3 experiments/2026-10-23-campaign-current/current-performance/table.py. No compiler repair or new experiment was performed.

| Case | 4090 µs | 4090 MFU % | 5090 raw µs† | 5090 check |
|---|---:|---:|---:|---|
| p00_scalar_add | 3.072 | 0.00005 | 3.152 | PASS |
| p01_vector_add | 3.120 | 0.00040 | 3.104 | PASS |
| p02_scale_shift | 4.096 | 0.00242 | 3.168 | PASS |
| p03_relu_mul | 4.672 | 0.00053 | 3.104 | PASS |
| r01_array_sum | 4.096 | 0.00030 | 3.104 | PASS |
| r02_big_sum | 7.760 | 0.01022 | 8.832 | PASS |
| r03_mean | 6.144 | 0.00646 | 6.784 | PASS |
| r04_row_sum | 3.264 | 0.01213 | 3.088 | PASS |
| m01_matmul_512 | 25.600 | 12.69463 | 21.696 | FAIL |
| m02_matmul_rect | 5.120 | 0.99177 | 4.992 | PASS |
| f01_rmsnorm | 5.120 | 0.06200 | 4.768 | PASS |
| f02_row_softmax | 3.072 | 0.01935 | 3.104 | PASS |
| f03_l2norm | 4.096 | 0.02904 | 3.184 | PASS |
| a01_attention | 18.432 | 0.06994 | 19.072 | PASS |
| a02_attention_128 | 23.552 | 0.11116 | 25.200 | PASS |
