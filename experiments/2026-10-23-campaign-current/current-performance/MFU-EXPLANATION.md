The MFU arithmetic was rechecked independently using Decimal. All 15 published percentages match; five representative work counts were independently rederived. CUDA event milliseconds are converted to microseconds by the worker, and then to seconds by the metrics calculator. No percent or time-unit conversion error was found.

The column describes **whole-GPU useful-FLOP efficiency relative to theoretical FP32 peak**. It is not measured hardware utilization. The original display omitted useful work and launch size beside that percentage, making it easy to misinterpret.

For m01: 2×512³ = 268,435,456 FLOPs. At 25.6 µs this is 10.486 TFLOP/s; divided by 82.6 TFLOP/s and multiplied by 100 gives 12.695%. For a01: QK+PV plus the stated softmax arithmetic gives 1,064,896 useful FLOPs. At 18.432 µs this is 57.774 GFLOP/s, giving 0.069945% of 82,600 GFLOP/s.

a01 launches four CTAs; r02 launches one CTA. The device has 128 SMs. These are hard limits on possible simultaneous SM coverage, not measured occupancy. Addressing, loads/stores, synchronization, exp/div, redundant work and any fixed timing cost consume time without increasing useful FLOPs. Their actual contributions remain unmeasured. The broad FMA peak is also not a demonstrated attainable ceiling for each workload's operation mix.

Low useful throughput can still identify an optimization question; these percentages do not establish which compiler or schedule mechanism causes it. No new GPU, profiler, endpoint or compiler work was performed for this check.
