# Performance measurement reflection

Decision: retain as baseline evidence only; no optimization was promoted.

The measurements establish absolute smoke latency and register counts for the ten
accepted open kernels. The reduction at 65,536 elements is substantially slower
than the 1,024-element reduction, consistent with extra memory traffic and loop
iterations; this is descriptive, not a controlled bottleneck proof. Pointwise
latency is dominated by launch overhead at these sizes. All compiler times are
about 0.2 s, so compile time is not currently a concern.

The emitted PTX is scalar: no `ld.global.v4`, `st.global.v4`, no `cp.async`, and
no tensor-core path. The compiler also recomputes each address expression per
element. Those are candidate optimization debts, but must not be changed until
the unsupported categories lower correctly and regression coverage exists.

All timings come from the documented local runner on an RTX 4060, not the hosted
RTX 5090. No speedup or parity claim is made.
