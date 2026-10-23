# Selective subnormal handling — 2026-10-23

The compiler now uses **one FTZ exponential**, with a small range correction only
for arguments that could produce tiny results. It preserves the existing
subnormal-rescaling behavior without computing both fast and fallback
exponentials. Source SHA:
`12202ea108f615fffd9ab00b5dce985edb3cbe11ffad55042d3d9625b6a49354`.

This is a conservative numerical-range decision. It does not try to predict
whether losing a particular tiny value would eventually violate the whole
kernel's output tolerance. Unknown downstream amplification remains supported.

## Mechanism

Let `t = x * log2(e)` be the existing F32 exponential argument. The lowering is:

```text
if t < -125:
    result = exp2_ftz(t + 126) * 2^-126
else:
    result = exp2_ftz(t)
```

The shift moves the range capable of producing nonzero subnormals into a normal
range for `exp2_ftz`. Multiplication by the exact power of two restores the scale.
That final `mul.rn.f32` intentionally has no FTZ modifier, so it can produce a
subnormal. The -125 boundary conservatively includes a margin above underflow.
Exponents far below the representable range still round to zero. This remains
an approximate F32 exponential, not a correctly rounded general math library.

The shift and scale-back instructions are predicated. Both sm_89 and sm_120
machine code show a single `MUFU.EX2` per logical value and selective `FADD` and
`FMUL` corrections. There is no second exponential or warp vote. The two initial
branch candidates were rejected after cheap disassembly checks: the assembler
converted their control flow into predicated code. Their source and artifacts
remain in the research store, outside the owner checkout.

## Existing-case validation

The [source-bound manifest](evidence/selective-exp/manifest.json) and
[correctness receipts](evidence/selective-exp/correctness.json) establish:

- **93/93 default cases pass**, including all 15 published problems and all
  retained marked-operator witnesses, at unchanged metadata tolerances.
- **20 required sanitizer checks pass**; execution is isolated with guards and
  input-integrity checks, three repetitions per case.
- Both previously saved numerical witnesses pass. `subnormal_exp_rescale` has
  maximum absolute error `5.59e-7`, versus `0.08194` for FTZ-only in the earlier
  diagnosis. The normal/subnormal-boundary witness has error `4.77e-6` and passes
  its original combined relative/absolute tolerance.
- All 95 cases (93 default plus those two existing witnesses) assemble for
  **sm_89 and sm_120**. No new test cases, inputs, references or tolerances were
  created, and the default policy remains 93 cases.
- **51 existing CPU/compiler/selector tests pass**. One instruction-spelling
  assertion now accepts an FTZ-based exponential implementation; the existing
  case and numerical checks remain intact.

Only a01, a02 and f02 have changed published cubins. The other 12 published
kernels are byte-identical on both targets; see
[the identity comparison](evidence/selective-exp/unchanged-kernels.json).

## Performance

Representative paired candidate medians on RTX 4090/sm_89:

| Kernel | Previous guarded source µs | Selective source µs | FTZ-only powenn µs |
| --- | ---: | ---: | ---: |
| a01_attention | 11.264 | 10.240 | 9.216 |
| a02_attention_128 | 9.216 | 7.168 | 6.144 |
| f02_row_softmax | 4.096 | 4.096 | 4.096 |

The attention observations repeat in two initial 32-pair blocks and independent
32-pair confirmation. f02 was measured separately with A/A controls and an
independent repeat; no resolved change was observed. Source-bound all-15 gates
precede timing. Ratios are medians of paired ratios and need not equal the ratio
of separately summarized latency medians. a02 A/A shows ordering/timer bias
(initial median 0.90, confirmation 0.988); the candidate's paired median is
approximately 0.70 in both phases, below those controls. a01's improvement is
smaller, with paired ratio 0.909 and A/A lower decile 0.917 in both phases.

Keep that noise limitation beside the reported gains. The code does not claim
FTZ-only performance parity, a 5090 speedup, or universal bit equality with the
previous approximate exponential.

[Initial results](evidence/selective-exp/timing-summary.json),
[confirmation](evidence/selective-exp/confirmation-summary.json),
[softmax confirmation](evidence/selective-exp/softmax-confirmation-summary.json),
[all 1,024 raw pairs](evidence/selective-exp/paired-samples.csv),
[A/A noise](evidence/selective-exp/noise.json),
[machine-code summaries](evidence/selective-exp/sass-summary.json) and
[post-job observations](evidence/selective-exp/host-observations.json) are retained.
The GPU observations report no observed contention; periodic sampling is not a
reservation or proof that no transient overlap occurred.

The accepted source was copied back byte-for-byte after GPU validation and
independent timing confirmation. The owner branch remains uncommitted. RTX 5090
execution is unverified; sm_120 assembly is not an execution receipt. The remote
retained baseline pointer is unchanged; accepted source, gates, exact cases and
artifacts are under durable key `sync-powenn-20261023/cold-exp-v1`.
