# Attention slowdown: exponential lowering

**The guarded exponential introduced in the merge explains the measured gap to
powenn on RTX 4090.** Removing that guard from merged recovers powenn's latency;
putting the identical guard into powenn reproduces merged's latency. Both
interventions agree across two initial blocks and independent confirmation.
Other differences between these compiler versions leave no resolved attention
latency gap when exponential lowering matches.

At diagnosis completion the owner compiler remained at SHA
`7d5d4d2d0984aaa61c713c5cba2550628f08cb2331c194e781188332333cd2d0`.
The unguarded diagnostic variant was not promoted. A subsequent
[selective exponential implementation](selective-exponential.md) preserves
subnormal results with one exponential and is now incorporated.

## Controlled result

All arms use the same existing frozen IRs, inputs, references, metadata launch,
tolerances, evaluator and sm_89 execution device. Three source-only variants
change exponential lowering; no test cases or inputs were generated.

| Source / exponential lowering | a01 µs | a02 µs |
| --- | ---: | ---: |
| Powenn: unconditional FTZ | 9.216 | 6.144 |
| Merged: guarded FTZ/full-result fallback | 11.264 | 9.216 |
| Merged with only that lowering changed to FTZ | 9.216 | 6.144 |
| Powenn with the merged guard inserted | 11.264 | 9.216 |
| Merged with plain non-FTZ exponential, no outer guard | 10.240 | 7.168 |

These are candidate medians from the paired comparisons, reproduced in independent
confirmation. Baseline medians vary between comparisons; ratios are calculated
within each pair, not by dividing unrelated table cells. The direct
powenn-to-merged-FTZ comparison has median ratio 1.0 on both problems in both
initial blocks and confirmation. Inserting the guard into powenn gives ratios
1.2222 and 1.5 throughout. Removing it from merged gives approximately 0.8182 and
0.6667. This transfers the entire previously observed gap within timing resolution.

## Mechanism in machine code

The guarded PTX looks like a comparison and mutually exclusive exponentials:

```ptx
setp.ge.f32 normal, scaled, -125;
@normal  ex2.approx.ftz.f32 result, scaled;
@!normal ex2.approx.f32     result, scaled;
```

That spelling does not guarantee one exponential instruction executes. The
assembler expands the full-result operation to handle very negative arguments,
including comparisons, conditional halving, `MUFU.EX2`, squaring and selection.
At inspected sites, both fast and fallback `MUFU.EX2` instructions are
unconditional; predicates select results or correction operations afterward.
The rare-path calculation therefore adds work even for ordinary softmax inputs.

The changed instruction body also changes assembler unrolling and register
allocation. For example, sm_89 a02 uses 167 registers in powenn and 88 in merged;
its static exponential-site counts differ because the loops are unrolled
differently. Static instruction counts are not dynamic work counts. The
controlled interventions attribute the latency change to exponential lowering as
a whole; they do not allocate exact microseconds between SFU work, predication,
loop scheduling and unrolling. No hardware-counter bottleneck claim is made.

Plain non-FTZ lowering is faster than our guarded version, while retaining the
full-result operation. Its remaining measured cost against FTZ is attributable
to that lowering change, including the assembler's resulting scheduling choices.
This also shows why adding an explicit guard around an already-expanded fallback
was not a free fast path.

The stricter softmax recognizer is a compile-time check; both versions activate
two-pass softmax for these problems. Missing cooperative score computation,
tensor-core selection and lower occupancy do not explain this measured gap.

## Evidence and limits

- [Source identities, gates and file hashes](evidence/attention-cause/manifest.json).
- [45 correctness checks](evidence/attention-cause/correctness.json): each of the
  three diagnostic variants passes all 15 existing published problems, including
  required isolated atomic sanitizer checks, before timing.
- [Initial paired results](evidence/attention-cause/timing-summary.json) and
  [independent confirmation](evidence/attention-cause/confirmation-summary.json):
  two blocks of 32 pairs, then 32 new pairs per comparison on the same inputs.
- [All 1,344 raw pairs](evidence/attention-cause/paired-samples.csv) and
  [A/A controls](evidence/attention-cause/noise.json). a02 merged A/A shows order/
  timing bias in the first run (median 0.903); confirmation is 0.983. The 1.5
  slowdown and mirrored intervention persist in both runs. Small differences
  between matched-FTZ sources are unresolved.
- [Machine-code summaries](evidence/attention-cause/sass-summary.json) and full
  disassembly beneath `evidence/attention-cause/sass/` cover sm_89 and sm_120.
- [Post-job observations](evidence/attention-cause/host-observations.json) report
  no observed contention and successful exits. Periodic observations do not prove
  absence of every transient overlap.

Performance evidence is RTX 4090/sm_89 only. sm_120 assembly/disassembly is not
RTX 5090 execution or a promise of the same latency effect.

The FTZ-only diagnostic is not a general replacement: existing saved evidence
shows subnormal exponentials can matter after rescaling. Those cases are archived
stress checks under the condensed policy, and this diagnosis does not expand the
gate. A production change should preserve the required numerical behavior with a
real cold path or a proved numerical specialization, then use the existing
93-case suite. No candidate compiler was copied into the owner checkout here.
