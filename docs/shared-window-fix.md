# Shared-window correctness repair — 2026-10-23

The narrow fix from already available `origin/main` commit `2647806` is merged
into the current compiler, preserving all intervening compiler improvements.
Official RTX 5090 / compute capability 12.0 (`sm_120`) grading passed **15/15**;
the previous retained compiler passed 14/15 with m01 failing. This is a general
address-lowering repair, without fixture-specific conditions or changed launch,
ABI, inputs, references, precision or tolerances.

## Cause and discriminating evidence

Dynamic `cp.async` lowering used the destination's offset within its shared slab
but omitted that slab's position in the shared-state window. The corrected
32-bit destination is `low32(slab_symbol) + low32(destination - slab_symbol)`.
The symbol is a shared-state address, not a generic address. The formula also
preserves static slabs' positions. No architecture-specific numeric base is used.

With CUDA 13.0.88, the observed dynamic symbol was zero for the original SM8.9
kernel and `0x400` for SM12.0. Those are assembler allocation observations.
Adding an active 1024-byte static shared prefix exposed the old bug on SM8.9:
262,054/262,144 m01 outputs failed, maximum absolute error 145.3918877. The
corrected source passed both original and prefixed kernels in three isolated
repetitions, maximum error 8.392333984375e-05. Guards and immutable inputs passed.
The unprefixed m01 SM8.9 cubins were byte-identical before and after the repair.
These probes are retained under `m01-branch-diagnosis-v1/`; the prefix probe is
not a claim of target-device execution. Official SM12.0 execution now confirms
the repaired original case independently.

## Verification

- RTX 4090 / SM8.9: **122/122** fixed, independently generated regression cases,
  comprising 15 published and 107 generated, with three isolated repetitions,
  buffer guards and immutable-input checks. All **26** required Compute Sanitizer
  checks passed, including isolated-first m02 and prescribed per-launch resets.
- CPU: all **32** compiler tests passed, including both-target published assembly.
  One existing test now evaluates static/dynamic destination address arithmetic at
  zero and nonzero symbolic bases; it fails the old compiler at both nonzero
  dynamic bases and passes the repair. Test count and GPU corpus are unchanged.
  The first CPU invocation had a missing `tools` import path; the corrected
  invocation passed all 32. Both invocation logs are retained remotely.
- Official RTX 5090 / SM12.0: **15/15** published cases, complete returned outputs
  at unchanged metadata tolerances. One serial batch used **16 charged requests**,
  health plus 15 compile-and-run requests, with the shared spacing/budget helper.
  Health explicitly identified compute capability 12.0 and default `sm_120`.
  Emitted PTX remains `.version 8.0`, `.target sm_89`; the runner assembles it
  for `sm_120`. Local assembler version is 13.0.88. The official health response
  advertises the assembler executable path but not its version; that version
  remains unverified. Exact architecture records are in remote `architectures.json`.
  Metadata argument order, launch, shared bytes and per-launch zeroing were
  preserved. PTX was submitted for runner assembly; local cubins were not sent.
- No observed correctness regressions in the 122 local or 15 official cases.
  Generated cases remain unverified on SM12.0. No fresh paired latency comparison
  was performed, so performance regressions and speedups are unmeasured. Endpoint
  latencies are indicative only; this repair claims correctness, not a speedup.

| Official published case | SM12.0 result | Maximum absolute error |
|---|---|---:|
| `a01_attention` | pass | 6.5565109e-07 |
| `a02_attention_128` | pass | 4.4703484e-07 |
| `f01_rmsnorm` | pass | 1.9073486e-06 |
| `f02_row_softmax` | pass | 7.4505806e-09 |
| `f03_l2norm` | pass | 1.4901161e-08 |
| `m01_matmul_512` | pass | 8.392334e-05 |
| `m02_matmul_rect` | pass | 1.1444092e-05 |
| `p00_scalar_add` | pass | 0 |
| `p01_vector_add` | pass | 0 |
| `p02_scale_shift` | pass | 0 |
| `p03_relu_mul` | pass | 0 |
| `r01_array_sum` | pass | 3.8146973e-06 |
| `r02_big_sum` | pass | 1.5258789e-05 |
| `r03_mean` | pass | 9.3132257e-10 |
| `r04_row_sum` | pass | 5.7220459e-06 |

## Exact identities and resume state

| Record | Identity |
|---|---|
| Compiler SHA-256 | `51bd61bded75a2e132c94f963974f29c93783135283b8c1d028c30e848da4f00` |
| Frozen source | `13b17fd9634637c9d863403a6920ef89e84d4ac5845fd737f6e766c7d255f0db` |
| Evaluator suite | `9fa9f76ce9a0606484aa4eaa6bd6d602712cc8a7458ffddd2876f62513d02bc6` |
| Local all-15 gate | `d8e6d4da8dc9aa595de848ad1a516a50070a50446ca338df268f08c2c143afe9` |
| Confirmed local evaluation | `702e4a63c5bad92df71204e7fbebf5181ab1a3a92be994f6b7303f05fced409a` |
| Official correctness artifact | `ffe0d405f0bc0e8dd7fddf8027f176acdb29b1b51d7859bb768904141b587d6f` |
| Promotion decision | `888e55c2dbe5ff025109d9ab46b274aaa8ba3f696e338fafa940fc4da07a2a12` |

Resolve the research root from private runtime configuration. Authoritative remote
handoff: `/campaigns/campaign-20261023-12h-v2/m01-shared-window-repair-v1/HANDOFF.md`.
That directory contains `FINAL.json`, exact preparation and validation receipts,
CPU test logs and adapter source. Full official requests/responses and grading
source remain under `/calibration/official-20261023-shared-window-v1/`.
The raw request hashes are bound in the official correctness artifact above.

The canonical `/regression-suites/current.json` and `/retained/current.json`
are synchronized to the repaired compiler and the unchanged compact-v4 policy.
The repair run is closed, discovery is disabled, and no owned GPU/preparation
job remains. The singleton grading batch is consumed: do not retry it. Historical
sources and failed diagnosis evidence remain intact. No GitHub publication.

The four kernel skills now share [compute-capability and careful-submission
instructions](compute-capability.md); all four pass the skill validator. This
change adds no permission to start discovery, exceed budgets or submit work.
