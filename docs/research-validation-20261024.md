# Compiler validation, 24 October 2026

The retained compiler is unchanged: source `b4d3b3238c85e619f59dc7954c8518d90f8bfcd0152c5d9a8005ba9580101143`, compiler SHA `712fb81b24d332b80348cd24a2ef00e3ca328fea9684095fefab36033e7243cf`. The five earlier independently confirmed promotions remain retained. This bounded continuation evaluated three new compiler sources; none produced a resolved speed improvement, so none was installed or promoted.

The research runtime works again on the configured RTX 4090. Deployment details and the resumable runtime handoff remain in ignored private configuration. All new execution was on the 4090. Both sm89 and sm120 assembled; sm120 assembly establishes no new 5090 correctness claim.

## Correctness and test budget

The retained baseline and all three candidates each passed the consolidated **100-case profile: 66 GPU cases plus 34 additional CPU leaves**. It includes all 15 published contracts, the 50 audited core witnesses and the existing essential alias regression. All 111 CPU assertion sites remain represented. Numerical tolerances, realized inputs, reference arrays, published metadata and challenge fixtures are unchanged. No new case was generated; existing p02 already exposes the performance mechanism under investigation.

Source-bound all-15 gates, isolated correctness, required sanitizers and clean host observations precede timing. A second, distinct baseline confirmation precedes the independent measurement-qualification phase. Exact source, suite, assembly and GPU receipt identities are in the [evidence index](evidence/campaign-20261024-validate.json).

The historical 118-check overrun remains documented separately; these receipts neither relabel that run nor waive it. The consolidated workflow is documented in [experiment-gate.md](experiment-gate.md).

## Measurements

Negative changes mean a lower observed median time. These readings are **inconclusive**, not confirmed improvements or regressions.

| Candidate | p01 vector add | p02 scale/shift | p03 ReLU/multiply | Native scope |
| --- | ---: | ---: | ---: | --- |
| Exact unsigned Tile decoding (`2fadc9ee256f`) | Unmeasured | −2.86% | Unmeasured | All 66 cases change; p02 diagnostic only |
| Input packetization (`9e4dd2f13e34`) | 0.00% | +5.71% | 0.00% | Only these three cases change |
| Combined (`ddbe6f6acb15`) | 0.00% | −5.71% | 0.00% | Only these three cases change |

The p02 A/A uncertainty was **8.57%**. Its predeclared warm-cache R32 protocol passed both initial and independent fresh qualification, with actual buffer separation, graph counts, outputs, inputs, guards, cubins, GPU identity and host admission independently checked. This measures elapsed time per invocation in repeated warm graph execution; it is not standalone kernel latency or streaming throughput.

p01 and p03 retain the original R1 protocol, with approximately **100% A/A uncertainty**. Their median ratios of 1.0 cannot resolve small effects. All seven comparisons finished five fixed blocks each: two A/A admission blocks, two A/B blocks and a closing A/A block, totaling 35 blocks and 1,120 pairs. No result was redrawn or rescued under a different metric. The observed +5.71% load-only slowdown remains explicitly reported.

The general Tile-decoding source has 65 other changed cases without performance measurements, including m02 with an unresolved reset-inclusive measurement qualification. Its p02 diagnostic cannot justify retaining the broader source. The two narrower variants have identical native outputs to the baseline for the other 63 cases on both targets.

## Findings and retained state

Unsigned strength reduction removes the native reciprocal and quotient-correction sequence from p02. Aligned input packetization activates a vector input load with scalar fallback and reduces registers. The combined lowering activates both changes. These static effects and correctness passes did not establish a latency improvement above measured noise. No new compiler source was copied into the owner checkout.

The timing-only E6 evaluator adds one independently reviewed safety certificate for the existing p02 case and changes its expected digest. The remaining worker bytes and AST, other tools, prior certificates, cases, oracles and tolerances match the prior evaluator.

The original official batch remains spent: nine HTTP attempts, with one published a01 case checked on an actual 5090. This continuation made zero official requests. The exact previously checked compiler and supporting local commits remain retained; there was no new commit, Git network operation or GitHub publication.

The run closed at 15:56 UTC, before its 16:03:27 UTC stage deadline and the unchanged 16:22:58 UTC session deadline. Fourteen evaluation entries plus six conservatively reserved fingerprints and the prior 108 give a session upper bound of **128/300**. No GPU jobs remain active. Research is disabled after this completed run; future work must preserve these outcomes and start with a fresh bounded authorization.

Durable final artifact: `b2ed659f127f7fb949be0caa42e7432025849b0332779e3cbb2ece6257335562`.

Independent screen audit: `3b9cb29abba09044e45b7e9e798226fc3251e043442e04fbc7e2f3209ab0048e`. Metric audit: `259e7f7f518d3e1121e2f8242cfde944abc65140bc90c6533efaf448c51d74e6`. Evaluator projection audit: `34d84076eb47c3696c3a6ed3c33f905ba87c23727d6af9bd5d004ed2b7f95807`.
