# Campaign 20261024 speed handoff

Five independently confirmed incremental improvements are installed on `n00bcak`. The one authorized official helper batch completed successfully: nine HTTP attempts, six output comparisons with zero mismatches, on the RTX5090. The batch is spent; do not retry or create a second batch. These files support the authorized local commit. Publication remains owner-controlled.

The supporting checks have now been [consolidated into a 100-case
gate](../experiment-gate.md): the same 66 GPU cases plus 34 additional CPU leaf
checks. All 111 CPU assertion sites are preserved. The new qualification reuses
the unchanged source-bound GPU record and executes the shared PTX assertions
and CPU checks; it does not relabel the original 118-check run. See
[consolidation evidence](experiment-gate-consolidation.json).

The source object is `b4d3b3238c85e619f59dc7954c8518d90f8bfcd0152c5d9a8005ba9580101143`; `compile.py` SHA256 is `712fb81b24d332b80348cd24a2ef00e3ca328fea9684095fefab36033e7243cf`. Full identities, promotion history, verification scope and the official result are in [the evidence record](campaign-20261024-speed.json).

## Confirmed local results

Each row compares against the immediately preceding retained source. Do not multiply these ratios into a claimed cumulative benchmark.

| Increment | a01 attention time reduction | Existing d51 witness reduction |
|---|---:|---:|
| Shared probability reuse | 15.42% | 15.42% |
| Duplicate maximum folds | 8.92% | 8.92% |
| Scalar address packets | 2.06% | 2.06% |
| Producer maximum fusion | 5.28% | 5.53% |
| Producer probability fusion | 6.13% | 6.15% |

These are warm-cache CUDA graph groups of 32 launches, reported per launch, rather than isolated launch latency. First and independent phases cleared source-specific A/A controls. The final increment changes only these two native images; the other 64 cases were unchanged and were not retimed. Earlier a02 timing remained inconclusive. No hidden-set or overall-suite speed claim follows.

Fresh source-bound correctness passed all 15 published cases and the fixed 66-case GPU suite for both final comparison arms, at unchanged tolerances. Existing 32 named unit methods also passed. **The user clarified that all checks count toward the 100-check limit: 66 GPU cases plus 52 CPU leaf checks total 118, exceeding it.** Earlier accounting of 98 named cases/methods was insufficient. The deviation is retained in the evidence; historical results were not rewritten. No further local tests or compiler variants ran after clarification. Guarded repetitions and sanitizer executions are recorded separately.

Both sm89 and sm120 assembled. Full-suite execution was on the RTX4090. The separate official RTX5090 result below covers only one published representative.

Only one new case was generated: a demonstrated mutable-register alias regression needed for the prescale optimization. Existing 65 cases had no corresponding Assign/Accum observer. The baseline already passed it; failed candidate and repair evidence were retained. No challenge fixture or tolerance changed.

Rejected r02 first screens were slower by 23.19% (vector SUM), 19.45% (vector ILP), and 9.88% (rolling reduction). None was installed. m02 reset timing qualification failed; it was not replaced with timing that omitted resets.

## Official representative result

The unchanged existing helper made one batch of nine requests: health, two compilations, and three executions per arm. Health identified a runnable RTX5090/sm120 endpoint. Every output comparison passed unchanged `rtol=0.001`, `atol=0.0001`, with zero mismatches and maximum absolute error `6.556510925292969e-7`.

| a01 attention | Samples (microseconds) | Median |
|---|---|---:|
| Parent source | 8.864, 8.864, 8.896 | 8.864 |
| Retained source | 6.816, 6.848, 6.864 | 6.848 |

The ratio is `0.7725631768953067`, an indicative 22.74% reduction versus the immediate parent. Three samples per arm are not a significance-qualified benchmark. This is one representative hosted comparison, **not hosted all15 validation or a hidden-set score**. The raw result SHA256 is `de6cee4f2be5b646a30394c7f51c97f7de6feef0f8e601b66128150fc3933a40`.

A genuine new single-block companion measurement supplied the helper's one-burst case/PTX binding; no main measurements were filtered or relabeled. Its eligibility artifact is `3b94824fdb43a27fc5b64e85152ddd2f276dd091a7aa3c5263d4be7fcde41e97`. The first reservation was separately shortlisted before the fifth promotion.

## Durable state and resumption

Resolve the ignored private runtime configuration and read its research root's `START.md`, root `OWNER-SYNC.md`, and current campaign `HANDOFF.md`. Machine paths, connection details and launch commands remain private. Campaign ID: `campaign-20261024-speed-v1`.

The research store retains source, exact cases and realized inputs, references, PTX/cubins/SASS, worker identities, both-source all15 gates, admission records, raw timings, HTTP request/response journals, rejected candidates, assessments and replay commands. Main assessment: `863a746688ff1bef6a6d4d25a8d01e870d088bedcb53c51ba0471d534c232abe`. Fifth promotion decision: `0f2c2594c7877121e427f7cffe8fd90a7e9ce43f107fe52a98d30da94a7791d5`. Do not reissue or repromote.

SSH trust was restored after a host-key interruption. The original GPU container could not restart because its configured NVIDIA CDI device was unresolved. It and host GPU settings were left unchanged. The official client ran in a temporary CPU-only companion from the same verified image and mounts. This required no new local GPU execution and does not change the scope of historical 4090 evidence.

The user restored 6h30 remaining at 24 October 2026, 17:52:58 SGT. The resulting maximum deadline was 25 October 2026, 00:22:58 SGT. The original run ID, 76 evaluated entries, 24 distinct evaluated source objects and conservative ceiling of 89/200 were preserved; manual debug sources were already counted. Budget restoration artifact: `c50fee6efe581824c6eeb1017214727b9eb043274fb0fd2225ec1efd96249658`. The restored time did not authorize a second official batch or Git publication.

Any future compiler research needs a fresh explicit bounded authorization. Do not silently restart this campaign, change its tolerances, reinterpret the recorded test-cap overrun, or reuse the spent submission allowance.
