# Access reuse speculative comparison, 2026-10-27

Current compiler SHA256: `8ce9868462003fd5d44db861e21cd94165438ceeb5f10eed5544a4a5f0bb2410`. P1 pointwise vector access SHA256: `5da1941927c226d2c53dd274b9749cd5b03ff3261cfe5444fb16fa12066fb281`. P3 sibling fusion SHA256: `1412ce6a5f0d29594c00ee0e7201c627fb12782160c7686b2146a6c2f92f3295`.

The configured durable research store contains campaign `speculative-access-reuse-20261027`. Its `HANDOFF.md` and `source-bound-all15-receipt.json` bind exact sources, evaluator, published cases, and raw GPU results. Current source, P1, and P3 each passed all 15 published cases under unchanged metadata tolerances on the RTX 4090 (width 1, three repetitions per case). All three passed m02 Compute Sanitizer memcheck with zero errors. The existing scaled p03 4M and a01 512x64 cases passed guarded full-output correctness for their respective candidates. Both sm89 and sm120 assembly succeeded. No cases or tests were written.

Five-pair 4090 timing, same-source control then candidate comparison:

| Case | Port | Control candidate/baseline ratios | Candidate/baseline ratios | Median candidate / baseline | Verdict |
| --- | --- | --- | --- | --- | --- |
| p03_relu_mul_4m | P1 | 0.4000, 1.0000, 1.0000, 3.0625, 0.9631 | 0.3469, 1.0204, 0.9981, 3.2667, 0.9800 | 50.176 / 50.176 us | Unresolved; control is unstable |
| a01_attention_512x64 | P3 | 1.0349, 0.9909, 1.0000, 1.0000, 0.9926 | 0.9994, 0.9636, 0.9727, 0.9725, 0.9636 | 108.544 / 112.640 us | Small speculative gain within control variation |

The timing jobs ran under exclusive host admission, exited 0, and recorded `observed_no_contention=true`. Raw jobs, results, cubins, PTX, SASS, and host attestations remain under the remote research root. These observations do not qualify either port for promotion. No latency regression was established from these two cases; latency on other cases was not measured.

The first official 5090 batch (`hosted-speculative-v1`) stopped after its `/health` request failed to resolve the endpoint hostname in the network sandbox. It made zero `/run` submissions. After the user explicitly requested an actual submission, a new bounded recovery batch (`hosted-speculative-v2`) ran outside that sandbox. All 17 planned HTTP requests returned 200, within the 25-attempt cap. The runner reported an RTX 5090 and sm120. Both arms passed full-output checks at exact metadata tolerances on the first pair of each case. Sources were unchanged at completion.

| Official case | Port | AA baseline timings | Three paired candidate / baseline timings | Median ratio | Verdict |
| --- | --- | --- | --- | --- | --- |
| p03_relu_mul_4m | P1 | 10.896, 10.880 µs | 8.912 / 10.896; 8.928 / 10.912; 8.896 / 10.864 µs | 0.8182 | Faster under the batch's 2% noise margin |
| a01_attention_512x64 | P3 | 115.344, 115.360 µs | 113.312 / 115.328; 113.280 / 115.328; 113.328 / 115.296 µs | 0.9825 | Unresolved under the batch's 2% noise margin |

The [official batch summary](access-reuse-speculative-official-20261027.json) is retained in this repository. Its raw request records and exact submitted PTX are under `/private/tmp/access-reuse-hosted-compare/hosted-speculative-v2/` and `/private/tmp/access-reuse-hosted-compare/ptx/`. An earlier official comparison of the whole access-reuse branch is separate evidence and must not be attributed to these selective ports.

P2 cross-phase row caching was not ported into the current compiler. It depends on a larger reuse-analysis and hoisting subsystem absent from this source; a partial port would not exercise the mechanism safely. At the time of this speculative comparison no compiler candidate had been copied into the owner repository or committed. P1 and P3 were subsequently integrated and validated together; see [the combined-source record](access-reuse-merge-20261027.md). The stopped v1 batch was not retried; v2 is a separate, completed batch authorized by the user's correction.
