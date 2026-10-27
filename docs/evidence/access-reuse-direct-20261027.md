# Direct hosted comparison: access-reuse plan vs n00bcak

The existing `origin/compiler-design/access-reuse-plan` compiler was submitted directly against the current `n00bcak` compiler on the official submission endpoint's RTX 5090 (`sm_120`). Both arms used identical frozen IR, input, expected-output, metadata, warmup (5), and timing iterations (20). The single bounded batch made 25 HTTP attempts, including health, and completed without service errors. Three alternating A/B pairs per case followed a baseline A/A pair. The endpoint comparison used no newly authored tests.

| Existing perf case | Access-reuse / n00bcak median | A/B pair ratios | Result |
| --- | ---: | --- | --- |
| `p03_relu_mul_4m` | 0.8186 | 0.8186, 0.8894, 0.8159 | Faster; both full outputs passed exactly |
| `f02_row_softmax_3072x2048` | 0.7591 | 0.7591, 0.7575, 0.7598 | Faster; hosted output truncated at the runner's return limit |
| `a01_attention_512x64` | 3.5911 | 3.5911, 3.5914, 3.5809 | Slower; both full outputs passed |

The A/A ratios were 0.9985, 0.9981, and 1.0000 respectively. All three A/B pairs clear the predeclared 2% noise margin plus 1% decision buffer in the stated direction. These are per-case latency observations, not a claim about other shapes or hidden cases. The branch has source-bound full-output f02 receipts at its exact compiler hash in `docs/evidence/p2-shared-reuse/` and `docs/evidence/p3-sibling-fusion/`, but the hosted f02 call could not return all 6,291,456 output values.

The measured compiler SHA-256 identities were `8ce9868462003fd5d44db861e21cd94165438ceeb5f10eed5544a4a5f0bb2410` (`n00bcak`) and `1f220a4dabcc436c49ab933bcc505e0e4d6ad1e04aef347a14a0b6cbd2fd0d1d` (access-reuse plan). Exact case, IR, PTX and source hashes are in [compile manifest](access-reuse-direct-compile-20261027.json); all timing, checks, attempt count, and endpoint GPU identity are in [result](access-reuse-direct-result-20261027.json). The raw attempt journal and frozen inputs are under `/private/tmp/access-reuse-hosted-compare/hosted-direct-v1/` and `/private/tmp/access-reuse-hosted-compare/cases/` on this machine.

**Disposition at the time of comparison:** Do not replace the current compiler with the branch wholesale; attention regresses by about 3.59×. P1's four-wide pointwise access and P2's second-pass softmax row cache were candidates for selective porting. The branches differed by 4,608 lines in `compile.py`; a port would be a new source and need its own source-bound all-15 published gate before any timing. No compiler code was changed by this comparison. P1 and a separate bounded P3 mechanism were subsequently integrated; see [the combined-source record](access-reuse-merge-20261027.md).
