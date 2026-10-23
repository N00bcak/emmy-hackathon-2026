# Powenn synchronization — 2026-10-23

The working branch incorporates the evaluated combination of our compiler and
`powenn` (`a3e5ab57e498975205e6a436a2628b5c5b73e49b`). Only `compile.py`,
`tensor_core.py` and `tf32_emitter.py` are transferred from the shortlisted
candidate. The challenge submodule remains `ec8cde1943f88efea3827a95bda6085553d2d407`.
No new test cases were created for this continuation. Existing fixtures,
references, tolerances and the 93-case default are unchanged. No Git commit or
publication was performed.

## What was incorporated, and why

The original is our `51bd61bd…` compiler; merged is `7d5d4d2d…`.
[The manifest](evidence/powenn-sync/manifest.json) binds all three source files,
the comparison arms and source-bound all-15 gates. Timing measures the complete
combination; it does not establish an isolated speedup for every constituent pass.

| Incorporation | Improvement and mechanism | Receipt and qualification |
| --- | --- | --- |
| Metadata `.maxntid` | Gives ptxas the actual block geometry when allocating registers. | All published launches and both-target assembly pass; included in the measured bundle. |
| Cached division/remainder expressions | Reuses versioned index calculations rather than emitting repeated arithmetic. | Existing address/cache and hoisting checks pass; no separate latency claim. |
| Straight-line strided-loop unrolling | Exposes independent loads to the scheduler while preserving each fold's operand order. | Existing reduction/tail checks pass. r02 timing remains inconclusive. The incoming strided vector-reassociation path was not transferred. |
| Affine shared-load vectorization | Proves alignment, combines adjacent loads and computes base addresses outside the loop. | m01 falls from 25.600 to 19.456 µs on sm_89; both initial blocks and independent confirmation agree. |
| Cooperative hoisting, staging and register tiling | Computes repeated scores once per distinct coordinate and shares operands across output columns. | a01 falls from 18.432 to 11.264 µs; a02 from 23.552 to 9.216 µs. Existing normalization, seed, tail and fallback witnesses pass. |
| Loop value numbering | Reuses identical pure bindings and compatible carried folds. The merge includes dtype stamps and rejects unsafe shadowing/alias use. | The previously saved Load-stamp counterexample fails an intermediate candidate and passes the merged source; existing typed and lexical checks pass. |
| Two-pass normalized softmax | Computes the final maximum before accumulating normalized numerator/denominator states. | Enabled only after proving the complete shared-rescaling recurrence and its external uses. The existing prefix-maximum counterexample passes; attention improvements above apply to the complete bundle. |
| Guarded fast exponential | Uses FTZ only in a normal-result range and retains a full-result fallback. | Existing subnormal-rescaling and boundary receipts pass. No separate performance claim. |
| Typed tensor-core recognizer and three-term TF32 emitter | Proves dense contraction/lane ownership, then combines high/high, high/low and low/high products; otherwise preserves SIMT. | Existing positive corpus: published m01 plus 14 generated programs pass. Eight direct/transposed positives actually emit MMA; other positives select SIMT. This is coverage and precision evidence, not a tensor-core speedup claim. |

## What ours does better, and what powenn still does better

The [saved tolerance-based correctness receipts](evidence/powenn-sync/correctness.json)
show the merged source passes every selected case: **93/93**, including all 15
published problems, every existing marked-operator witness and 20 cases requiring
sanitizer checks. These are unchanged cases selected from the already completed
122/122 isolated run, not a newly generated set. The previous wider run retained
26 sanitizer passes. Incoming powenn passes **50/93** in the selected set;
its extra-scope failures are not presented as a challenge score.
The [selected v5 receipt](evidence/powenn-sync/selected-v5-receipts.json) binds
each retained case record, its unchanged tolerances and saved numerical errors,
plus fresh PTX/cubin hashes: all 95 kernels assemble for both targets.

Our scalar ABI, reordered arguments, physical launch indexing, signed address
arithmetic, typed conversions, marked operators, partial collectives and lexical
scope behavior are retained. The old error that required argument declaration
order to equal positional `arg_order` was corrected during condensation; the
merger preserves that correction and the shared-window repair.

[Previously recorded counterexamples](evidence/powenn-sync/counterexamples.json)
explain the semantic guards: an unrestricted running-max rewrite changes an
observable prefix sum from 11 to 16; unconditional FTZ loses an exponential that
becomes visible after rescaling; missing Load dtype stamps can merge distinct
values. They were created before the instruction to stop making new cases and
were not regenerated here. Their extreme-domain examples do not restore archived
stress cases to the default gate.

Powenn's generic `pow` expansion is not imported: `log2(x)` does not implement
negative-base integral powers generally, and a metadata-dependent constant path
is not a proof for runtime inputs. General `pow` remains explicitly unsupported;
we do not claim this is a challenge correctness failure in powenn. Its existing
f16 and scalar-math support is largely already present in ours.

Powenn remains faster on the two published attention examples. Direct independent
comparisons put merged/powenn at **1.222× for a01** and **1.500× for a02**. The
m01 ratio is 1.027× and is small relative to controls. At merge time these measurements did not isolate the cause. A subsequent
[controlled diagnosis](attention-slowdown-cause.md) identifies exponential
lowering: removing our guard recovers powenn latency, and inserting it into
powenn reproduces the slowdown.
The merge preserves the correctness coverage; the measured attention gap remains.

## Performance and validation limits

| Problem | Original µs | Powenn µs | Merged µs | Merged change vs original |
| --- | ---: | ---: | ---: | ---: |
| a01_attention | 18.432 | 9.216 | 11.264 | 38.9% lower latency |
| a02_attention_128 | 23.552 | 6.144 | 9.216 | 60.9% lower latency |
| m01_matmul_512 | 25.600 | 18.432 | 19.456 | 24.0% lower latency |

The [per-problem results](evidence/powenn-sync/performance.csv),
[paired samples](evidence/powenn-sync/paired-samples.csv) and
[block/confirmation summaries](evidence/powenn-sync/timing-blocks.json) retain all
15 outcomes. The other 12 problems have no independently confirmed change beyond
their A/A controls. Initial comparisons use two blocks of 32 pairs, followed by
independent 32-pair confirmation. Timing uses the same frozen published inputs
and evaluator, warm buffers and exclusive admitted GPU execution. Source-bound
all-15 gates precede timing. These timings belong to RTX 4090/sm_89 only.

The existing 184-test candidate run passed. The continuation additionally passed
all **51 existing condensed compiler, selector and oracle tests** on Linux,
including assembly for both sm_89 and sm_120. Source transfer is byte-identical to
the evaluated candidate. No new GPU correctness draw or test case was made.
The [continuation manifest](evidence/powenn-sync/continuation.json) binds the
final receipt files, the 51-test log and clean post-job GPU observations.

The user reports incoming powenn passes all trials on RTX 5090. The merged source
has sm_120 assembly evidence, but no new RTX 5090 execution or timing receipt.
Prior official results for our pre-merge compiler do not qualify this new source.

Remote retained baseline promotion is separate from the working-branch merge:
its immutable `51bd61bd…` baseline and v5 suite remain available for comparisons.
The merged snapshot and its gates live under durable evidence key
`sync-powenn-20261023`; no archive, baseline object or installed source pointer
was overwritten.
