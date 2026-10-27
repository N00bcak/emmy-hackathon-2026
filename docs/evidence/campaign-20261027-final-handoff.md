# 27 October compiler campaign handoff

The owner branch is `n00bcak`. The retained compiler is `compile.py` SHA256
`8ce9868462003fd5d44db861e21cd94165438ceeb5f10eed5544a4a5f0bb2410`,
frozen as research source `bd379dd5df11f3eaec74e525bc449176ac1e5998a7f409336e325115e2cd830a`.
The private research root's authoritative handoff is
`/mnt/nfs/bcy/emmy/research/START.md`, pointing to
`campaigns/campaign-20261027-main-perf-cont-v2/HANDOFF.md`. The bounded run
was closed after the fifth independent speed confirmation, with no active GPU
job and a conservative 89/200 evaluated variants. The user's extended absolute
deadline was 27 October 2026 16:45:43 Singapore time.

Five incremental speed improvements were retained. Their source-bound receipts
are `speed26-f02-promotion.json`, `speed26-f01-promotion.json`,
`speed26-r04-promotion.json`, `speed26-r02-pointer-promotion.json`, and
`speed26-a01-promotion.json` in this directory. Each comparison had both
sources pass all 15 published cases before timing. The current source and
comparison also passed the unchanged fixed 100-case experiment gate. No new
tests or relaxed tolerances were introduced. The 27 October `origin/main`
`perf-v1` suite (commit `40b1321`, archive SHA256
`883655e1a84e8857eab07dcd1065d0afacdad095deb9f62f05914080c28726b4`)
was staged and used on the 4090; only this exact incoming suite had the user's
exception to the 100-test limit.

The fifth improvement retiled a01 so one CTA owns a whole row. The exact
incoming-main `perf-b` a01 512×64 case passed guarded full-output correctness
and sm89/sm120 assembly. Two separate, host-attested 4090 paired runs resolved
a gain; the independent run's block ratios were 0.8800 and 0.87943 against
the prior retained compiler. The smaller `perf-a` a01 256×64 case passed
correctness, but its latency was below the fixed 10× launch floor and remains
unresolved. Other scaled suite cases were not independently paired-timed for
this promotion; do not infer a per-case speed or regression result for them.

The single official grading/calibration batch is spent. The existing budgeted
helper's attested-direct path validated the frozen case and both local runs
without HTTP, then completed exactly nine HTTP attempts. Its result is
`/mnt/nfs/bcy/emmy/research/calibration/official-20261027-a01-retile-v1/result.json`:
all six full-output checks passed on the RTX 5090, and the one-case hosted
median candidate/baseline ratio was 0.90350. Hosted timing is indicative and
does not establish a whole-suite speedup or general 5090 correctness. Do not
repeat or expand this batch.

The 27 October `origin/main` f01 row-map mechanism was adapted to this typed
direct compiler. A separate m01 grouped-K port matched main PTX and passed all
66 GPU witnesses, but failed the unchanged CPU loop-backedge check, so it was
rejected before timing and not retained. See the f01 promotion receipt. Local
Git remained on `n00bcak`; no Git network, remote Git, or publication was used.

On resumption, read the private `START.md`, verify the source and case hashes,
and inspect recorded live jobs before any new work. The five-promotion and
official-batch objective is complete; publication remains owner-controlled.
