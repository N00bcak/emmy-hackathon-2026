# Commands

Hosted runs used frozen source copies, not the mutable worktree. Both completed
fresh primary runs used the same bound local correctness receipt.

## CPU

```sh
/home/ada/Projects/emmy-hackathon-2026/.venv/bin/python -m unittest discover -s tests -v
```

This worktree has an empty `emmy-hackathon-2026/` directory, so fixture-path
tests error before exercising the pinned challenge files. The full environment
result is retained in `cpu-tests.log`.

```sh
/home/ada/Projects/emmy-hackathon-2026/.venv/bin/python -m unittest tests.test_compiler -v
```

The P4 direct-scale, fail-closed, current-binding/range, and final-SASS tests
pass. Fixture-dependent tests are blocked by the missing submodule checkout;
see `cpu-test-compiler.log`.

## Receipt binding

The harness's own `correctness_binding()` was invoked against
`local-correctness-receipt.json` and the two frozen source copies. It returned
`null` (bound) for both `p02_scale_shift_8m` and `p03_relu_mul_4m`.
The hosted harness also enforced that binding on every run.

## Hosted RTX 5090 primaries

Fresh P02 primary, completed:

```sh
ROOT=/home/ada/Projects/emmy-hackathon-2026
RUN=$ROOT/.emmy/research/performance-v1/p4-hosted-74164ed5-vs-p2-20261027
$ROOT/.venv/bin/python .emmy/harness/hosted_scaled.py \
  --server http://runner.riftstack.ai:48701 \
  --case "$ROOT/.emmy/research/performance-v1/cases-v3/perf-a/p02_scale_shift_8m" \
  --baseline "$RUN/sources/p2_baseline_compile.py" \
  --candidate "$RUN/sources/p4_candidate_compile.py" \
  --local-correctness "$RUN/local_correctness_receipt.json" \
  --out "$RUN/p02-primary-v2" \
  --repetitions 7 --warmup 10 --iters 50 \
  --deadband 0.052508751458576364
```

Fresh P03 primary, completed:

```sh
ROOT=/home/ada/Projects/emmy-hackathon-2026
RUN=$ROOT/.emmy/research/performance-v1/p4-hosted-74164ed5-vs-p2-20261027
$ROOT/.venv/bin/python .emmy/harness/hosted_scaled.py \
  --server http://runner.riftstack.ai:48701 \
  --case "$ROOT/.emmy/research/performance-v1/cases-v3/perf-a/p03_relu_mul_4m" \
  --baseline "$RUN/sources/p2_baseline_compile.py" \
  --candidate "$RUN/sources/p4_candidate_compile.py" \
  --local-correctness "$RUN/local_correctness_receipt.json" \
  --out "$RUN/p03-primary" \
  --repetitions 7 --warmup 10 --iters 50 \
  --deadband 0.06844741235392304
```

At most 16 hosted requests were issued per primary. P02 was run first, then
P03, serially. No confirmation command was run because neither primary
resolved faster. The pre-registered deadbands were not altered or re-pooled.
