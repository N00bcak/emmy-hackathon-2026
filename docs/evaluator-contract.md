# Local evaluation and observed hosted behavior

The local runner replicates the documented execution protocol for the tested
requests. It is not a copy of the hosted server's source or proof of exact backend
parity. The endpoint is an execution service; expected-output scoring belongs to
the challenge client.

## Evaluation pipeline

1. The client reads problem metadata and expected.npz, and invokes our compiler
   for each IR. Compiler stdout must contain only PTX.
2. `/compile` assembles that PTX; its result supplies the entry names, register
   diagnostics and a content-addressed cubin handle. Identical compile requests
   return a stable handle. The documented cache also depends on assembler version.
3. `/run` receives positional arguments and metadata launch geometry. Intermediate
   results, when present, are passed by the client to later kernels in order.
4. The client rejects truncated results and compares complete outputs to the
   expected arrays with the per-problem rtol/atol. Neither server reads expected.npz.

The hosted GPU is an RTX 5090; the local GPU is a different model. The compiler
emits PTX 8.0 targeting sm_89; each runner assembles for its own selected target.
A 4090 latency cannot be treated as the corresponding 5090 latency.

The target also changes the representation of dynamic shared memory. On
`sm_120`, `ptxas` materializes a nonzero shared-window base for an
`.extern .shared` slab; on `sm_89` the same base folds to zero. A `cp.async`
destination must therefore be `low32(slab) + (destination - slab)`, not the
destination's offset alone. Subtracting only the base passed every local
sm_89 case but made `m01_matmul_512` fail on the hosted sm_120 runner. The
corrected form passes m01 on both the local sm_89 runner and the hosted sm_120
runner; this is execution evidence for that case, not a hidden-set guarantee.

## Benign contract checks

A bounded study made **16 hosted HTTP requests**, including health and OpenAPI.
It used the locally GPU-verified pointwise and reduction compiler outputs, plus
locally assembled one-thread scalar and atomic probes. All checks passed:

- The two compiler outputs matched fixture expectations within their tolerances.
- Repeated identical compile requests returned the same cubin_id.
- A seeded f32 input matched `default_rng(seed).standard_normal(shape).astype(float32)`.
- Exact input data and scalar-by-value arguments were preserved in the tested cases.
- Truncation returned a prefix while retaining the full shape and full-array sum.
  In the tested f32 case the sum matched native float32 NumPy accumulation, not
  float64 accumulation. This observation does not establish every dtype's sum rules.
- Two warmups plus three iterations of a one-thread increment produced five.
  Naming that output in zero_outputs produced one. A subsequent request produced
  five again, establishing fresh output initialization in the tested case.
- Omitting repetition counts produced 25 increments and reported iters=20,
  consistent with OpenAPI defaults warmup=5/iters=20. The upstream client overrides
  these defaults with warmup=0/iters=1.

The same checks run against the local implementation. Raw requests/responses are
retained privately with the experiment artifacts; summary findings are recorded
in experiments/001-baseline-evaluator/reflection.md.

## Local usage

Run GPU commands in a CUDA-capable evaluation environment through its GPU
admission mechanism. The public tool has no host-specific access information
and performs no GPU admission itself. Start-and-cleanup test helpers stay within one admitted
job, keep the service on loopback, and stop it before returning:

```sh
python tools/check_local.py --compiler compile.py --out "$EMMY_JOB_DIR/local-check"
python tests/test_compiler.py --generate-fixtures "$EMMY_JOB_DIR/synthetic"
python tools/check_contract.py --compiler compile.py --out "$EMMY_JOB_DIR/contract-check" --synthetic-dir "$EMMY_JOB_DIR/synthetic"
```

Set `EMMY_JOB_DIR` to a per-job artifact directory, manually or through your
runtime launcher. The first
helper exercises all 15 published problems and a reordered scale/shift ABI.
It performs correctness checks by default. Add `--timing` only when indicative
latencies are wanted; timing is refused unless the same source passed all 15.
Using `--problem` for a partial diagnostic sweep does not satisfy this gate. Assembly tests require `ptxas` on PATH:

```sh
python -m unittest discover -s tests -v
```

For an already running local service, the unchanged challenge client works with
`--server http://127.0.0.1:48702`. tools/evaluate.py now defaults to that local
address and uses file hashes rather than Git commands for provenance. It fails
rather than silently falling back to the public endpoint. Its result records whether all 15 published problems passed for unchanged
source. Partial passes are debugging evidence, never benchmark qualification.

The reusable local server has a content-addressed assembly cache and a fresh CUDA
subprocess per run, with bounded worker lifetime and serialized GPU execution.
It supports positional pointers/scalars, all documented argument dtypes, output
resets, and capped output return; GPU conformance tests currently cover f32.
CPU tests cover validation, caching, error stages and worker timeout cleanup.

## Limits and things not established

No numeric rate quota is published; do not infer unlimited capacity. Use one
public request at a time and stop on busy. No saturation, deliberate GPU faults,
private cases, scans or filesystem probes were used.

The local service uses CUDA events around each launch and returns additional
samples_us diagnostics. It excludes copies/reset/context setup from that interval.
The hosted timing implementation, clock policy, caching lifetime, internal process
pool, exact validation behavior for every possible request, and hidden-set scoring
are not established by this study. Hosted timings are documented as indicative.
Local compile IDs are local handles, not interchangeable with hosted IDs.

The local service is a trusted loopback research tool, not a sandbox for hostile
PTX. Its request validation is intentionally strict and has an additional 96 MiB
HTTP-body cap; malformed-request responses need not match the hosted service's
Pydantic details. Neither success on open cases nor the simulator-free local
execution proves hidden-set correctness or universal race freedom.
