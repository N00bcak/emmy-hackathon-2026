# Current strategy

Confirmed target: direct Kernel IR to PTX. Upstream remains pinned in the
parent repository, and no existing renderer/compiler performs our lowering.

## Current baseline and implemented increment

The pre-tensor-core compiler source `compile.py` has SHA-256
`6db2dad4944d87ae46194fd4deaf9331327cda21b21adaa434a1db41f515e6ba`. It emits
PTX 8.0 for `sm_89` and passes all 15 published problems at their metadata
tolerances on the local RTX 4060 Laptop GPU. The frozen gate, 31 compiler tests,
18 typed GPU variants, sanitizer receipts, and paired timings are recorded in
`015-rtx4060-f16-scalar-math`; the authoritative summary is
[docs/baseline-correctness.md](../docs/baseline-correctness.md). This does not
establish hidden-set correctness or RTX 5090 behavior.

The current compiler adds a three-term TF32 `mma.sync` path for proved
one-atom-per-warp matmul plans. m01 and 14 independently generated corpus
`Plan` cases pass independent CPU references and local `sm_89` execution; all
15 published cases pass, and every non-`Plan` case keeps its frozen SIMT PTX
bytes. m02 remains `UnsupportedMatch` because four atoms do not match eight
warps. The source hashes, corpus, assembly/SASS checks, execution logs, and
audit dispositions are in
[the tensor-core evidence bundle](tensor-core-v1/README.md). No tensor-core
timing, speedup, or promotion claim is made.

The only clear local speed result is `m01_matmul_512` at median `0.4893x` the
frozen pre-CSE baseline. `m02_matmul_rect` is small and noisy (`0.9831x` here,
`0.9453x` in `014`) and is not claimed as stable. `f02_row_softmax`,
`a01_attention`, and `a02_attention_128` are neutral within timer resolution and
matching A/A controls. These are indicative paired diagnostics, not an
evaluator-issued performance qualification.

## Order of work

Correctness remains ahead of performance. The published gate is necessary but
not sufficient: hidden cases vary shapes and op mixes within the five challenge
categories. The next work should be driven by failures or gaps in the unstarred
Kernel IR surface documented in `TASK.md`, not by every constructible starred
form. The TF32 matmul path is implemented and correctness-gated but not timed;
attention, TMA, matrix transport, and `wgmma` remain out of scope as recorded in
[docs/tensor-core-plan.md](../docs/tensor-core-plan.md).

Optimization proceeds concurrently with scope broadening only through
generality-preserving typed-IR analysis and emission. A candidate must first
pass all 15 published cases at unchanged tolerances; only then may it be timed.
Keep new fast paths behind structural predicates with a correct generic fallback,
and never infer a bottleneck from PTX length alone. Rejected experiments stay in
this directory as evidence rather than being reused as current claims.

## Validation and research loop

Run the compiler suite and the complete published gate after any source change;
a partial sweep diagnoses failures but cannot qualify performance. Use isolated
GPU batches for typed variants, and require fresh sanitizer confirmation for any
atomic path. Timing is exclusive, paired, case-specific, and A/A-controlled.

Discovery remains opt-in. When requested, the coordinator owns generators and
oracles, workers receive distinct questions and isolated workspaces, and all
source, inputs, references, commands, identities, measurements, failures, and
resumable handoffs are stored under the configured durable research root. Local
Git history belongs to the owner; remote GPU workspaces are plain snapshots.
