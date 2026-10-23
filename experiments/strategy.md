# Current strategy

The next task is to diagnose the official RTX 5090 `m01_matmul_512` failure using
the saved exact request, PTX, input/reference arrays and launch contract. The
retained compiler passes all 15 published cases on RTX 4090 and 14/15 at the
official endpoint. Do not qualify RTX 5090 performance until its all-15 gate passes.

Use the synchronized 73-case routine suite (15 published and 58 generated), plus
targeted archived expansion and independently generated cases for the changed
mechanism. Repeated seeds and shapes do not accumulate into the default suite.
Keep general typed-IR lowering and unchanged ABI, tolerances and sanitizer rules.

Discovery is closed. Reading these records starts no workers, GPU jobs or endpoint
requests. Resume candidates require explicit activation. Exact source identities,
receipts, retained/rejected results and the next check are in
[the handoff](../docs/HANDOFF.md). Remain on `n00bcak`; no publication or remote Git.
