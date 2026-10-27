# P4 hosted timing preregistration

- Baseline: accepted P2 source `0a6f12033a942f4da9176ebf1e963c16193d8883610d3d4807bbc9870ab0de2a`.
- Candidate: frozen P4 source `74164ed51aff8fc3886e4b5daa029f526d26d4fe5e2d0cf86ac706ec1cfb602d`.
- Cases: scaled `p02_scale_shift_8m` and `p03_relu_mul_4m`, `perf-a` tier.
- Harness: `.emmy/harness/hosted_scaled.py`, identical flags, sm_120, candidate-vs-baseline.
- Pre-registered case-specific deadbands (maximum of prior hosted A/A noise and 1%):
  - p02: 0.052508751458576364 (prior A/A max absolute bias).
  - p03: 0.06844741235392304 (prior A/A max absolute bias).
- Primary comparison: 7 alternating paired blocks, warmup 10, 50 iterations per block.
- Decision: candidate-faster on a primary requires median ratio < 1 - deadband; regression requires median > 1 + deadband; otherwise unresolved.
- Confirmation: run isolated confirmation only for a resolved candidate-faster case using a fresh output directory and the same preregistered deadband.
- Correctness binding: `local_correctness_receipt.json`, derived from the frozen local full-output receipt for `74164ed5...`; both p02 and p03 must pass full outputs.
- No hosted correctness, hidden-set, or streaming-throughput claim.
