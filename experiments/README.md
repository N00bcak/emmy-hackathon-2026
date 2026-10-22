# Experiment index

These directories are durable research records. Measurements and source hashes
belong to the experiment that produced them; they are not current claims unless
the status below says so. Large logs and PTX belong in each experiment's ignored
`artifacts/` directory. Compiled cubins are ignored artifacts: manifests retain
their hashes, and the recorded source and commands reproduce them.

| Experiment | Status | Scope |
|---|---|---|
| `001-baseline-evaluator` | Historical | Evaluator contract and initial local runner study. |
| `002-rtx4060-baseline` | Invalid | Initial sweep without a running runner; all cases failed to connect. |
| `003-rtx4060-full` | Historical | First complete local published-case sweep. |
| `004-rtx4060-perf` | Historical | Initial indicative performance observations. |
| `005-rtx4060-audit` | Historical | Compiler/evaluator audit before optimization. |
| `006-optimizer-hoist` | Rejected | Full address-chain hoisting regressed attention; not promoted. |
| `007-optimizer-cse-only` | Historical | Scalar/address CSE candidate evidence. |
| `008-rtx4060-final` | Superseded | Earlier scalar-CSE "final" record at `1c300b0...`. |
| `009-rtx4060-cf-fix` | Historical | Control-flow repair. |
| `010-rtx4060-cf-broaden` | Historical | Broader control-flow lowering. |
| `011-rtx4060-cf-treehalve` | Historical | `TreeHalve` control-flow work. |
| `012-rtx4060-cpasync` | Rejected | First `cp.async` attempt; did not qualify for promotion. |
| `013-rtx4060-cpasync-fixed` | Rejected | Follow-up `cp.async` attempt. |
| `014-rtx4060-cpasync-static` | Superseded | Static `cp.async` candidate at `b3b0a3ac...`. |
| `015-rtx4060-f16-scalar-math` | **Current** | Verified published gate, variants, sanitizer receipts, and paired timing at `6db2dad4...`. |

The current correctness report is [docs/baseline-correctness.md](../docs/baseline-correctness.md).
The research queue and current priorities are in [strategy.md](strategy.md).
